# -*- coding: utf-8 -*-
"""
LinkMe 全流程在线 SSVEP 识别

流程：
1. 使用 LinkMe 采集的 BDF 文件训练 TDCA 模型；
2. BDF 前 8 路为脑电，HardWareAnnotion 为硬件标签通道；
3. 在线通过 COM16 + LinkMe.dll 接收 8 路脑电和 1 路 trigger；
4. 刺激程序通过独立打标口（例如 COM15）向 LinkMe 发送 1~7 单字节标签；
5. MetaBCI Marker 按 trigger 截取试次并在线分类；
6. 分类结果通过 UDP 和 LSL 输出。

注意：
- 厂商采集软件不能同时占用 COM16；
- LinkMe.dll 位数必须和 Python 位数一致；
- 离线训练 BDF 必须由 LinkMe 采集，并包含 HardWareAnnotion 通道；
- 离线和在线的通道顺序必须一致。
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
import serial
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet

from metabci.brainflow.amplifiers import BaseAmplifier, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition import FBTDCA
from metabci.brainda.algorithms.decomposition.base import (
    generate_cca_references,
    generate_filterbank,
)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut,
)
from metabci.brainda.utils import upper_ch_names


# =============================================================================
# 用户配置：先修改这里
# =============================================================================

# LinkMe 在线参数
SRATE = 1000
LINKME_DATA_PORT = "COM16"
LINKME_BAUDRATE = 460800
LINKME_DLL_PATH = Path(r"F:\WheelchairControl\MetaBCI-NCL\MetaBCI6\demos\brainflow_demos\LinkMe.dll")

# 新采集的 LinkMe 训练数据。可以添加多个 BDF 文件。
RUN_FILES = [
    r"F:\1.bdf",
]

# LinkMe BDF 的硬件标签通道名称。
BDF_TRIGGER_CHANNEL = "HardWareAnnotion"

# LinkMe 8 路脑电顺序。必须与 BDF 和 LinkMe.dll 输出顺序一致。
LINKME_CHANNEL_NAMES = [
    "PQZ",
    "PO3",
    "PO7",
    "PO4",
    "PO8",
    "OZ",
    "O1",
    "O2",
]

# 训练和在线识别使用的通道。建议先全部使用。
PICK_CHANNELS = LINKME_CHANNEL_NAMES.copy()

# 7 类标签及对应频率。必须与刺激程序顺序一致。
STIM_LABELS = [1, 2, 3, 4, 5, 6, 7]
STIM_FREQS = np.array([10, 11, 12, 13, 14, 15, 16], dtype=float)

# 刺激开始后 0.14~2.14 秒的数据用于识别，共 2 秒。
STIM_INTERVAL = [0.14, 2.14]

# 识别结果发送给刺激程序。
UDP_FEEDBACK_ADDRESS = ("192.168.3.15", 6000)

# LSL 配置。
LSL_SOURCE_ID = "meta_online_worker"
FEEDBACK_WORKER_NAME = "feedback_worker"
WAIT_FOR_LSL_CONSUMER = False

# 先跑通时建议 False；需要查看离线准确率时改成 True。
RUN_OFFLINE_VALIDATION = False

# LinkMe.dll 输出的脑电通常可直接作为微伏使用。
# 如果在线脑电幅值比 BDF 训练数据小约 1e6 倍，再改成 1e6。
ONLINE_EEG_SCALE = 1.0

# 启动和调试参数。
WORKER_STARTUP_TIMEOUT = 300
PRINT_STREAM_STATUS = True


# =============================================================================
# 通用函数
# =============================================================================


def encode_labels(y: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """把原始标签 1~7 转换成模型类别 0~6。"""
    label_to_index = {int(label): i for i, label in enumerate(labels)}
    encoded = np.empty(len(y), dtype=np.int64)

    for i, value in enumerate(y):
        value_int = int(value)
        if value_int not in label_to_index:
            raise ValueError(f"未知标签：{value_int}")
        encoded[i] = label_to_index[value_int]

    return encoded


def safe_standardize(X: np.ndarray) -> np.ndarray:
    """每个试次去均值并标准化，避免除零。"""
    X = X - np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=(-1, -2), keepdims=True)
    std[std < 1e-12] = 1.0
    return X / std


# =============================================================================
# LinkMe 在线数据接收
# =============================================================================


class LinkMeAmplifier(BaseAmplifier):
    """COM16 原始数据 -> LinkMe.dll -> [8 EEG + 1 trigger]。"""

    EEG_CHANNELS = 8
    OUTPUT_COLUMNS = 9

    def __init__(
        self,
        port: str,
        dll_path: str | Path,
        srate: int = 1000,
        baudrate: int = 460800,
        eeg_scale: float = 1.0,
    ):
        super().__init__()

        self.port_name = port
        self.dll_path = Path(dll_path).expanduser().resolve()
        self.srate = int(srate)
        self.baudrate = int(baudrate)
        self.eeg_scale = float(eeg_scale)

        self.serial_port: serial.Serial | None = None
        self.dll = self._load_dll()
        self._configure_dll()

        self.raw_bytes_total = 0
        self.decoded_points_total = 0
        self.last_status_time = time.monotonic()
        self.last_status_bytes = 0
        self.last_status_points = 0

    def _load_dll(self):
        if not self.dll_path.exists():
            raise FileNotFoundError(f"找不到 LinkMe.dll：{self.dll_path}")

        try:
            return ctypes.CDLL(str(self.dll_path))
        except OSError as exc:
            raise OSError(
                f"LinkMe.dll 加载失败：{self.dll_path}\n"
                "请确认 Python 与 DLL 位数一致。"
            ) from exc

    def _configure_dll(self) -> None:
        self.dll.dataProtocol.argtypes = (
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        )
        self.dll.dataProtocol.restype = ctypes.c_int

        self.dll.getData.argtypes = (
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        )
        self.dll.getData.restype = ctypes.c_int

        if hasattr(self.dll, "setFS"):
            self.dll.setFS.argtypes = (ctypes.c_int,)
            self.dll.setFS.restype = None

    def connect_serial(self) -> None:
        if self.serial_port is not None and self.serial_port.is_open:
            return

        self.serial_port = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.02,
            write_timeout=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

        # 与厂家有线示例一致。
        self.serial_port.dtr = True
        self.serial_port.rts = True
        self.serial_port.reset_input_buffer()

        if hasattr(self.dll, "setFS"):
            self.dll.setFS(self.srate)

        print(
            f"LinkMe 已连接：{self.port_name}，"
            f"{self.baudrate} baud，{self.srate} Hz",
            flush=True,
        )

    def recv(self):
        """返回 [n_samples, 9]，最后一列为 trigger。"""
        if self.serial_port is None or not self.serial_port.is_open:
            return []

        first_byte = self.serial_port.read(1)
        if not first_byte:
            return []

        waiting = self.serial_port.in_waiting
        raw_bytes = first_byte
        if waiting > 0:
            raw_bytes += self.serial_port.read(waiting)

        self.raw_bytes_total += len(raw_bytes)

        input_buffer = (
            ctypes.c_ubyte * len(raw_bytes)
        ).from_buffer_copy(raw_bytes)

        data_size = int(
            self.dll.dataProtocol(input_buffer, len(raw_bytes))
        )
        if data_size <= 0:
            self._print_stream_status()
            return []

        # 留足缓冲，防止 DLL 一次吐出较多数据点。
        max_rows = max(512, data_size + 64)
        output_buffer = (
            ctypes.c_double * (max_rows * self.OUTPUT_COLUMNS)
        )()

        real_rows = int(
            self.dll.getData(
                output_buffer,
                max_rows,
                self.OUTPUT_COLUMNS,
            )
        )

        if real_rows <= 0:
            if real_rows == -1:
                print("LinkMe getData 缓冲区不足。", flush=True)
            self._print_stream_status()
            return []

        if real_rows > max_rows:
            print(
                f"LinkMe 返回行数异常：{real_rows} > {max_rows}",
                flush=True,
            )
            return []

        data = np.ctypeslib.as_array(output_buffer)
        data = data.reshape(max_rows, self.OUTPUT_COLUMNS)
        data = data[:real_rows].copy()

        # 前 8 列脑电，第 9 列标签。
        data[:, : self.EEG_CHANNELS] *= self.eeg_scale
        data[:, 8] = np.rint(data[:, 8])
        self.decoded_points_total += real_rows

        triggers = data[:, 8].astype(np.int64)
        nonzero = triggers[triggers != 0]
        if nonzero.size:
            print(
                "[在线触发] "
                + ", ".join(map(str, nonzero.tolist())),
                flush=True,
            )

        self._print_stream_status()
        return data.tolist()

    def _print_stream_status(self) -> None:
        if not PRINT_STREAM_STATUS:
            return

        now = time.monotonic()
        dt = now - self.last_status_time
        if dt < 1.0:
            return

        byte_rate = (
            self.raw_bytes_total - self.last_status_bytes
        ) / dt
        point_rate = (
            self.decoded_points_total - self.last_status_points
        ) / dt

        # print(
        #     f"[数据流] 原始={byte_rate:.0f} B/s，"
        #     f"解包={point_rate:.1f} 点/s，"
        #     f"累计={self.decoded_points_total}",
        #     flush=True,
        # )

        self.last_status_time = now
        self.last_status_bytes = self.raw_bytes_total
        self.last_status_points = self.decoded_points_total

    def _inner_loop(self) -> None:
        """覆盖默认循环，让在线异常直接显示。"""
        self._exit.clear()

        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except Exception:
                print("[LinkMe 在线循环异常]", flush=True)
                traceback.print_exc()
                time.sleep(0.05)

    def start_trans(self) -> None:
        self.connect_serial()
        self.start()

    def stop_trans(self) -> None:
        try:
            self.stop()
        finally:
            if self.serial_port is not None and self.serial_port.is_open:
                self.serial_port.close()
            print("LinkMe 已停止", flush=True)


# =============================================================================
# LinkMe BDF 训练数据读取
# =============================================================================


def find_trigger_channel(
    raw: mne.io.BaseRaw,
    wanted_name: str,
) -> str:
    name_map = {name.upper(): name for name in raw.ch_names}
    wanted_upper = wanted_name.upper()

    if wanted_upper in name_map:
        return name_map[wanted_upper]

    # 兼容厂家通道拼写差异。
    keywords = (
        "HARDWAREANNOTION",
        "HARDWAREANNOTATION",
        "TRIGGER",
        "STATUS",
    )

    for name in raw.ch_names:
        upper_name = name.upper()
        if any(keyword in upper_name for keyword in keywords):
            return name

    raise ValueError(
        f"没有找到标签通道 {wanted_name!r}。\n"
        f"BDF 实际通道：{raw.ch_names}"
    )


def extract_events_from_trigger(
    raw: mne.io.BaseRaw,
    trigger_channel: str,
) -> np.ndarray:
    """优先使用 MNE，失败时直接从标签通道提取非零起点。"""
    events = mne.find_events(
        raw,
        stim_channel=trigger_channel,
        output="onset",
        consecutive=True,
        shortest_event=1,
        initial_event=True,
        verbose=False,
    )

    if len(events) > 0:
        return events.astype(np.int64)

    trigger = raw.get_data(picks=[trigger_channel])[0]
    trigger = np.rint(trigger).astype(np.int64)

    if trigger.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    changes = np.flatnonzero(np.diff(trigger) != 0) + 1
    if trigger[0] != 0:
        changes = np.concatenate(([0], changes))

    onsets = changes[trigger[changes] != 0]
    if len(onsets) == 0:
        return np.empty((0, 3), dtype=np.int64)

    previous = np.zeros(len(onsets), dtype=np.int64)
    valid = onsets > 0
    previous[valid] = trigger[onsets[valid] - 1]

    return np.column_stack(
        [
            onsets + int(raw.first_samp),
            previous,
            trigger[onsets],
        ]
    ).astype(np.int64)


def read_linkme_bdf(
    run_files: Sequence[str],
    pick_channels: Sequence[str],
    interval: Sequence[float],
    labels: Sequence[int],
    trigger_channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 LinkMe BDF 并生成 [trial, channel, time] 训练数据。"""
    mne.set_log_level(verbose=False)

    requested_channels = [name.upper() for name in pick_channels]
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for filename in run_files:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"训练文件不存在：{path}\n"
                "请把 RUN_FILES 改成新采集的 LinkMe BDF 路径。"
            )

        print(f"读取 LinkMe 训练文件：{path}", flush=True)

        probe = mne.io.read_raw_bdf(
            str(path),
            preload=False,
            stim_channel=None,
            verbose=False,
        )
        try:
            actual_trigger_original = find_trigger_channel(
                probe,
                trigger_channel,
            )
            print(
                f"标签通道：{actual_trigger_original}",
                flush=True,
            )
        finally:
            probe.close()

        # 明确指定 stim_channel，避免标签被当作脑电电压缩放。
        raw = mne.io.read_raw_bdf(
            str(path),
            preload=True,
            stim_channel=actual_trigger_original,
            verbose=False,
        )
        raw = upper_ch_names(raw)
        actual_trigger = actual_trigger_original.upper()

        missing_channels = [
            name for name in requested_channels if name not in raw.ch_names
        ]
        if missing_channels:
            raise ValueError(
                f"{path} 缺少脑电通道：{missing_channels}\n"
                f"实际通道：{raw.ch_names}"
            )

        events = extract_events_from_trigger(raw, actual_trigger)
        if len(events) == 0:
            trigger_data = raw.get_data(picks=[actual_trigger])[0]
            values = np.unique(np.rint(trigger_data).astype(np.int64))
            raise ValueError(
                f"{path} 的 {actual_trigger} 中没有检测到标签。\n"
                f"通道数值：{values[:50].tolist()}"
            )

        all_event_values, all_event_counts = np.unique(
            events[:, 2],
            return_counts=True,
        )
        print(
            "BDF 全部标签：",
            {
                int(value): int(count)
                for value, count in zip(
                    all_event_values,
                    all_event_counts,
                )
            },
            flush=True,
        )

        events = events[np.isin(events[:, 2], labels)]
        if len(events) == 0:
            raise ValueError(
                f"{path} 没有 1~7 训练标签。"
            )

        values, counts = np.unique(events[:, 2], return_counts=True)
        label_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
        }
        print("用于训练的标签：", label_counts, flush=True)

        missing_labels = [
            int(label)
            for label in labels
            if int(label) not in label_counts
        ]
        if missing_labels:
            raise ValueError(
                f"训练数据缺少标签：{missing_labels}\n"
                f"当前统计：{label_counts}"
            )

        channel_picks = mne.pick_channels(
            raw.ch_names,
            requested_channels,
            ordered=True,
        )

        event_id = {str(label): int(label) for label in labels}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=float(interval[0]),
            tmax=float(interval[1]),
            picks=channel_picks,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=False,
        )

        for label in labels:
            key = str(label)
            if key not in epochs.event_id or len(epochs[key]) == 0:
                raise ValueError(f"标签 {label} 没有可用试次。")

            # MNE 中脑电单位为 V，转换为 µV；去掉边界多出的一个采样点。
            X = epochs[key].get_data()[..., 1:] * 1e6
            y = np.full(len(X), int(label), dtype=np.int64)

            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    y = encode_labels(y_raw, labels)

    print(
        f"训练数据：X={X.shape}，y={y.shape}，"
        f"类别={np.unique(y).tolist()}",
        flush=True,
    )

    return X, y


# =============================================================================
# TDCA 模型
# =============================================================================


def train_tdca(
    X: np.ndarray,
    y: np.ndarray,
    srate: int,
):
    """训练 7 类 FBTDCA。"""
    if len(STIM_FREQS) != len(STIM_LABELS):
        raise ValueError("STIM_FREQS 数量必须与 STIM_LABELS 一致。")

    X = resample(X, up=250, down=srate)
    X = safe_standardize(X)
    y = np.asarray(y).reshape(-1)

    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
    weights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)

    references = generate_cca_references(
        STIM_FREQS,
        srate=250,
        T=1,
        n_harmonics=1,
    )

    model = FBTDCA(
        filterbank,
        padding_len=2,
        n_components=1,
        filterweights=np.asarray(weights),
    )

    return model.fit(X, y, references)


def predict_tdca(
    X: np.ndarray,
    model,
    srate: int,
) -> np.ndarray:
    """X 可以是 [channel, time] 或 [trial, channel, time]。"""
    X = np.asarray(X, dtype=np.float64)
    X = X.reshape((-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=250, down=srate)
    X = safe_standardize(X)
    return model.predict(X)


def validate_tdca(
    X: np.ndarray,
    y: np.ndarray,
    srate: int,
) -> float:
    """可选离线交叉验证。"""
    y = np.asarray(y).reshape(-1)
    splitter = EnhancedLeaveOneGroupOut(return_validate=False)
    accuracies: list[float] = []

    for train_index, test_index in splitter.split(X, y=y):
        model = train_tdca(
            np.copy(X[train_index]),
            np.copy(y[train_index]),
            srate,
        )
        predicted = predict_tdca(
            np.copy(X[test_index]),
            model,
            srate,
        )
        accuracies.append(float(np.mean(predicted == y[test_index])))

    accuracy = float(np.mean(accuracies))
    print(f"离线交叉验证准确率：{accuracy:.4f}", flush=True)
    return accuracy


# =============================================================================
# 在线分类 Worker
# =============================================================================


class FeedbackWorker(ProcessWorker):
    def __init__(
        self,
        run_files,
        pick_channels,
        online_channel_names,
        trigger_channel,
        stim_interval,
        stim_labels,
        srate,
        lsl_source_id,
        udp_address,
        timeout,
        worker_name,
    ):
        self.run_files = list(run_files)
        self.pick_channels = [name.upper() for name in pick_channels]
        self.online_channel_names = [
            name.upper() for name in online_channel_names
        ]
        self.trigger_channel = trigger_channel
        self.stim_interval = list(stim_interval)
        self.stim_labels = list(stim_labels)
        self.srate = int(srate)
        self.lsl_source_id = lsl_source_id
        self.udp_address = udp_address

        self.ready_event = mp.Event()
        self.startup_error_queue = mp.Queue()

        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        try:
            print("[Worker] 读取 LinkMe BDF……", flush=True)
            X, y = read_linkme_bdf(
                run_files=self.run_files,
                pick_channels=self.pick_channels,
                interval=self.stim_interval,
                labels=self.stim_labels,
                trigger_channel=self.trigger_channel,
            )

            missing_online = [
                name
                for name in self.pick_channels
                if name not in self.online_channel_names
            ]
            if missing_online:
                raise ValueError(
                    f"在线通道表缺少：{missing_online}\n"
                    f"LINKME_CHANNEL_NAMES={self.online_channel_names}"
                )

            self.online_indices = np.asarray(
                [
                    self.online_channel_names.index(name)
                    for name in self.pick_channels
                ],
                dtype=int,
            )

            if RUN_OFFLINE_VALIDATION:
                validate_tdca(X, y, self.srate)

            print("[Worker] 训练最终 TDCA 模型……", flush=True)
            self.model = train_tdca(X, y, self.srate)

            info = StreamInfo(
                name="meta_feedback",
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="int32",
                source_id=self.lsl_source_id,
            )
            self.outlet = StreamOutlet(info)
            self.udp_socket = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )

            if WAIT_FOR_LSL_CONSUMER:
                print("[Worker] 等待 LSL 消费者……", flush=True)
                while not self._exit.is_set():
                    if self.outlet.wait_for_consumers(1e-3):
                        break

            self.ready_event.set()
            print("[Worker] 在线分类器准备完成", flush=True)

        except BaseException:
            error_text = traceback.format_exc()
            try:
                self.startup_error_queue.put(error_text)
            except Exception:
                pass

            self.ready_event.set()
            print("[Worker 启动失败]", flush=True)
            print(error_text, flush=True)
            raise

    def get_startup_error(self):
        try:
            return self.startup_error_queue.get_nowait()
        except queue.Empty:
            return None

    def consume(self, data):
        trial = np.asarray(data, dtype=np.float64)
        print(f"[Worker] 收到试次：{trial.shape}", flush=True)

        if trial.ndim != 2 or trial.shape[1] < 9:
            print(
                f"在线数据形状错误：{trial.shape}，"
                "应为 [time, 9]。",
                flush=True,
            )
            return

        # [time, 9] -> [9, time] -> 选择脑电通道。
        trial = trial.T
        eeg = trial[self.online_indices]

        predicted = predict_tdca(
            eeg,
            model=self.model,
            srate=self.srate,
        )

        # 模型输出 0~6，反馈标签转换回 1~7。
        pred_label = int(np.asarray(predicted).reshape(-1)[0]) + 1

        self.udp_socket.sendto(
            str(pred_label).encode("ascii"),
            self.udp_address,
        )

        print(f"在线识别结果：{pred_label}", flush=True)

        if self.outlet.have_consumers():
            self.outlet.push_sample([pred_label])

    def post(self):
        udp_socket = getattr(self, "udp_socket", None)
        if udp_socket is not None:
            udp_socket.close()


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    if len(LINKME_CHANNEL_NAMES) != 8:
        raise ValueError("LINKME_CHANNEL_NAMES 必须恰好包含 8 个通道。")

    if len(STIM_LABELS) != len(STIM_FREQS):
        raise ValueError("标签数量与频率数量不一致。")

    worker = FeedbackWorker(
        run_files=RUN_FILES,
        pick_channels=PICK_CHANNELS,
        online_channel_names=LINKME_CHANNEL_NAMES,
        trigger_channel=BDF_TRIGGER_CHANNEL,
        stim_interval=STIM_INTERVAL,
        stim_labels=STIM_LABELS,
        srate=SRATE,
        lsl_source_id=LSL_SOURCE_ID,
        udp_address=UDP_FEEDBACK_ADDRESS,
        timeout=5e-2,
        worker_name=FEEDBACK_WORKER_NAME,
    )

    marker = Marker(
        interval=STIM_INTERVAL,
        srate=SRATE,
        events=STIM_LABELS,
    )

    amplifier = LinkMeAmplifier(
        port=LINKME_DATA_PORT,
        dll_path=LINKME_DLL_PATH,
        srate=SRATE,
        baudrate=LINKME_BAUDRATE,
        eeg_scale=ONLINE_EEG_SCALE,
    )

    amplifier.register_worker(
        FEEDBACK_WORKER_NAME,
        worker,
        marker,
    )
    amplifier.up_worker(FEEDBACK_WORKER_NAME)

    print("等待 LinkMe 离线模型训练完成……", flush=True)

    if not worker.ready_event.wait(WORKER_STARTUP_TIMEOUT):
        raise TimeoutError(
            f"Worker 在 {WORKER_STARTUP_TIMEOUT} 秒内未准备完成。"
        )

    startup_error = worker.get_startup_error()
    if startup_error:
        raise RuntimeError("Worker 启动失败：\n" + startup_error)

    if not worker.is_alive():
        raise RuntimeError(
            f"Worker 已退出，exitcode={worker.exitcode}。"
        )

    try:
        amplifier.start_trans()
        print(
            "LinkMe 在线识别已启动。\n"
            "程序会截取 0.14~2.14 秒数据并输出一次分类结果。",
            flush=True,
        )
        input("按回车键退出。\n")

    except KeyboardInterrupt:
        print("收到停止指令。", flush=True)

    finally:
        try:
            amplifier.down_worker(FEEDBACK_WORKER_NAME)
        except Exception as exc:
            print("停止 Worker 提示：", exc, flush=True)

        time.sleep(0.5)

        try:
            amplifier.stop_trans()
        except Exception as exc:
            print("停止 LinkMe 提示：", exc, flush=True)

        try:
            amplifier.clear()
        except Exception:
            pass

        print("程序已退出。", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
