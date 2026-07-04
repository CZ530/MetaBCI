import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import mne
from mne.filter import resample

from metabci.brainda.utils import upper_ch_names
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank,
    generate_cca_references,
)
from metabci.brainda.algorithms.decomposition import FBTDCA, FBTRCA, FBSCCA
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut,
)


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


def read_data(run_files, ch_ind, interval, labels):
    """
    读取离线 BDF 训练数据。
    这里保持和 online nano.py 里的 read_data 尽量一致，保证验证结果和在线训练一致。
    """
    mne.set_log_level(verbose=False)
    Xs, ys = [], []
    last_ch_picks = None

    for run_file in run_files:
        print("读取文件:", run_file)

        raw = mne.io.read_raw_bdf(
            run_file,
            preload=True,
            stim_channel="Trigger/Status",
            verbose=False
        )

        raw = upper_ch_names(raw)

        raw.filter(
            6,
            40,
            l_trans_bandwidth=2,
            h_trans_bandwidth=5,
            phase="zero-double"
        )

        events = mne.find_events(raw, verbose=False)
        print("事件数量:", len(events))

        last_ch_picks = mne.pick_channels(
            raw.ch_names,
            ch_ind,
            ordered=True
        )

        print("使用通道:", ch_ind)
        print("通道索引:", last_ch_picks)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=labels,
            tmin=interval[0],
            tmax=interval[1],
            picks=last_ch_picks,
            baseline=None,
            verbose=False
        )

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:] * 1e6
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)

    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    print("X shape:", Xs.shape)
    print("y shape:", ys.shape)
    print("类别标签:", np.unique(ys))

    return Xs, ys, last_ch_picks


def train_model(X, y, algorithms="eTRCA", srate=1000):
    """
    训练模型。参数保持和 online nano.py 一致。
    """
    y = np.reshape(y, (-1))

    # 1000 Hz -> 250 Hz
    X = resample(X, up=250, down=srate)

    wp = [
        [6, 88],
        [14, 88],
        [22, 88],
        [30, 88],
        [38, 88],
    ]

    ws = [
        [4, 90],
        [12, 90],
        [20, 90],
        [28, 90],
        [36, 90],
    ]

    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)

    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    all_freq_list = np.arange(10, 17, 1)

    if algorithms == "TDCA":
        freqs = all_freq_list
        Yf = generate_cca_references(
            freqs,
            srate=250,
            T=1,
            n_harmonics=1
        )

        model = FBTDCA(
            filterbank,
            padding_len=2,
            n_components=1,
            filterweights=np.array(filterweights)
        )

        model = model.fit(X, y, Yf)

    elif algorithms == "eTRCA":
        model = FBTRCA(
            filterbank=filterbank,
            n_components=2,
            ensemble=True,
            filterweights=np.array(filterweights),
            n_jobs=-1
        )

        model = model.fit(X, y)

    elif algorithms == "FBCCA":
        freqs = all_freq_list
        Yf = generate_cca_references(
            freqs,
            srate=250,
            T=5,
            n_harmonics=5
        )

        model = FBSCCA(
            filterbank,
            filterweights=filterweights
        )

        model = model.fit(X=X, y=y, Yf=Yf)

    else:
        raise ValueError(f"不支持的算法: {algorithms}")

    return model


def model_predict(X, srate=1000, model="eTRCA"):
    """
    离线验证预测。保持和 online nano.py 一致。
    """
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))

    X = resample(X, up=250, down=srate)

    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    p_labels = model.predict(X)
    return p_labels


def offline_validation(X, y, srate=1000, algorithms="eTRCA", t=2):
    """
    离线交叉验证。
    注意：这一步只用于评估训练文件质量，不用于实时在线处理。
    """
    X = X[:, :, :int(srate * t)]
    y = np.reshape(y, (-1))

    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []

    for fold_id, (train_ind, test_ind) in enumerate(spliter.split(X, y=y), start=1):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(
            X_train,
            y_train,
            srate=srate,
            algorithms=algorithms
        )

        p_labels = model_predict(
            X_test,
            srate=srate,
            model=model
        )

        acc = np.mean(p_labels == y_test)
        kfold_accs.append(acc)

        print(f"Fold {fold_id} accuracy: {acc * 100:.2f}%")

    mean_acc = np.mean(kfold_accs)

    print("================================")
    print("算法:", algorithms)
    print("信号长度:", t, "秒")
    print("离线交叉验证准确率: {:.2f}%".format(mean_acc * 100))
    print("================================")

    return mean_acc


if __name__ == "__main__":
    # =========================
    # 接收 BDF 文件路径
    # =========================
    if len(sys.argv) >= 2:
        run_files = sys.argv[1:]
    else:
        # run_files = [r"D:\0425\7.bdf"]
        run_files = [r"F:\WheelchairControl\WheelchairControl\1.bdf"]
    for file_path in run_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"离线训练文件不存在: {file_path}")

    print("当前验证文件:")
    for file_path in run_files:
        print("  ", file_path)

    # =========================
    # 参数需要和在线处理保持一致
    # =========================
    srate = 1000
    stim_interval = [0.14, 2.14]
    stim_labels = list(range(1, 8))

    pick_chs = [
        "P3",
        "PZ",
        "P4",
        "POZ",
        "PO6",
        "PO3",
        "PO4",
        "O1",
        "O2",
    ]

    algorithms = "eTRCA"
    signal_length = 2

    # =========================
    # 读取数据并交叉验证
    # =========================
    X, y, ch_picks = read_data(
        run_files=run_files,
        ch_ind=pick_chs,
        interval=stim_interval,
        labels=stim_labels
    )

    print("Loading train data successfully")

    offline_validation(
        X,
        y,
        srate=srate,
        algorithms=algorithms,
        t=signal_length
    )
