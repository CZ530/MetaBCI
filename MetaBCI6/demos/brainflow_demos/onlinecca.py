# -*- coding: utf-8 -*-
"""
SSVEP Feedback on NanoEEG with BDF offline training using FBSCCA.
"""

import numpy as np
import socket
import struct
from typing import Tuple

import pyedflib
import mne
from pylsl import StreamInfo, StreamOutlet

from metabci.brainflow.amplifiers import Marker, BaseAmplifier
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank,
    generate_cca_references
)
from metabci.brainda.algorithms.decomposition import FBSCCA
from metabci.brainda.utils import upper_ch_names


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


def load_digital_trigger(bdf_path, trig_name="TRIGGER/STATUS"):
    f = pyedflib.EdfReader(bdf_path)
    labels = [x.upper() for x in f.getSignalLabels()]

    if trig_name.upper() not in labels:
        f.close()
        raise ValueError(f"Trigger channel '{trig_name}' not found in BDF.")

    trig_idx = labels.index(trig_name.upper())
    trig = f.readSignal(trig_idx, digital=True)
    f.close()

    return np.asarray(trig, dtype=np.int64)


def extract_onset_events(trigger):
    """
    Extract 0 -> non-zero rising edges.
    Return MNE events array: [sample, 0, code]
    """
    change_idx = np.where(np.diff(trigger) != 0)[0] + 1
    onset_mask = (trigger[change_idx - 1] == 0) & (trigger[change_idx] != 0)

    onset_idx = change_idx[onset_mask]
    onset_codes = trigger[onset_idx].astype(int)

    events = np.column_stack([
        onset_idx.astype(int),
        np.zeros(len(onset_idx), dtype=int),
        onset_codes
    ])
    return events


def fix_data_length(X, target_len):
    """
    Force the last dimension of X to target_len.
    Crop if longer, zero-pad if shorter.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.shape[-1] > target_len:
        X = X[..., :target_len]
    elif X.shape[-1] < target_len:
        pad_width = [(0, 0)] * X.ndim
        pad_width[-1] = (0, target_len - X.shape[-1])
        X = np.pad(X, pad_width, mode="constant")

    return X


class NanoEEGAmplifier(BaseAmplifier):
    def __init__(
        self,
        device_address: Tuple[str, int] = ('127.0.0.1', 1895),
        srate=1000,
        num_chans=64
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.onePacketSize = 10
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def recv(self):
        data = None
        try:
            data = self.tcp_link.recv(9216)
        except Exception as e:
            print("TCP receive error:", e)
            self.tcp_link.close()
        else:
            if data is not None:
                data = self.__upack_data(data)
                return data
            else:
                print("TCP receive failed, returned empty data.")
                return []

    def __upack_data(self, data):
        data = struct.unpack(f">{(self.num_chans + 1) * self.onePacketSize}i", data)
        data = np.array(data, dtype=np.int32)
        data = data.reshape((self.num_chans + 1, self.onePacketSize))
        data = np.transpose(data)
        return data.tolist()

    def start_trans(self):
        self.connect_tcp()
        print("TCP connected successfully.")
        self.start()

    def stop_trans(self):
        self.stop()
        self.tcp_link.close()


def read_data(run_files, ch_ind, interval, labels, event_map=None, srate=1000):
    Xs, ys = [], []
    stim_len = interval[1] - interval[0]
    sample_point = int(round(stim_len * srate))

    for run_file in run_files:
        raw = mne.io.read_raw_bdf(run_file, preload=True, verbose=False)
        raw = upper_ch_names(raw)

        trigger = load_digital_trigger(run_file, trig_name="Trigger/Status")
        events = extract_onset_events(trigger)

        print("events shape:", events.shape)

        keep_mask = np.isin(events[:, 2], labels)
        events = events[keep_mask]

        print("filtered events shape:", events.shape)
        if len(events) == 0:
            raise ValueError(f"No valid events found in {run_file}. Labels={labels}")

        event_id = {str(label): label for label in labels}

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=interval[0],
            tmax=interval[1],
            baseline=None,
            picks=ch_ind,
            preload=True,
            verbose=False
        )

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]   # Keep consistent with your offline code
            X = fix_data_length(X, sample_point)
            y = np.ones(len(X)) * label

            Xs.append(X)
            ys.append(y)

    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_ind


def build_fbscca(srate=1000, stim_len=2.0, freq_list=(8, 10, 12)):
    wp = [
        [6, 88],
        [14, 88],
        [22, 88],
        [30, 88],
        [38, 88]
    ]
    ws = [
        [4, 90],
        [12, 90],
        [20, 90],
        [28, 90],
        [36, 90]
    ]

    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, srate)
    Yf = generate_cca_references(freq_list, srate=srate, T=stim_len, n_harmonics=1)

    model = FBSCCA(
        filterbank=filterbank,
        n_components=1,
        filterweights=filterweights,
        n_jobs=-1
    )

    return model, Yf


def train_model(X, y, srate=1000, stim_len=2.0):
    """
    Train FBSCCA exactly following your offline script style.
    """
    y = np.reshape(y, (-1))
    X = fix_data_length(X, int(round(srate * stim_len)))

    model, Yf = build_fbscca(
        srate=srate,
        stim_len=stim_len,
        freq_list=[8, 10, 12]
    )
    model.fit(X, y, Yf=Yf)
    return model


def model_predict(X, srate=1000, stim_len=2.0, model=None):
    """
    Predict with FBSCCA.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = fix_data_length(X, int(round(srate * stim_len)))

    p_labels = model.predict(X)
    return p_labels


def offline_validation(X, y, srate=1000, stim_len=2.0):
    """
    This follows your offline example:
    fit on all data and predict on the same data.
    Note: this is training-set accuracy, not cross-validation accuracy.
    """
    y = np.reshape(y, (-1))
    model = train_model(X, y, srate=srate, stim_len=stim_len)
    p_labels = model_predict(X, srate=srate, stim_len=stim_len, model=model)
    return np.mean(p_labels == y)


class FeedbackWorker(ProcessWorker):
    def __init__(
        self,
        run_files,
        ch_ind,
        stim_interval,
        stim_labels,
        event_map,
        srate,
        lsl_source_id,
        timeout,
        worker_name
    ):
        self.run_files = run_files
        self.ch_ind = ch_ind
        self.stim_interval = stim_interval
        self.stim_len = stim_interval[1] - stim_interval[0]
        self.stim_labels = stim_labels
        self.event_map = event_map
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, self.ch_ind = read_data(
            run_files=self.run_files,
            ch_ind=self.ch_ind,
            interval=self.stim_interval,
            labels=self.stim_labels,
            event_map=self.event_map,
            srate=self.srate
        )
        print("Loading train data successfully")

        acc = offline_validation(X, y, srate=self.srate, stim_len=self.stim_len)
        print("Current Model accuracy: {:.4f}".format(acc))

        self.estimator = train_model(X, y, srate=self.srate, stim_len=self.stim_len)

        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id
        )
        self.outlet = StreamOutlet(info)

        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        udp_socket2Stim = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr2Stim = ('192.168.10.2', 6000)

        data = np.array(data, dtype=np.float64).T

        if data.shape[0] <= np.max(self.ch_ind):
            print(f"Channel index out of range. data.shape={data.shape}, ch_ind={self.ch_ind}")
            return

        data = data[self.ch_ind]

        p_labels = model_predict(
            data,
            srate=self.srate,
            stim_len=self.stim_len,
            model=self.estimator
        )

        pred_id = int(np.asarray(p_labels).reshape(-1)[0] + 1)

        udp_socket2Stim.sendto(str(pred_id).encode(), udp_addr2Stim)

        out_label = [pred_id]
        print('predict_id_paradigm', out_label)

        if self.outlet.have_consumers():
            self.outlet.push_sample(out_label)

    def post(self):
        pass


if __name__ == '__main__':
    srate = 1000
    stim_interval = [0.14, 2.14]
    stim_labels = list(range(1, 4))
    event_map = {str(e): e for e in range(1, 255)}

    run_files = [r"D:\SUB\414.bdf"]

    # Zero-based indexing for Python
    ch_ind = np.array([14, 16, 17, 18, 19, 20, 21, 22, 23], dtype=int) - 1

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        ch_ind=ch_ind,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        event_map=event_map,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name
    )

    marker = Marker(
        interval=stim_interval,
        srate=srate,
        events=stim_labels
    )

    ns = NanoEEGAmplifier(
        device_address=('127.0.0.1', 1895),
        srate=srate,
        num_chans=32
    )

    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    ns.start_trans()