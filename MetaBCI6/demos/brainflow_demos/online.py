# -*- coding: utf-8 -*-
"""
SSVEP Feedback on NanoEEG with BDF offline training.
"""

import time
import numpy as np
import socket
import struct
from typing import Tuple
import pyedflib
import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import Marker, BaseAmplifier
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references
)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut
)
from metabci.brainda.algorithms.decomposition import FBTDCA,FBTRCA,FBSCCA
from metabci.brainda.utils import upper_ch_names
from sklearn.base import BaseEstimator, ClassifierMixin


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y

class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y
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

class NanoEEGAmplifier(BaseAmplifier):

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=1000,
                 num_chans=64):
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
            print("连接出错", e)
            self.tcp_link.close()
        else:
            if data is not None:
                data = self.__upack_data(data)
                return data
            else:
                print("连接出错,此次请求数据失败，返回空数据")
                return []

    def __upack_data(self, data):
        data = struct.unpack(f">{(self.num_chans + 1) * self.onePacketSize}i", data)
        data = np.array(data, dtype=np.int32)
        data = data.reshape((self.num_chans + 1, self.onePacketSize))
        data = np.transpose(data)

        return data.tolist()

    def start_trans(self):
        self.connect_tcp()
        print("连接成功")
        self.start()

    def stop_trans(self):
        self.stop()
        self.tcp_link.close()

# def read_data(run_files, ch_ind, interval, labels, event_map=None):
#     Xs, ys = [], []
#
#     for run_file in run_files:
#         raw = mne.io.read_raw_bdf(run_file, preload=True,  stim_channel="Trigger/Status", verbose=False)
#         raw = upper_ch_names(raw)
#
#         # Read trigger from digital status channel instead of annotations
#         # trigger = load_digital_trigger(run_file, trig_name="Trigger/Status")
#         # events = extract_onset_events(trigger)
#
#         events = mne.find_events(raw)
#         print("events shape:", events.shape)
#
#         # Keep only wanted labels
#         # keep_mask = np.isin(events[:, 2], labels)
#         # events = events[keep_mask]
#
#         print("filtered events shape:", events.shape)
#         if len(events) == 0:
#             raise ValueError(f"No valid events found in {run_file}. Labels={labels}")
#
#         # Build event_id dict for MNE
#         event_id = {str(label): label for label in labels}
#
#         epochs = mne.Epochs(
#             raw,
#             events,
#             event_id=event_id,
#             tmin=interval[0],
#             tmax=interval[1],
#             baseline=None,
#             picks=ch_ind,
#             preload=True,
#             verbose=False
#         )
#
#         for label in labels:
#             X = epochs[str(label)].get_data()[..., 1:]
#             Xs.append(X)
#             ys.append(np.ones((len(X))) * label)
#
#     Xs = np.concatenate(Xs, axis=0)
#     ys = np.concatenate(ys, axis=0)
#     ys = label_encoder(ys, labels)
#
#     return Xs, ys, ch_ind

def read_data(run_files, ch_ind, interval, labels):
    mne.set_log_level(verbose=False)
    Xs, ys = [], []
    for run_file in run_files:
        print(run_file)
        raw = mne.io.read_raw_bdf(run_file, preload=True, stim_channel="Trigger/Status",verbose=False)
        raw = upper_ch_names(raw)
        raw.filter(6, 40, l_trans_bandwidth=2, h_trans_bandwidth=5,
                   phase='zero-double')
        events = mne.find_events(raw, verbose=False)
        ch_picks = mne.pick_channels(raw.ch_names, ch_ind, ordered=True)
        epochs = mne.Epochs(raw, events,
                            event_id=labels,
                            tmin=interval[0],
                            tmax=interval[1],
                            picks=ch_picks,
                            baseline=None,
                            verbose=False)
        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:] * 1e6
            Xs.append(X)
            ys.append(np.ones((len(X)))*label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)

    return Xs, ys, ch_picks

# def train_model(X, y, srate=1000):
#     y = np.reshape(y, (-1))
#     X = resample(X, up=256, down=srate)
#
#     wp = [[6, 68], [14, 68], [22, 68], [30, 68], [38, 68]]
#     ws = [[4, 70], [12, 70], [20, 70], [28, 70], [36, 70]]
#
#     filterweights = np.arange(1, 6) ** (-1.25) + 0.25
#     filterbank = generate_filterbank(wp, ws, 256)
#
#     X = X - np.mean(X, axis=-1, keepdims=True)
#     X = X / np.std(X, axis=(-1, -2), keepdims=True)
#
#     # freqs = np.arange(8, 16, 1)
#     # Yf = generate_cca_references(freqs, srate=256, T=1, n_harmonics=5)
#     # model = FBTDCA(
#     #     filterbank,
#     #     padding_len=3,
#     #     n_components=1,
#     #     filterweights=np.array(filterweights)
#     # )
#
#     model = FBTRCA(
#         filterbank,
#         ensemble= True,
#         filterweights=np.array(filterweights)
#     )
#
#     model = model.fit(X, y)
#     return model
#
#
# def model_predict(X, srate=1000, model=None):
#     X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
#     X = resample(X, up=256, down=srate)
#     X = X - np.mean(X, axis=-1, keepdims=True)
#     X = X / np.std(X, axis=(-1, -2), keepdims=True)
#     p_labels = model.predict(X)
#     return p_labels
#
#
# def offline_validation(X, y, srate=1000):
#     y = np.reshape(y, (-1))
#     spliter = EnhancedLeaveOneGroupOut(return_validate=False)
#
#     kfold_accs = []
#     for train_ind, test_ind in spliter.split(X, y=y):
#         X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
#         X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
#
#         model = train_model(X_train, y_train, srate=srate)
#         p_labels = model_predict(X_test, srate=srate, model=model)
#         kfold_accs.append(np.mean(p_labels == y_test))
#
#     return np.mean(kfold_accs)

def train_model(X, y, algorithms, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=250, down=srate)

    wp = [
        [6, 48], [14, 48], [22, 48], [30, 48], [38, 48]   # 88
    ]
    ws = [
        [4, 50], [12, 50], [20, 50], [28, 50], [36, 50]   # 90
    ]

    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    all_freq_list = np.arange(8, 14, 1)

    if algorithms == 'TDCA':
        freqs = all_freq_list
        Yf = generate_cca_references(freqs, srate=250, T=t_task, n_harmonics=5)
        model = FBTDCA(filterbank, padding_len=2, n_components=2,
                       filterweights=np.array(filterweights))

        model = model.fit(X, y, Yf)
    elif algorithms == 'eTRCA':
        model = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights), n_jobs=-1)
        model = model.fit(X, y)
    elif algorithms == 'FBCCA':
        freqs = all_freq_list
        Yf = generate_cca_references(freqs, srate=250, T=5, n_harmonics=5)
        model = FBSCCA(
            filterbank, filterweights=filterweights)
        model = model.fit(X=X, y=y, Yf=Yf)
    elif algorithms == 'msSAME-eTRCA':
        freq_list = all_freq_list
        phase_list = np.array([
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
            0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
        ])
        # phase_list = np.array([
        #     0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0, 0, 0,
        # ])
        mssame = MSSAME(fs=250, Nh=5, flist=freq_list, plist=phase_list, n_Aug=8,
                        n_Neig=1)  # When n_Neig=1, the result is similar to SAME
        mssame.fit(X, y)
        X_aug, y_aug = mssame.augment()
        X_train_new = np.concatenate((X, X_aug), axis=0)
        y_train_new = np.concatenate((y, y_aug), axis=0)
        model = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights),
                           n_jobs=-1)
        model.fit(X=X_train_new, y=y_train_new)
    else:
        print('unsupported algorithms"')
    return model

def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=250, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels

def offline_validation(X, y, srate=1000, algorithms = 'eTRCA', t = 2):
    X = X [:,:,:int(srate * t)]
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        model = train_model(X_train, y_train, srate=srate, algorithms = algorithms)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
    print('algorithms=',algorithms,'; signal length=', t, '; accuracy=', np.mean(kfold_accs))
    return np.mean(kfold_accs)

class FeedbackWorker(ProcessWorker):
    def __init__(
        self,
        run_files,
        ch_ind,         # 修改：直接接收 channel index
        stim_interval,
        stim_labels,
        # event_map,
        srate,
        lsl_source_id,
        timeout,
        worker_name
    ):
        self.run_files = run_files
        self.pick_chs = ch_ind
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        # self.event_map = event_map
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, self.ch_ind = read_data(
            run_files=self.run_files,
            ch_ind=self.pick_chs,
            interval=self.stim_interval,
            labels=self.stim_labels
        )
        print("Loading train data successfully")

        acc = offline_validation(X, y, srate=self.srate)
        print("Current Model accuracy: {:.2f}".format(acc))

        self.estimator = train_model(X, y, srate=self.srate, algorithms = 'eTRCA')

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
        udp_addr2Stim = ('192.168.3.12', 6000)

        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]

        p_labels = model_predict(data, srate=self.srate, model=self.estimator)

        pred_id = int(np.asarray(p_labels).reshape(-1)[0] + 1)

        udp_socket2Stim.sendto(str(pred_id).encode(), udp_addr2Stim)

        out_label = [pred_id]
        print('predict_id_paradigm', out_label)

        if self.outlet.have_consumers():
            self.outlet.push_sample(out_label)

    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000

    # Data epoch duration, 0.14 s visual delay was taken into account
    stim_interval = [0.14, 2.14]

    # Label types
    stim_labels = list(range(1, 7))
    #stim_labels = list([1, 6, 11, 16 ,21, 26, 31, 36])

    # event_map = {str(e): e for e in range(1, 255)}
    #cnts = 1
    filepath = r"D:\0425"
    run_files = ['{:s}\\{:d}.bdf'.format(
        filepath, run) for run in range(11, 12)]
    #runs = list(range(1, cnts + 1))
    # 修改：直接使用整数索引切片
    pick_chs = ['P3', 'PZ', 'P4',  'POZ',  'PO6', 'O2','PO3','PO4']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        ch_ind=pick_chs,         # 传入序列
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        # event_map=event_map,
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
        num_chans=32  # 修改：必须是32，否则在线接收数据切片时会因为通道数不够而越界
    )

    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)

    ns.start_trans()


    # ns.down_worker(feedback_worker_name)
    # time.sleep(0.5)
    # ns.stop_trans()
    # try:
    #     ns.clear()
    # except Exception:
    #     pass
    # print('bye')