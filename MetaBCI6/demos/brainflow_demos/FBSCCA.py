import matplotlib.pyplot as plt
import numpy as np
from metabci.brainda.algorithms.decomposition import FBSCCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
from mne.io import read_raw_bdf, read_raw_cnt
import mne
import pyedflib

from scipy.signal import filtfilt, lfilter


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
    """Extract rising-edge events: 0 -> non-zero."""
    change_idx = np.where(np.diff(trigger) != 0)[0] + 1
    onset_mask = (trigger[change_idx - 1] == 0) & (trigger[change_idx] != 0)
    onset_idx = change_idx[onset_mask]
    onset_codes = trigger[onset_idx].astype(int)
    events = np.column_stack([onset_idx.astype(int), np.zeros(len(onset_idx), dtype=int), onset_codes])
    return events

def read_bdf_file(run_files, trials, nlabels, stimlen, chs):
    fs = 1000
    delay = 0.14
    labels = list(range(1, nlabels + 1))
    for run_file in run_files:
        Xs, ys = [], []
        raw = read_raw_bdf(run_file, preload=True, verbose=False)
        # raw.notch_filter(np.arange(50, 251, 50), n_jobs=1)
        events = mne.events_from_annotations(raw, event_id=lambda x: int(x), verbose=False)[0]

        trigger = load_digital_trigger(run_file, trig_name="Trigger/Status")
        events = extract_onset_events(trigger)

        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events, event_id=labels, tmin=delay, tmax=delay + stimlen, baseline=None,
                            picks=ch_picks,
                            preload=True, verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            y = np.ones(len(X)) * label
            Xs.append(X)
            ys.append(y)
        Xs_all = np.concatenate(Xs, axis=0)
        # Xs_all = Xs_all - np.mean(Xs_all, axis=2, keepdims=True)
        ys_all = np.concatenate(ys, axis=0)
        ys_all = label_encoder(ys_all, labels)
    return Xs_all, ys_all


# '''列建模'''
# # # If everything is fine, you will get the accuracy about 0.9417.
srate = 1000
stimlen = 2
sample_point = int(srate * stimlen)

pick_chs = np.array([14, 16, 17, 18, 19, 20, 21, 22, 23], dtype=int) - 1
pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

freq_list = [8, 10, 12]
Yf = generate_cca_references(freq_list, srate=1000, T=2, n_harmonics=2)

wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
      ]
ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
      ]
filterweights = np.arange(1, 6) ** (-1.25) + 0.25
filterbank = generate_filterbank(wp, ws, 1000)

'''在线'''
# online_filepath = 'D:\\ZFEEG\\app\\data\\20240828zaz_dry_new\\2step\\offline'
online_filepath = ['D:\SUB/ZPC.bdf']
# n_cnt_online = range(1, 2)
# online_run_files = ['{:s}\\{:d}.bdf'.format(online_filepath, run) for run in n_cnt_online]
n_label = 3
n_trial = 33

'''online'''
online_stimlen = 2

X_simu, y_simu = read_bdf_file(online_filepath, n_trial, n_label, online_stimlen, pick_chs)
print(X_simu)
print('read data successful')
kfold_accs = []
model = FBSCCA(filterbank=filterbank, n_components=1, filterweights=filterweights, n_jobs=-1)
model.fit(X_simu, y_simu, Yf=Yf)
p_label = model.predict(X_simu)
kfold_accs.append(np.mean(p_label == y_simu))
print(np.mean(kfold_accs))
