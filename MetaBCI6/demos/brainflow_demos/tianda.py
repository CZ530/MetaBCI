import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import matplotlib

# 强制使用独立窗口
matplotlib.use('TkAgg')

# 1. 基础配置
bdf_file = r"D:\齐鲁工业\shuju\5.bdf"
target_freqs = [8, 9, 10, 11, 12, 13, 14, 15]
n_harmonics = 2
tmin, tmax = 0.14, 2.14

# 2. 读取并放大信号
raw = mne.io.read_raw_bdf(bdf_file, preload=True)
raw.apply_function(lambda x: x * 1e6)
raw.filter(l_freq=5.0, h_freq=40.0, verbose=False)

# 3. 提取事件并过滤标签
events, event_id_dict = mne.events_from_annotations(raw)

# --- 核心修改：去掉标签 '0' ---
# 仅保留标签名不是 '0' 的项
valid_event_id = {k: v for k, v in event_id_dict.items() if k != '0'}
print(f"分析将包含的标签: {valid_event_id}")

# 自动挑选枕区通道
picks = mne.pick_types(raw.info, eeg=True, selection=['O', 'P', 'PO'])
if len(picks) == 0:
    picks = mne.pick_channels(raw.ch_names, raw.ch_names[-5:])

# 在切分时传入过滤后的 valid_event_id，MNE 会自动丢弃标签 '0'
epochs = mne.Epochs(raw, events, event_id=valid_event_id,
                    tmin=tmin, tmax=tmax, picks=picks,
                    baseline=None, preload=True, verbose=False)

# 4. CCA 核心工具函数
def get_reference_signals(freq, fs, n_samples, harmonics):
    t = np.arange(n_samples) / fs
    ref = []
    for h in range(1, harmonics + 1):
        ref.append(np.sin(2 * np.pi * h * freq * t))
        ref.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(ref)

# 5. 执行分类分析
X = epochs.get_data()  # 此时 X 已经不包含标签 '0' 的数据
fs = raw.info['sfreq']
n_trials, n_chans, n_samples = X.shape

all_corrs = []
print(f"\n{'试次':<4} | {'真实标签':<6} | {'预测频率':<8} | {'最大相关度':<8}")
print("-" * 45)

for i in range(n_trials):
    trial_data = X[i].T
    rhos = []

    for f in target_freqs:
        ref = get_reference_signals(f, fs, n_samples, n_harmonics).T
        cca = CCA(n_components=1)
        cca.fit(trial_data, ref)
        u, v = cca.transform(trial_data, ref)
        rho = np.corrcoef(u.T, v.T)[0, 1]
        rhos.append(rho)

    all_corrs.append(rhos)
    predicted_idx = np.argmax(rhos)
    pred_freq = target_freqs[predicted_idx]

    # 获取当前试次的真实标签名
    current_event_val = epochs.events[i, -1]
    # 从过滤后的字典中反查标签名
    true_label = [k for k, v in valid_event_id.items() if v == current_event_val][0]

    print(f"{i:<6} | {true_label:<10} | {pred_freq:>2} Hz    | {max(rhos):.4f}")

# 6. 可视化
plt.figure(figsize=(10, 6))
plt.imshow(all_corrs, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient (Rho)')
plt.xticks(range(len(target_freqs)), [f"{f}Hz" for f in target_freqs])
plt.yticks(range(n_trials), [f"Trial {i}" for i in range(n_trials)])
plt.xlabel("Target Frequencies")
plt.ylabel("Trials (Excluding '0')")
plt.title("CCA Result Matrix (Pure Stimulus Trials)")
plt.show()