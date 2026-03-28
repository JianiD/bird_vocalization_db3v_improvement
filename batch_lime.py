import os
import torch
import numpy as np
from pathlib import Path
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from TDNN2 import tdnn_IFN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ROOTS = {
    "1": "results/tdnn_IFN_aug_sampler_adv/local/ckpt/best_acc.pth.tar",
    "2": "results/tdnn_IFN_2_CycleGAN_aug_sampler_adv/local/ckpt/best_acc.pth.tar",
    "3": "results/tdnn_IFN_3/local/ckpt/best_acc.pth.tar",
}
DATA_ROOT = "meta-v02"
OUTPUT_DIR = "lime_outputs_green_top20"

def build_model(ckpt_path):
    model = tdnn_IFN.TDNN(feat_dim=128, embedding_size=512, num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model.to(DEVICE).eval()

def preprocess(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128,
        f_min=20.0, f_max=8000.0
    )(waveform).squeeze(0)
    spec = torchaudio.transforms.AmplitudeToDB()(spec)
    if spec.shape[-1] < 251:
        spec = F.pad(spec, (0, 251 - spec.shape[-1]), value=spec.min())
    else:
        spec = spec[:, :251]
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
    return spec

def segment_mel(spec, n_freq=4, n_time=16):
    H, W = spec.shape
    seg = np.zeros((H, W), dtype=np.int32)
    f_bins = np.array_split(np.arange(H), n_freq)
    t_bins = np.array_split(np.arange(W), n_time)
    idx = 0
    for fi in f_bins:
        for tj in t_bins:
            seg[np.ix_(fi, tj)] = idx
            idx += 1
    return seg

def build_masks(K, S=1000, p_keep=0.8):
    X = (np.random.rand(S, K) < p_keep).astype(np.float32)
    X[0, :] = 1.0
    return X

def lime_explain(model, spec, target, n_freq=2, n_time=16,
                 S=1000, p_keep=0.8, kernel_width=0.25, ridge=1e-2):
    seg = segment_mel(spec, n_freq, n_time)
    K = seg.max() + 1
    X = build_masks(K, S, p_keep)
    batch = []
    for mask in X:
        mod_spec = spec.copy()
        for sid, keep in enumerate(mask):
            if keep < 0.5:
                mod_spec[seg == sid] = 0.0
        batch.append(mod_spec)
    batch_np = np.array(batch)
    batch_tensor = torch.tensor(batch_np).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = model(batch_tensor)
        prob = torch.softmax(logits, dim=1)[:, target].cpu().numpy()
    d = (1.0 - X).sum(axis=1) / K
    W = np.exp(-d**2 / kernel_width**2)
    W[0] = W.max()
    Xw = X * W[:, None]
    A = Xw.T @ X + ridge * np.eye(K)
    b = Xw.T @ prob
    theta = np.linalg.solve(A, b)
    heatmap = np.zeros_like(seg, dtype=np.float32)
    for sid in range(K):
        heatmap[seg == sid] = theta[sid]
    return heatmap


def save_overlay(spec, heatmap, out_img, out_npy):
    os.makedirs(out_img.parent, exist_ok=True)
    os.makedirs(out_npy.parent, exist_ok=True)
    np.save(out_npy, heatmap)
    np.save(out_npy.with_name(out_npy.stem + "_spec.npy"), spec)
    
    fig = plt.figure(figsize=(10, 3))
    threshold = np.percentile(heatmap, 80)  # top 20%
    mask = np.ma.masked_where(heatmap <= threshold, heatmap)

    plt.imshow(spec, origin="lower", aspect="auto", cmap="gray")
    plt.imshow(mask, cmap="Greens", origin="lower", aspect="auto", alpha=0.6)

    plt.axis('off'); plt.tight_layout()
    fig.savefig(out_img, dpi=150)
    plt.close(fig)

    pass

if __name__ == "__main__":
    for model_id, ckpt_path in MODEL_ROOTS.items():
        model = build_model(ckpt_path)
        for data_id in ["1", "2", "3"]:
            print(f"[INFO] Running model{model_id} on data{data_id}")
            data_dir = Path(DATA_ROOT) / data_id
            out_root = Path(OUTPUT_DIR) / f"model{model_id}_on_data{data_id}"
            lines = []
            for subset in ["train", "val", "test"]:
                txt = data_dir / f"{subset}_set.txt"
                if txt.exists():
                    with open(txt) as f:
                        for line in f:
                            line = line.strip().replace("\\", "/")
                            if line:
                                lines.append(line)
            for rel_path in tqdm(lines, desc=f"model{model_id}_on_data{data_id}"):
                abs_path = Path("Data") / rel_path
                if not abs_path.exists():
                    continue
                try:
                    spec = preprocess(abs_path)
                    x = spec.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(x)
                        target = logits.argmax(dim=1).item()
                    heatmap = lime_explain(model, spec.numpy(), target)
                    # heatmap = lime_explain(model, spec.numpy(), target, n_freq=4, n_time=16)

                    species = Path(rel_path).parts[-2].replace(" ", "_")
                    stem = Path(rel_path).stem
                    out_img = out_root / species / f"{stem}.png"
                    out_npy = out_root / species / f"{stem}.npy"
                    save_overlay(spec.numpy(), heatmap, out_img, out_npy)
                except Exception:
                    traceback.print_exc()
