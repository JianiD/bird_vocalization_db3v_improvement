import os
import torch
import torchaudio
import torch.nn.functional as F
from TDNN2.tdnn_IFN import TDNN
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path


MODEL_ROOTS = {
    "1": "results/tdnn_IFN_aug_sampler_adv/local/ckpt/best_acc.pth.tar",
    "2": "results/tdnn_IFN_2_CycleGAN_aug_sampler_adv/local/ckpt/best_acc.pth.tar",
    "3": "results/tdnn_IFN_3/local/ckpt/best_acc.pth.tar",
}
DATA_ROOT = "meta-v02"
OUTPUT_DIR = "gradcam_outputs"
LAYER_NAME = "tdnn3" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    print(f"\n[Load Model] {model_path}")
    model = TDNN(feat_dim=128, embedding_size=512, num_classes=10)
    state = torch.load(model_path, map_location=DEVICE)

    if any(k.startswith("base_model.") for k in state.keys()):
        print("[Info] Detected base_model wrapping, stripping prefix...")
        state = {k.replace("base_model.", ""): v for k, v in state.items() if k.startswith("base_model.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warning] Missing keys:", missing)
    if unexpected:
        print("[Warning] Unexpected keys:", unexpected)

    model.to(DEVICE).eval()
    return model

def register_hooks(model, layer_name):
    layer_map = {
        "tdnn1": model.xvector[0],
        "tdnn3": model.xvector[2],
        "affine": model.xvector[-2],
    }
    target_layer = layer_map[layer_name]
    output = {}

    def forward_hook(module, inp, out):
        output['feature'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        output['grad'] = grad_out[0].detach()

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    print(f"[Hook] Registered on layer: {layer_name}")
    return output

def preprocess_audio(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    spec_transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128,
            f_min=20.0, f_max=8000.0
        ),
        torchaudio.transforms.AmplitudeToDB()
    )
    spec = spec_transform(waveform).squeeze(0)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
    spec = F.pad(spec, (0, max(0, 251 - spec.shape[-1])))[:, :251]
    return spec.unsqueeze(0).to(DEVICE)

def compute_gradcam(model, inputs, target_class, hook_data):
    output, _ = model(inputs)
    model.zero_grad()
    output[0, target_class].backward()
    weights = hook_data['grad'].mean(dim=0)
    cam = torch.sum(weights * hook_data['feature'].squeeze(0), dim=0)
    cam = F.relu(cam)

    if cam.ndim == 0:
        raise ValueError("Grad-CAM output is scalar — cannot interpolate. Consider using tdnn3 or tdnn1 instead.")

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    cam_interp = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(251,), mode='linear', align_corners=False)
    return cam_interp.squeeze().cpu().numpy()


def save_visual(cam, spec, out_img_path, out_npy_path):
    os.makedirs(out_img_path.parent, exist_ok=True)

    spec_np = spec.cpu().numpy()
    spec_np = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-9)

    cam_np = np.asarray(cam)
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-9)

    h, w = spec_np.shape
    
    cam_2d = np.tile(cam_np[None, :], (spec_np.shape[0], 1))


    plt.figure(figsize=(10, 3))
    plt.imshow(spec_np, origin='lower', aspect='auto', cmap='gray')
    plt.imshow(cam_2d, origin='lower', aspect='auto', cmap='jet', alpha=0.45)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    plt.close()

    np.save(out_npy_path, cam_2d)
    np.save(out_npy_path.with_name(out_npy_path.stem + "_spec.npy"), spec_np)
    
def run():
    for model_region, model_path in MODEL_ROOTS.items():
        model = load_model(model_path)
        hook_data = register_hooks(model, LAYER_NAME)

        for data_region in MODEL_ROOTS.keys():
            test_txt = os.path.join(DATA_ROOT, data_region, "test_set.txt")
            if not os.path.exists(test_txt):
                print(f"[Skip] Missing: {test_txt}")
                continue
            with open(test_txt, "r") as f:
                lines = [line.strip().replace("\\", "/") for line in f if line.strip()]

            print(f"\n[Model D{model_region}] on [Data D{data_region}] — Total: {len(lines)} files")
            for rel_path in tqdm(lines, desc=f"D{model_region}->D{data_region}"):
                abs_path = os.path.join("Data", rel_path)
                if not os.path.exists(abs_path):
                    print(f"[Warn] Missing file: {abs_path}")
                    continue
                try:
                    inputs = preprocess_audio(abs_path)
                    spec_vis = inputs.squeeze(0).cpu()
                    output, _ = model(inputs)
                    target_class = output.argmax(dim=1)
                    cam = compute_gradcam(model, inputs, target_class, hook_data)

                    parts = Path(rel_path).parts
                    species = parts[-2].replace(" ", "_")
                    filename = Path(parts[-1]).stem
                    out_dir = Path(OUTPUT_DIR) / f"model{model_region}_on_data{data_region}" / species
                    save_visual(cam, spec_vis, out_dir / f"{filename}.png", out_dir / f"{filename}.npy")
                except Exception as e:
                    print(f"[Error] {rel_path}: {e}")

if __name__ == "__main__":
    run()
