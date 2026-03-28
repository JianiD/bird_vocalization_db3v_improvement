import os
import torch
import math
import json
import tempfile
import shutil
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score
from Data.bird_ds import BirdsDS_IMG as BirdsDS
from train_tdnn_adv import TDNNWithDomainAdversarial, extract_region_id
from hydra.utils import get_original_cwd
import torchaudio


# Noise
class NoiseRobustnessTester:
    def __init__(self, device, field_noise_path=None):
        self.device = device
        self.field_noise = None
        # BirdCLEF Noise
        if field_noise_path and os.path.exists(field_noise_path):
            print(f"Loading field noise bank from: {field_noise_path}")
            self.field_noise, _ = torchaudio.load(field_noise_path)
            self.field_noise = self.field_noise.to(device)

    def add_white_noise(self, spec_batch, snr_db):
        std = torch.std(spec_batch)
        noise_std = std / (10 ** (snr_db / 20))
        noise = torch.randn_like(spec_batch) * noise_std
        return spec_batch + noise

    def add_field_noise(self, spec_batch, snr_db):
        if self.field_noise is None:
            return spec_batch
            
        target_t = spec_batch.shape[-1]
        start = torch.randint(0, self.field_noise.shape[-1] - target_t, (1,))
        noise_seg = self.field_noise[:, start : start + target_t]
        
        sig_energy = torch.mean(10 ** (spec_batch / 10))
        noise_energy = torch.mean(10 ** (noise_seg / 10))
        desired_noise_energy = sig_energy / (10 ** (snr_db / 10))
        scale = torch.sqrt(desired_noise_energy / (noise_energy + 1e-9))
        
        return spec_batch + (noise_seg * scale * 0.1)

    def evaluate_all_conditions(self, model, dataloader, snr_levels=[5, 10, 25, 35]):
        results = {}
        model.eval()
        
        print("\n" + "="*50)
        print("STARTING ROBUSTNESS EVALUATION (White & Field Noise)")
        print("="*50)

        for snr in snr_levels:
            print(f"\n[Test] White Noise @ {snr}dB")
            results[f"White_{snr}dB"] = self._run_eval_core(model, dataloader, noise_type='white', snr=snr)
            
        if self.field_noise is not None:
            print(f"\n[Test] Real Field Noise (BirdCLEF) @ 10dB")
            results["Field_10dB"] = self._run_eval_core(model, dataloader, noise_type='field', snr=10)
            
        return results

    def _run_eval_core(self, model, dataloader, noise_type=None, snr=None):
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch, label, _ in tqdm(dataloader, leave=False, desc=f"{noise_type} {snr}dB"):
                batch = batch.to(self.device)
                
                if noise_type == 'white':
                    batch = self.add_white_noise(batch, snr)
                elif noise_type == 'field':
                    batch = self.add_field_noise(batch, snr)
                
                out, _ = model(batch)
                pred = out.argmax(dim=1)
                y_true.extend(label.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        return {
            "acc": accuracy_score(y_true, y_pred),
            "uar": recall_score(y_true, y_pred, average='macro'),
            "f1": f1_score(y_true, y_pred, average='macro')
        }



@hydra.main(version_base=None, config_path="Config", config_name="config_tdnn")
def evaluate(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Evaluation Settings ===")
    print(f"Dataset: {cfg.evaluation.ds}")
    print(f"Model Path: {cfg.meta.result}")

    test_ds = BirdsDS(cfg.evaluation.ds, phase='test')
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=cfg.hparams.bs, shuffle=False)

    tdnn_type = cfg.model.tdnn
    if 'tdnn_BN' in tdnn_type: from TDNN2 import tdnn_BN as model_zoo
    elif 'tdnn_IFN' in tdnn_type: from TDNN2 import tdnn_IFN as model_zoo
    elif 'tdnn_LSTM' in tdnn_type: from TDNN2 import tdnn_LSTM as model_zoo
    elif 'tdnn_both' in tdnn_type: from TDNN2 import tdnn_both as model_zoo
    elif 'tdnn_GW' in tdnn_type: from TDNN2 import tdnn_GW as model_zoo
    elif 'tdnn_TN' in tdnn_type: from TDNN2 import tdnn_TN as model_zoo
    elif 'tdnn_RIFN' in tdnn_type: from TDNN2 import tdnn_RIFN as model_zoo
    else: raise ValueError(f"Unknown type: {tdnn_type}")

    base_model = model_zoo.TDNN(feat_dim=128, embedding_size=512, num_classes=cfg.model.num_classes)
    model = TDNNWithDomainAdversarial(base_model, num_domains=3, num_classes=cfg.model.num_classes)

    ckpt_dir = cfg.meta.result
    candidates = ["best_acc.pth.tar", "best_uar.pth.tar", "last.pth.tar"]
    ckpt_path = next((os.path.join(ckpt_dir, f) for f in candidates if os.path.exists(os.path.join(ckpt_dir, f))), None)
    
    if ckpt_path is None:
        print(f"Checkpoint not found in {ckpt_dir}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch, label, _ in tqdm(test_dl, desc="Clean Eval"):
            batch, label = batch.to(device), label.to(device)
            out, _ = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc_clean = accuracy_score(y_true, y_pred)
    uar_clean = recall_score(y_true, y_pred, average='macro')
    f1_clean = f1_score(y_true, y_pred, average='macro')


    noise_path = os.path.join(get_original_cwd(), "Data/birdclef_noise_bank.wav")
    tester = NoiseRobustnessTester(device, field_noise_path=noise_path)
    noise_results = tester.evaluate_all_conditions(model, test_dl)

    res_final = {"acc": acc_clean, "uar": uar_clean, "f1": f1_clean, "robustness": noise_results}
    
    with open("eval_result_adv.json", 'w') as f:
        json.dump(res_final, f, indent=4)
        
    print(f"\n[Final] Clean Acc: {acc_clean:.4f} | Noise Results Saved.")
    return acc_clean, uar_clean, f1_clean

if __name__ == "__main__":
    evaluate()