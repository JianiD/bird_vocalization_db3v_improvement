import os
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score
from Data.bird_ds import BirdsDS_IMG as BirdsDS
from train_tdnn_adv import TDNNWithDomainAdversarial
from hydra.utils import get_original_cwd
import json
import tempfile
import shutil


@hydra.main(version_base=None, config_path="Config", config_name="config_tdnn")
def evaluate(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Evaluation Settings ===")
    print(f"Dataset path: {cfg.evaluation.ds}")
    print(f"Result/model path: {cfg.meta.result}")

    
    test_ds = BirdsDS(cfg.evaluation.ds, phase='test')
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=cfg.hparams.bs,
                                          shuffle=False)

    tdnn_type = cfg.model.tdnn
    if tdnn_type == 'tdnn_BN':
        from TDNN2 import tdnn_BN as model_zoo
    elif tdnn_type == 'tdnn_IFN':
        from TDNN2 import tdnn_IFN as model_zoo
    elif tdnn_type == 'tdnn_LSTM':
        from TDNN2 import tdnn_LSTM as model_zoo
    elif tdnn_type == 'tdnn_both':
        from TDNN2 import tdnn_both as model_zoo
    elif tdnn_type == 'tdnn_GW':
        from TDNN2 import tdnn_GW as model_zoo
    elif tdnn_type == 'tdnn_TN':
        from TDNN2 import tdnn_TN as model_zoo
    elif tdnn_type == 'tdnn_RIFN':
        from TDNN2 import tdnn_RIFN as model_zoo
    else:
        raise ValueError(f"Unsupported TDNN type: {tdnn_type}")

    base_model = model_zoo.TDNN(feat_dim=128,
                                 embedding_size=512,
                                 num_classes=cfg.model.num_classes)

    model = TDNNWithDomainAdversarial(base_model,
                                       num_domains=3,
                                       num_classes=cfg.model.num_classes)

    
    ckpt_dir = cfg.meta.result
    candidates = ["best_acc.pth.tar","best_uar.pth.tar", "last.pth.tar"]
    ckpt_path = next((os.path.join(ckpt_dir, f) for f in candidates
                      if os.path.exists(os.path.join(ckpt_dir, f))), None)
    
    print(f"Looking for checkpoint under: {cfg.meta.result}")
    print(f"Candidate files: {candidates}")
    print(f"Found: {ckpt_path if ckpt_path else 'None'}")

    if ckpt_path is None:
        print(f"\n No checkpoint found under {ckpt_dir}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\n Loaded checkpoint: {ckpt_path}")

    model.to(device)
    model.eval()

    
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch, label, _ in tqdm(test_dl, desc="Evaluate"):
            batch, label = batch.to(device), label.to(device)
            out, _ = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n[Eval Results] Acc: {acc:.4f}, UAR: {uar:.4f}, F1: {f1:.4f}")

    json_target = os.path.join(get_original_cwd(), "eval_result_adv.json")
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(json_target)) as tf:
        json.dump({"acc": acc, "uar": uar, "f1": f1}, tf)
        tmpname = tf.name
    shutil.move(tmpname, json_target)

    return acc, uar, f1


if __name__ == "__main__":
    evaluate()
