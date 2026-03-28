import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from hydra import compose, initialize
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
import time

from Utils import adjust_learning_rate
from Data.bird_ds import BirdsDS_IMG as BirdsDS
from imbalanced_utils import SAMPLE_COUNTS, build_weighted_sampler, build_weighted_loss

ALPHA_DOM   = 0.5        # domain-loss weight
GRL_WARMUP  = True       # whether to use λ_GRL linear warm-up
CYC_WEIGHT  = 0.1        # weight for CycleGAN generated samples
CYC_MATCH_KWS = ("train",)  # keywords to identify CycleGAN samples

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)

class TDNNWithDomainAdversarial(nn.Module):
    def __init__(self, base_model, num_domains=3, num_classes=10):
        super().__init__()
        self.base_model = base_model
        self.label_head = nn.Linear(512, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, x, lambda_grl=1.0):
        out, xvector = self.base_model(x)
        label_out = self.label_head(xvector)
        rev_feat = grad_reverse(xvector, lambda_grl)
        domain_out = self.domain_head(rev_feat)
        return label_out, domain_out

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))

def report_metrics(pred_aggregate_, gold_aggregate_):
    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))
    print(classification_report(gold_aggregate_, pred_aggregate_, zero_division=0.0))
    print(confusion_matrix(gold_aggregate_, pred_aggregate_))

def create_dl(cfg):
    ds_tr = BirdsDS(root_path=cfg.meta.train_ds, phase='train')
    ds_val = BirdsDS(root_path=cfg.meta.train_ds, phase='val')
    
    sampler = build_weighted_sampler(ds_tr, SAMPLE_COUNTS)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.hparams.bs, sampler=sampler, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.hparams.bs, shuffle=False, drop_last=True)
    return dl_tr, dl_val

def training_setting(model, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = build_weighted_loss(SAMPLE_COUNTS, device)
    return optimizer, loss_fn

def extract_region_id(path):
    path = path.lower().replace("\\", "/")
    if "dataset/3" in path: return 0
    elif "dataset/2" in path: return 1
    elif "dataset/1" in path: return 2
    else: return 1  

def is_cyclegan_sample(path: str) -> bool:
    path_lc = path.lower()
    return any(kw in path_lc for kw in CYC_MATCH_KWS)

def train(dl, optimizer, loss_fn, epoch, total_epochs, log_freq=10):
    losses, counter, correct, total = 0.0, 0, 0, 0
    tmp_losses, tmp_counter, tmp_correct, tmp_total = 0.0, 0, 0, 0
    model.train()

    lambda_grl = min(1.0, 0.1 + 0.9 * epoch / total_epochs) if GRL_WARMUP else 1.0

    for idx, (batch, label, path_list) in enumerate(dl):
        batch, label = batch.to(device), label.to(device)
        region_labels = torch.tensor([extract_region_id(p) for p in path_list], device=device)

        label_out, domain_out = model(batch, lambda_grl=lambda_grl)

        # GAN Weighted Classification Loss
        ce_vec = F.cross_entropy(label_out, label, reduction='none')
        weights = torch.tensor([CYC_WEIGHT if is_cyclegan_sample(p) else 1.0 for p in path_list], 
                               device=device, dtype=ce_vec.dtype)
        loss_cls = (weights * ce_vec).mean()
        
        loss_domain = F.cross_entropy(domain_out, region_labels)
        loss = loss_cls + ALPHA_DOM * loss_domain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(label_out, dim=1).detach().cpu().numpy()
            targets = label.detach().cpu().numpy()
            batch_correct = sum(preds == targets)
            
            losses += loss.item()
            counter += 1
            correct += batch_correct
            total += len(targets)
            
            tmp_losses += loss.item()
            tmp_counter += 1
            tmp_correct += batch_correct
            tmp_total += len(targets)

        if idx % log_freq == 0:
            print(f'  [{epoch}][{idx}] loss: {tmp_losses/tmp_counter:.4f}, Acc: {tmp_correct/tmp_total:.4f}')
            num_cyc = sum(is_cyclegan_sample(p) for p in path_list)
            print(f"     λ_GRL={lambda_grl:.2f}  Cls:{loss_cls.item():.4f}  Dom:{loss_domain.item():.4f}  Conv:{num_cyc}/{len(path_list)}")
            tmp_losses, tmp_counter, tmp_correct, tmp_total = 0.0, 0, 0, 0

    print(f'##> [{epoch}] Train loss: {losses/max(counter,1):.4f}, Acc: {correct/max(total,1):.4f}')

def eval(dl, loss_fn):
    losses, counter, correct, total = 0.0, 0, 0, 0
    pred_aggregate, gold_aggregate = [], []
    model.eval()

    for idx, (batch, label, _) in enumerate(dl):
        with torch.no_grad():
            batch, label = batch.to(device), label.to(device)
            # 调用 model 时不传 lambda_grl (或传 0)，确保与 evaluation 逻辑一致
            label_out, _ = model(batch, lambda_grl=0.0) 
            
            loss = loss_fn(label_out, label)
            preds = torch.argmax(label_out, dim=1).detach().cpu().numpy()
            targets = label.detach().cpu().numpy()

            losses += loss.item()
            counter += 1
            correct += sum(preds == targets)
            total += len(targets)
            pred_aggregate.extend(preds.tolist())
            gold_aggregate.extend(targets.tolist())

    acc = correct / max(total, 1)
    print(f'==> Val loss: {losses/max(counter,1):.4f}, Acc: {acc:.5f}')
    report_metrics(pred_aggregate, gold_aggregate)
    uar = recall_score(gold_aggregate, pred_aggregate, average='macro')
    print(f'#=> Val UAR: {uar:.4f}\n' + '-'*64)
    return acc, uar

if __name__ == "__main__":
    setup_seed(10)
    slurm_job_id = 'local' if os.getenv('MODEL_NAME') is None else os.getenv('MODEL_NAME')
    with initialize(config_path="Config"):
        args = compose(config_name="config_tdnn")
    
    experiment_folder = os.path.join(args.meta.result, slurm_job_id)
    ckpt_dir = os.path.join(experiment_folder, 'ckpt')
    for d in [ckpt_dir, os.path.join(experiment_folder, 'out'), os.path.join(experiment_folder, args.hparams.md_name)]:
        if not os.path.exists(d): os.makedirs(d)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    tdnn_map = {
        'tdnn_BN': 'tdnn_BN', 'tdnn_IFN': 'tdnn_IFN', 'tdnn_LSTM': 'tdnn_LSTM',
        'tdnn_both': 'tdnn_both', 'tdnn_GW': 'tdnn_GW', 'tdnn_TN': 'tdnn_TN', 'tdnn_RIFN': 'tdnn_RIFN'
    }
    import importlib
    module = importlib.import_module(f"TDNN2.{tdnn_map[args.model.tdnn]}")
    base = module.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)

    
    model = TDNNWithDomainAdversarial(base, num_domains=3, num_classes=args.model.num_classes).to(device)
    print_nn(model)

    tr_dl, val_dl = create_dl(args)
    optimizer, loss_fn = training_setting(model, args.hparams.lr)

    best_acc, best_uar = 0.0, 0.0
    for epoch in range(1, args.hparams.epoch + 1):
        adjust_learning_rate(optimizer, epoch, args.hparams.lr)
        train(tr_dl, optimizer, loss_fn, epoch, args.hparams.epoch, args.hparams.log_freq)
        val_acc, val_uar = eval(val_dl, loss_fn)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_acc.pth.tar'))
            print(f'ACC BEST: {best_acc:.5f} at epoch {epoch}')
        if val_uar > best_uar:
            best_uar = val_uar
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_uar.pth.tar'))
            print(f'UAR BEST: {best_uar:.5f} at epoch {epoch}')
        
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pth.tar'))