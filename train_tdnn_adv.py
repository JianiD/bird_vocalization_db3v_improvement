import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from hydra import compose, initialize
from omegaconf import (
    OmegaConf
)

from Utils import adjust_learning_rate
from Data.bird_ds import BirdsDS_IMG as BirdsDS
import time
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score

from imbalanced_utils import SAMPLE_COUNTS, build_weighted_sampler, build_weighted_loss
import torch.nn.functional as F


from torch import nn
from torch.autograd import Function


ALPHA_DOM   = 0.5        # domain-loss weight
GRL_WARMUP  = True       # whether to use λ_GRL linear warm-up


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


def save_model(save_path, current_model, current_epoch, marker, timestamp):
    save_path = os.path.join(save_path, timestamp)
    save_to = os.path.join(save_path, '{}_{}.pkl'.format(marker, current_epoch))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(current_model, save_to)
    print('<== Model is saved to {}'.format(save_to))


def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))


def print_flags(cfg):
    print('--------------------------- Flags -----------------------------')
    for flag in cfg.asdic():
        print('{} : {}'.format(flag, getattr(cfg, flag)))


def report_metrics(pred_aggregate_, gold_aggregate_):
    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))

    print(classification_report(gold_aggregate_, pred_aggregate_, zero_division=0.0))
    print(confusion_matrix(gold_aggregate_, pred_aggregate_))


def create_ds(cfg):
    ds_tr = BirdsDS(root_path=cfg.meta.train_ds, phase='train')
    ds_val = BirdsDS(root_path=cfg.meta.train_ds, phase='val')
    return ds_tr, ds_val


def create_tr_dl(dataset, batch_size=4):
    sampler = build_weighted_sampler(dataset, SAMPLE_COUNTS)
    dl_tr = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    # dl_tr = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl_tr


def create_val_dl(dataset, batch_size=4):
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return val_dl


def create_dl(cfg):
    ds_tr, ds_val = create_ds(cfg)
    dl_tr = create_tr_dl(ds_tr, cfg.hparams.bs)
    dl_val = create_val_dl(ds_val, cfg.hparams.bs)
    return dl_tr, dl_val


def training_setting(model, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # loss_fn = CrossEntropyLoss()
    loss_fn = build_weighted_loss(SAMPLE_COUNTS, device)

    return optimizer, loss_fn



def extract_region_id(path):
    path = path.lower().replace("\\", "/")
    if "dataset/3" in path:
        return 0
    elif "dataset/2" in path:
        return 1
    elif "dataset/1" in path:
        return 2
    else:
        return 1  # Default as D2


def train(dl, optimizer, loss_fn, epoch, total_epochs, log_freq=10):
    losses = 0.
    counter = 1
    tmp_losses = 0.
    tmp_counter = 0
    correct = 0
    total = 0
    tmp_correct = 0
    tmp_total = 0
    model.train()
    
    # Calculate λ_GRL
    if GRL_WARMUP:
        lambda_grl = min(1.0, 0.1 + 0.9 * epoch / total_epochs) 
    else:
        lambda_grl = 1.0
    
    for idx, (batch, label, path_list) in enumerate(dl):
        batch = batch.to(device)
        label = label.to(device)

        region_labels = torch.tensor([extract_region_id(p) for p in path_list], device=device)
        
        if idx < 3:
            print(f"[DEBUG] Batch {idx}")
            for i in range(min(4, len(path_list))):
                print(f"   Path: {path_list[i]} => Region ID: {extract_region_id(path_list[i])}")
                
                
        label_out, domain_out = model(batch, lambda_grl=lambda_grl)


        loss_cls = F.cross_entropy(label_out, label)
        loss_domain = F.cross_entropy(domain_out, region_labels)
        # loss = loss_cls + 0.1 * loss_domain
        loss = loss_cls + ALPHA_DOM * loss_domain


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        counter += 1
        correct += sum(np.argmax(label_out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        tmp_losses += loss.item()
        tmp_counter += 1
        tmp_correct += sum(np.argmax(label_out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        tmp_total += len(label.detach().cpu().numpy())

        if idx % log_freq == 0:
            print('  [{}][{}] loss: {:.4f}, Acc: {:.4}'.format(epoch, idx,
                                                               tmp_losses / tmp_counter,
                                                               tmp_correct / tmp_total))
            print(f"[{epoch}][{idx}] λ_GRL={lambda_grl:.2f}  "
                f"Cls:{loss_cls.item():.4f}  Dom:{loss_domain.item():.4f}  "
                f"Total:{loss.item():.4f}  Acc:{tmp_correct/tmp_total:.4f}")

            tmp_losses = 0.
            tmp_counter = 0
            tmp_correct = 0
            tmp_total = 0

    print('##> [{}] Train loss: {:.4f}, Acc: {:.4}'.format(epoch, losses / counter, correct / total))
    correct = 0
    total = 0
    tmp_correct = 0
    tmp_total = 0
    model.train()
    

def eval(dl, loss_fn):
    losses = 0.
    counter = 0
    correct = 0
    total = 0
    pred_aggregate = []
    gold_aggregate = []
    model.eval()

    for idx, (batch, label, _) in enumerate(dl):
        with torch.no_grad():
            batch = batch.to(device)
            label = label.to(device)

            out, xvector = model(batch)
        del batch
        loss = loss_fn(out, label)
        losses += loss.item()
        counter += 1
        correct += sum(np.argmax(out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        pred_aggregate.extend(np.argmax(out.detach().cpu().numpy(), axis=1).tolist())
        gold_aggregate.extend(label.detach().cpu().numpy().tolist())
    acc = correct / total
    print('==> Val loss: {:.4f}, Acc: {:.5}'.format(losses / counter, acc))
    report_metrics(pred_aggregate, gold_aggregate)
    uar = recall_score(gold_aggregate, pred_aggregate, average='macro')
    print('#=> Val UAR: {:.4f}'.format(uar))
    print('-' * 64)
    return acc, uar


if __name__ == "__main__":
    setup_seed(10)
    slurm_job_id = 'local' if os.getenv('MODEL_NAME') == None else os.getenv('MODEL_NAME')
    with initialize(config_path="Config"):
        args = compose(config_name="config_tdnn")
        print(args)
    
    args.meta.result = os.path.join(args.meta.result, slurm_job_id)
    experiment_folder = args.meta.result

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        os.makedirs(experiment_folder + '/ckpt')
        os.makedirs(experiment_folder + '/out')
        os.makedirs(experiment_folder + f'/{args.hparams.md_name}')

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model.tdnn == 'tdnn_BN':
        from TDNN2 import tdnn_BN
        model = tdnn_BN.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_IFN':
        from TDNN2 import tdnn_IFN
        model = tdnn_IFN.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_LSTM':
        from TDNN2 import tdnn_LSTM
        model = tdnn_LSTM.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_both':
        from TDNN2 import tdnn_both
        model = tdnn_both.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_GW':
        from TDNN2 import tdnn_GW
        model = tdnn_GW.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_TN':
        from TDNN2 import tdnn_TN
        model = tdnn_TN.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)
    if args.model.tdnn == 'tdnn_RIFN':
        from TDNN2 import tdnn_RIFN
        model = tdnn_RIFN.TDNN(feat_dim=128, embedding_size=512, num_classes=args.model.num_classes)

    
    model = TDNNWithDomainAdversarial(model, num_domains=3, num_classes=args.model.num_classes).to(device)
    print_nn(model)
    print('-' * 64)

    tr_dl, val_dl = create_dl(args)
    optimizer, loss_fn = training_setting(model, args.hparams.lr)
    best_acc = 0.
    best_uar = 0.
    best_epoch = 1
    for epoch in range(1, args.hparams.epoch + 1):
        adjust_learning_rate(optimizer, epoch, args.hparams.lr)
        # train(tr_dl, optimizer, loss_fn, epoch, args.hparams.log_freq)
        train(tr_dl, optimizer, loss_fn,
                epoch, total_epochs=args.hparams.epoch,
                log_freq=args.hparams.log_freq)

        val_acc, val_uar = eval(val_dl, loss_fn)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(experiment_folder + '/ckpt', 'best_acc.pth.tar'))
            print(f'ACC BEST model update with acc {best_acc} at epoch {epoch}')
        if val_uar > best_uar:
            best_uar = val_uar
            torch.save(model.state_dict(), os.path.join(experiment_folder + '/ckpt', 'best_uar.pth.tar'))
            print(f'UAR BEST model update with uar {best_uar} at epoch {epoch}')
        
        torch.save(model.state_dict(), os.path.join(experiment_folder + '/ckpt', 'last.pth.tar'))
        print(f'[{epoch}] last model is saved!')
