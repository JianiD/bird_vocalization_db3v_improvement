import os
import torch
from Data.bird_ds import BirdsDS_IMG as BirdsDS
from torch.utils.data import DataLoader
from hydra import compose, initialize

import numpy as np
import tqdm
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score

def report_metrics(pred_aggregate_, gold_aggregate_):
    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))

    print(classification_report(gold_aggregate_, pred_aggregate_, zero_division=0.0))
    print(confusion_matrix(gold_aggregate_, pred_aggregate_))

def evaluate(cfg, slurm_job_id, model=None):
    for idx in range(0, 3, 1):
        temp = os.path.join(cfg.meta.result, slurm_job_id + f'_{idx}')
        if os.path.exists(temp):
            print(f'evaluted on {slurm_job_id}_{idx}')
            slurm_job_id = f'{slurm_job_id}_{idx}'
            break
    
    cfg.meta.result = os.path.join(cfg.meta.result, slurm_job_id + '/ckpt')
    
    if model is None:
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
        
        ckpt_path = os.path.join(cfg.meta.result, "best_acc.pth.tar")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded checkpoint from {ckpt_path}")
        model.to(device)
    else:
        print('Evaluate on training models')
    
    ds = BirdsDS(root_path=cfg.evaluation.ds, phase='test')
    dl = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)

    counter = 0
    correct = 0
    total = 0
    pred_aggregate = []
    gold_aggregate = []
    model.eval()

    for (batch, label) in tqdm.tqdm(
            dl,
            total=len(dl),
            desc="Evaluate",
            disable=tqdm_disable
    ):
        with torch.no_grad():
            batch = batch.to(device)
            label = label.to(device)

            out, _ = model(batch)
        del batch
        counter += 1
        correct += sum(np.argmax(out.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        pred_aggregate.extend(np.argmax(out.detach().cpu().numpy(), axis=1).tolist())
        gold_aggregate.extend(label.detach().cpu().numpy().tolist())

    acc = correct / total
    report_metrics(pred_aggregate, gold_aggregate)
    uar = recall_score(gold_aggregate, pred_aggregate, average='macro')
    f1 = f1_score(gold_aggregate, pred_aggregate, average='macro')

    print('Acc:UAR:F1: ') 
    print(f'{acc:.4f}\t{uar:.4f}\t{f1:.4f}')
    print('-' * 64)
    return acc, uar, f1

if __name__ == "__main__":
    tqdm_disable = False if os.getenv('MODEL_NAME') == None else True

    with initialize(config_path="Config"):
        args = compose(config_name="config_tdnn")
    slurm_job_id = 'local'
    print(f'{slurm_job_id} will be evaluated on target dataset')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluate(args, slurm_job_id=slurm_job_id)
    print('=' * 60)