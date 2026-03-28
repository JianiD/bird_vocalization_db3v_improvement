import os
import torch
import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from sklearn.metrics import recall_score, classification_report

from Data.bird_ds import BirdsDS_IMG as BirdsDS

NAME_TO_ID = {
    "Red-winged Blackbird": 0,    # agelaius_phoeniceus
    "Brown-headed Cowbird": 1,    # molothrus_ater
    "Willet": 2,                  # tringa_semipalmata
    "Northern Cardinal": 3,       # cardinalis_cardinalis
    "American Yellow Warbler": 4, # setophaga_aestiva
    "American Robin": 5,          # turdus_migratorius
    "Brown Creeper": 6,           # certhia_americana
    "American Redstart": 7,       # setophaga_ruticilla
    "American Crow": 8,           # corvus_brachyrhynchos
    "American Goldfinch": 9       # spinus_tristis
}

def main():

    analyzer = Analyzer()
    
    dataset_path = "meta-v02/2"
    val_ds = BirdsDS(root_path=dataset_path, phase='test')
    
    golds = [] # real label
    preds = []
    unidentified_samples = []
    
    print(f"Test BirdNET...")
    print(f"Dataset: {dataset_path} | Samples total number: {len(val_ds)}")

    for i in range(len(val_ds)):
        _, label, file_rel = val_ds[i]
        file_abs = os.path.abspath(os.path.join("Data", file_rel))
        
       
        recording = Recording(analyzer, file_abs)
        recording.analyze()
        
        best_id = -1
        max_conf = 0.0
        
        if not recording.detections:
            pass 
        else:
            for dt in recording.detections:
                name = dt['common_name']
                conf = dt['confidence']
                if name in NAME_TO_ID and conf > max_conf:
                    max_conf = conf
                    best_id = NAME_TO_ID[name]
        
        if best_id == -1:
            unidentified_samples.append({
                "path": file_rel,
                "label": label,
                "detected": [d['common_name'] for d in recording.detections] if recording.detections else "None"
            })

        preds.append(best_id)
        golds.append(label)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(val_ds)}")

    # Unidentified sample is considered as mistake classified
    target_ids = list(range(10))
    target_names = list(NAME_TO_ID.keys())
    
    uar = recall_score(golds, preds, labels=target_ids, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("                BIRDNET BENCHMARK RESULT")
    print("="*50)
    print(f"UAR: {uar:.4f}")
    print(f"Unidentified: {len(unidentified_samples)} / {len(val_ds)}")
    print("-" * 50)
    
    
    report = classification_report(
        golds, 
        preds, 
        labels=target_ids, 
        target_names=target_names, 
        zero_division=0
    )
    print("Detailed classification (Target Species Only):")
    print(report)

    if unidentified_samples:
        print("\nThe first 5 unidentified samples:")
        for sample in unidentified_samples[:5]:
            print(f"  File: {sample['path']} | Label: {sample['label']} | BirdNET caught: {sample['detected']}")

if __name__ == "__main__":
    main()