import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import recall_score, classification_report

from Data.bird_ds import BirdsDS_IMG as BirdsDS

PERCH_IDX_TO_ID = {
    372: 0,   # red-winged blackbird -> agelaius_phoeniceus
    655: 1,   # brown-headed cowbird -> molothrus_ater
    10015: 2, # willet -> tringa_semipalmata
    632: 3,   # northern cardinal -> cardinalis_cardinalis
    790: 4,   # yellow warbler -> setophaga_aestiva
    312: 5,   # american robin -> turdus_migratorius
    1102: 6,  # brown creeper -> certhia_americana
    771: 7,   # american redstart -> setophaga_ruticilla
    565: 8,   # american crow -> corvus_brachyrhynchos
    960: 9    # american goldfinch -> spinus_tristis
}

TARGET_NAMES = [
    "Red-winged Blackbird", "Brown-headed Cowbird", "Willet", 
    "Northern Cardinal", "Yellow Warbler", "American Robin", 
    "Brown Creeper", "American Redstart", "American Crow", 
    "American Goldfinch"
]

def main():
    print("Loading Google Perch model (V4)...")
    model = hub.load('https://tfhub.dev/google/bird-vocalization-classifier/4')
    
    dataset_path = "meta-v02/2"
    test_ds = BirdsDS(root_path=dataset_path, phase='test')
    
    golds = []
    preds = []
    
    print(f"Test Perch...")
    print(f"Test Dataset: {dataset_path} | Samples total number: {len(test_ds)}")

    for i in range(len(test_ds)):
        _, label, file_rel = test_ds[i]
        file_abs = os.path.abspath(os.path.join("Data", file_rel))
        
        try:
            audio, _ = librosa.load(file_abs, sr=32000)
        except Exception as e:
            print(f"Fail: {file_abs}, error: {e}")
            continue
        
        target_samples = 160000 
        step_samples = 32000     
        all_window_probs = []

        for start in range(0, len(audio) - target_samples + 1, step_samples):
            chunk = audio[start : start + target_samples]
            logits, _ = model.infer_tf(chunk[np.newaxis, :].astype(np.float32))
            probs = tf.nn.softmax(logits).numpy() 
            all_window_probs.append(np.mean(probs, axis=0))
        
        if all_window_probs:
            mean_probs = np.mean(all_window_probs, axis=0)
        else:
            mean_probs = np.zeros(15000)

        best_id = -1
        max_v = -1.0
        for perch_idx, our_id in PERCH_IDX_TO_ID.items():
            if perch_idx < len(mean_probs):
                current_prob = mean_probs[perch_idx]
                if current_prob > max_v:
                    max_v = current_prob
                    best_id = our_id
        
        if i < 5:
            print(f"Debugging [Sample {i}]: Predicted ID={best_id}, Maximum Probability={max_v:.8f}")
        preds.append(best_id)
        golds.append(label)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(test_ds)}")


    target_ids = list(range(10))
    uar = recall_score(golds, preds, labels=target_ids, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("                GOOGLE PERCH BENCHMARK (TEST SET)")
    print("="*50)
    print(f"UAR: {uar:.4f}")
    print("-" * 50)
    
    report = classification_report(
        golds, 
        preds, 
        labels=target_ids, 
        target_names=TARGET_NAMES, 
        zero_division=0
    )
    print("Detailed classification:")
    print(report)

if __name__ == "__main__":
    main()