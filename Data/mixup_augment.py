import os
import random
import numpy as np
import torch
import torchaudio


class MixupAugmentor:
    def __init__(self, extractor, alpha: float = 0.4, enable_classes=None):
        self.extractor = extractor
        self.alpha = alpha
        self.enable_classes = enable_classes or set()

    def __call__(self,
                 spec: torch.Tensor,
                 label: int,
                 file_rel: str,
                 all_files: list[str],
                 label_lookup: dict[str, int]
                 ) -> torch.Tensor:
        
        if label not in self.enable_classes:
            return spec

        same_class_files = [
            f for f in all_files
            if label_lookup.get(f) == label and f != file_rel
        ]
        if not same_class_files:
            return spec  

        
        f2_rel = random.choice(same_class_files)
        f2_path = os.path.abspath(os.path.join("Data", f2_rel.replace("\\", "/")))
        try:
            wav2, _ = torchaudio.load(f2_path)
            spec2 = self.extractor(wav2)
            spec2 -= spec2.min()
            spec2 /= spec2.max() + 1e-9
        except Exception as e:
            print(f"[Warn]  {f2_path} fail, skip: {e}")
            return spec

        lam = np.random.beta(self.alpha, self.alpha)
        return lam * spec + (1.0 - lam) * spec2
