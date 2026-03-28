# -*- coding: utf-8 -*-
import os
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from Data.audio_augmentation import AudioAugment
from Data.mixup_augment import MixupAugmentor


class BirdsDS(Dataset):
    def __init__(self, root_path, phase: str = "train"):
        self.root_path = root_path
        txt_file = os.path.join(root_path, f"{phase}_set.txt")
        with open(txt_file) as f:
            self.files = [line.strip() for line in f if line.strip()]

        self.mel_spec_extractor = nn.Sequential(
            MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128),
            AmplitudeToDB()
        )

        self.class_dict = {
            "agelaius_phoeniceus": 0,
            "molothrus_ater": 1,
            "tringa_semipalmata": 2,
            "cardinalis_cardinalis": 3,
            "setophaga_aestiva": 4,
            "turdus_migratorius": 5,
            "certhia_americana": 6,
            "setophaga_ruticilla": 7,
            "corvus_brachyrhynchos": 8,
            "spinus_tristis": 9,
        }

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_rel = self.files[idx]
        file_abs = os.path.abspath(os.path.join("Data", file_rel.replace("\\", "/")))
        waveform, sr = torchaudio.load(file_abs)

        spec = self.mel_spec_extractor(waveform)  # [1, 128, T]
        label_txt = (
            os.path.basename(os.path.dirname(file_rel.replace("\\", "/")))
            .lower()
            .replace(" ", "_")
        )
        label = self.class_dict[label_txt]
        return spec, label, label_txt


class BirdsDS_IMG(BirdsDS):
    def __init__(self, root_path, phase: str = "train"):
        super().__init__(root_path, phase)

        self.audio_aug = AudioAugment(enable_classes={1, 2, 6})

        self.label_lookup = {}
        for f in self.files:
            f_norm = f.replace("\\", "/")
            cls = (
                os.path.basename(os.path.dirname(f_norm))
                .strip()
                .lower()
                .replace(" ", "_")
            )
            if cls in self.class_dict:
                self.label_lookup[f_norm] = self.class_dict[cls]
            else:
                print(f"[Warn] Unkown {cls!r}, skip {f}.")

        self.mixup_aug = MixupAugmentor(
            extractor=self.mel_spec_extractor,
            alpha=0.4,
            enable_classes={1, 2, 6},
        )

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        spec, label, _ = super().__getitem__(idx)
        
        spec = spec.squeeze(0)

        target_len = 251
        if spec.shape[-1] < target_len:
            spec = F.pad(spec, (0, target_len - spec.shape[-1]), mode='constant', value=0)
        elif spec.shape[-1] > target_len:
            spec = spec[:, :target_len]

        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)

        file_rel = self.files[idx].replace("\\", "/")
        file_abs = os.path.abspath(os.path.join("Data", file_rel))
        waveform, sr = torchaudio.load(file_abs)

        spec = self.audio_aug(
            spec, label, file_path=file_abs, waveform=waveform, sample_rate=sr
        )

        spec = self.mixup_aug(spec, label, file_rel, self.files, self.label_lookup)

        return spec, label, file_rel 

    
    

if __name__ == "__main__":
    ds = BirdsDS_IMG(root_path="meta-v02/2", phase="train")
    print(len(ds), "samples")
    x, y, z = ds[0]
    print("Spec shape:", x.shape, "Label:", y, "File path:", z)
