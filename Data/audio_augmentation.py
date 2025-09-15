import random
import torch
import numpy as np
import librosa

class AudioAugment:
    def __init__(self, enable_classes=None):
        # type
        self.enable_classes = enable_classes if enable_classes is not None else set()

    def augment_waveform(self, waveform, sample_rate):
        y = waveform.numpy().squeeze()

        # Pitch shift
        if random.random() < 0.5:
            y = librosa.effects.pitch_shift(y=y, sr=sample_rate, n_steps=random.choice([-2, -1, 1, 2]))


        # Time stretch
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            
            new_sr = int(sample_rate * random.uniform(0.9, 1.1))
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=new_sr)
            y = librosa.resample(y, orig_sr=new_sr, target_sr=sample_rate)


        return torch.from_numpy(y).unsqueeze(0)

    def augment_spec(self, spec):
        # noise
        if random.random() < 0.5:
            noise = torch.randn_like(spec) * 0.02
            spec += noise

        # frequency masking
        if spec.size(0) > 10 and random.random() < 0.5:
            f = torch.randint(0, spec.size(0) - 10, (1,))
            spec[f:f+10, :] = 0

        # time masking
        if spec.size(1) > 10 and random.random() < 0.5:
            t = torch.randint(0, spec.size(1) - 10, (1,))
            spec[:, t:t+10] = 0

        return spec

    def __call__(self, spec, label, file_path=None, waveform=None, sample_rate=16000):
        if label not in self.enable_classes:
            return spec

        # waveform augmentation
        if waveform is not None:
            waveform = self.augment_waveform(waveform, sample_rate)

        # spec augmentation
        spec = self.augment_spec(spec)
        
        if random.random() < 0.01:
            print(f"Augment applied to class {label}")


        return spec
