import torch
from torch.utils.data import WeightedRandomSampler
from torch.nn import CrossEntropyLoss

# SAMPLE_COUNTS = torch.tensor([54, 50, 29, 166, 9, 187, 12, 107, 123, 94], dtype=torch.float) #2
# SAMPLE_COUNTS = torch.tensor([54, 50, 128, 166, 534, 187, 260, 107, 123, 94], dtype=torch.float)  #2 after augmentation
SAMPLE_COUNTS = torch.tensor([1295, 392, 138, 778, 730, 1038, 345, 199, 645, 221], dtype=torch.float)  #1
# SAMPLE_COUNTS = torch.tensor([839, 96, 106, 1299, 297, 791, 132, 579, 435, 283], dtype=torch.float)  #3

def get_class_weights(sample_counts):
    class_weights = 1.0 / sample_counts
    return class_weights / class_weights.sum()

def build_weighted_sampler(dataset, sample_counts):
    class_weights = get_class_weights(sample_counts)
    sample_weights = [class_weights[label] for _, label, _ in dataset]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    return sampler


def build_weighted_loss(sample_counts, device):
    class_weights = get_class_weights(sample_counts).to(device)
    return CrossEntropyLoss(weight=class_weights)
