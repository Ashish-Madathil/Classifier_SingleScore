#Training 60%; Validation 20%; Testing 20%
#Class distribution is the same across different datasets

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from embryo_dataset import EmbryoDataset


def split_indices(n, train_pct, val_pct, seed=None, stratify=None):
    """ Return indices for train, validation, and test subsets, ensuring
        that class distribution remains the same across different datasets.
    """
    np.random.seed(seed)
    indices = np.random.permutation(n)

    if stratify is not None:
        stratify = np.array(stratify)
        sorted_idx = np.argsort(stratify)
        indices = indices[sorted_idx]

    train_end = int(train_pct * n)
    val_end = int(val_pct * n) + train_end

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices

full_dataset = EmbryoDataset(txt_path="ed4_as_target.txt", transform=transform)
train_indices, val_indices, test_indices = split_indices(len(full_dataset), train_pct=0.6, val_pct=0.2, seed=42, stratify=full_dataset.label_list)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(dataset=full_dataset, batch_size=32, sampler=test_sampler)




