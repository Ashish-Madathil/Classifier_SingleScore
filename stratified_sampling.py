#Training 60%; Validation 20%; Testing 20%
#Class distribution is the same across different datasets

import numpy as np


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






