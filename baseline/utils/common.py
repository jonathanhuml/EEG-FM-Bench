import os
import random
from typing import Union

import numpy as np
import torch
from torch import nn


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Conv1dWithConstraint(nn.Conv1d):
    """
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for
    EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    """

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    """
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based
    brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    """
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class ZScoreNorm:
    """
    Sample-wise z-score normalization for time series data.

    Normalizes each channel independently across the temporal dimension.
    This is critical for domain adaptation when using time series foundation models.

    Parameters
    ----------
    eps : float, default=1e-8
        Small constant for numerical stability.
    per_channel : bool, default=True
        If True, normalize each channel independently.
        If False, normalize across all channels together.
    """

    def __init__(self, eps: float = 1e-8, per_channel: bool = True):
        self.eps = eps
        self.per_channel = per_channel

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.per_channel:
            # Normalize along last dimension (time)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
        else:
            # Normalize across all channels and time
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True)

        return (x - mean) / (std + self.eps)
