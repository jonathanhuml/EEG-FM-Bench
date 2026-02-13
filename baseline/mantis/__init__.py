"""
Mantis model integration for EEG Foundation Model framework.

Mantis is a univariate time series foundation model that uses:
- Token Generation Unit: Temporal convolutions + scalar statistics encoding
- ViT Unit: Transformer encoder with CLS token for representation

For multivariate EEG data, channels are processed independently through the model.
"""

from .model import Mantis8M
from .mantis_config import MantisConfig, MantisDataArgs, MantisModelArgs
from .mantis_trainer import MantisTrainer, MantisEncoder, MantisUnifiedModel
from .mantis_adapter import MantisDatasetAdapter, MantisDataLoaderFactory

__all__ = [
    'Mantis8M',
    'MantisConfig',
    'MantisDataArgs', 
    'MantisModelArgs',
    'MantisTrainer',
    'MantisEncoder',
    'MantisUnifiedModel',
    'MantisDatasetAdapter',
    'MantisDataLoaderFactory',
]
