"""
Moment model integration for EEG Foundation Model framework.

Moment is a multi-channel time series foundation model that uses:
- T5 Encoder: Pretrained language model backbone
- Patch Embedding: Tokenizes time series into patches
- RevIN: Reversible instance normalization

Moment natively supports multi-channel input, making it suitable for EEG data.
"""

from .model import MOMENT, MOMENTPipeline, TimeseriesOutputs, TASKS
from .moment_config import (
    MomentConfig, 
    MomentDataArgs, 
    MomentModelArgs, 
    MomentTrainingArgs,
    MomentLoggingArgs,
)
from .moment_trainer import MomentTrainer, MomentEncoder, MomentUnifiedModel
from .moment_adapter import MomentDatasetAdapter, MomentDataLoaderFactory

__all__ = [
    'MOMENT',
    'MOMENTPipeline',
    'TimeseriesOutputs',
    'TASKS',
    'MomentConfig',
    'MomentDataArgs',
    'MomentModelArgs',
    'MomentTrainingArgs',
    'MomentLoggingArgs',
    'MomentTrainer',
    'MomentEncoder',
    'MomentUnifiedModel',
    'MomentDatasetAdapter',
    'MomentDataLoaderFactory',
]
