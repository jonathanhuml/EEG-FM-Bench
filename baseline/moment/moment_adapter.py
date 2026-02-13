"""
MOMENT Adapter that inherits from AbstractDatasetAdapter.

MOMENT processes time series data with T5 encoder backbone:
1. Requires fixed sequence length (512 by default, compatible with patch size)
2. Uses RevIN normalization for instance normalization
3. Processes each channel through patchification then T5 encoder

This adapter supports:
- Multi-variate EEG data (multiple channels)
- Sequence length normalization to MOMENT requirements
- Sample-wise normalization (handled by model's RevIN)
"""

import logging
from typing import List

from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin
from common.utils import ElectrodeSet

logger = logging.getLogger("baseline")


class MomentDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """MOMENT dataset adapter that handles EEG data for the MOMENT foundation model."""
    
    def __init__(
        self, 
        dataset: HFDataset, 
        dataset_names: List[str], 
        dataset_configs: List[str],
        target_seq_len: int = 512,
    ):
        self.target_seq_len = target_seq_len
        self.electrode_set = ElectrodeSet()

        super().__init__(dataset, dataset_names, dataset_configs)
        self.model_name = 'moment'
        self.scale = 1.0  # RevIN handles normalization in the model

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported (standard EEG channels)."""
        return self.electrode_set.Electrodes


class MomentDataLoaderFactory(AbstractDataLoaderFactory, StandardEEGChannelsMixin):
    """MOMENT DataLoader factory that inherits from AbstractDataLoaderFactory."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        target_seq_len: int = 512,
    ):
        super().__init__(batch_size, num_workers, seed)
        self.target_seq_len = target_seq_len
    
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> MomentDatasetAdapter:
        return MomentDatasetAdapter(
            dataset=dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            target_seq_len=self.target_seq_len,
        )
