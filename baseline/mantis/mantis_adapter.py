"""
Mantis Adapter that inherits from AbstractDatasetAdapter.

Mantis processes time series data with Token Generation + ViT:
1. Requires fixed sequence length (512 by default, multiple of 32)
2. Processes each channel independently through the encoder
3. Concatenates channel embeddings for classification

This adapter supports:
- Multi-variate EEG data (multiple channels)
- Sequence length normalization to Mantis requirements
- Sample-wise z-score normalization
"""

import logging
from typing import List, Dict, Any

from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin
from baseline.utils.common import ZScoreNorm
from common.utils import ElectrodeSet

logger = logging.getLogger("baseline")


class MantisDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """Mantis dataset adapter that handles EEG data for the Mantis foundation model."""
    
    def __init__(
        self, 
        dataset: HFDataset, 
        dataset_names: List[str], 
        dataset_configs: List[str],
        target_seq_len: int = 512,
        use_zscore: bool = True,
    ):
        """
        Initialize Mantis dataset adapter.
        
        Parameters
        ----------
        dataset : HFDataset
            The HuggingFace dataset to adapt.
        dataset_names : List[str]
            Names of datasets.
        dataset_configs : List[str]
            Configuration names.
        target_seq_len : int
            Target sequence length (must be multiple of 32).
        use_zscore : bool
            Whether to apply z-score normalization.
        """
        self.target_seq_len = target_seq_len
        self.use_zscore = use_zscore
        self.normalizer = ZScoreNorm() if use_zscore else None
        self.electrode_set = ElectrodeSet()

        super().__init__(dataset, dataset_names, dataset_configs)
        self.model_name = 'mantis'
        self.scale = 1.0  # Z-score handles scaling

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported (standard EEG channels)."""
        return self.electrode_set.Electrodes
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample for Mantis.
        
        Mantis expects input of shape (batch, n_channels, seq_len) where:
        - n_channels: Number of channels (reduced if using reducer)
        - seq_len: Fixed sequence length (512 by default, multiple of 32)
        
        Each channel is processed independently through the encoder.
        """
        # Get base processed sample
        result = super()._process_sample(sample)
        
        data = result['data']  # Shape: (n_channels, n_timepoints)
        
        # Apply z-score normalization per channel
        if self.normalizer is not None:
            data = self.normalizer(data)

        # Build result dictionary
        # Mantis format: (n_channels, seq_len)
        result['data'] = data
        
        return result


class MantisDataLoaderFactory(AbstractDataLoaderFactory, StandardEEGChannelsMixin):
    """Mantis DataLoader factory that inherits from AbstractDataLoaderFactory."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        target_seq_len: int = 512,
        use_zscore: bool = True,
    ):
        """
        Initialize Mantis DataLoader factory.
        
        Parameters
        ----------
        batch_size : int
            Batch size for data loading.
        num_workers : int
            Number of data loading workers.
        seed : int
            Random seed.
        target_seq_len : int
            Target sequence length (must be multiple of 32).
        use_zscore : bool
            Whether to apply z-score normalization.
        """
        super().__init__(batch_size, num_workers, seed)
        self.target_seq_len = target_seq_len
        self.use_zscore = use_zscore
    
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> MantisDatasetAdapter:
        return MantisDatasetAdapter(
            dataset=dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            target_seq_len=self.target_seq_len,
            use_zscore=self.use_zscore,
        )
