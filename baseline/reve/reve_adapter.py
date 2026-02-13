from typing import List, Dict, Any

import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory
from baseline.reve.pos_bank import RevePositionBank
from common.utils import ElectrodeSet


class ReveDatasetAdapter(AbstractDatasetAdapter):
    """Reve dataset adapter that handles EEG data and position mapping."""

    def __init__(
            self,
            dataset: HFDataset,
            dataset_names: List[str],
            dataset_configs: List[str],
            pos_bank_dict: Dict[str, torch.Tensor],
            channel_restricted: bool = False,
    ):
        self.pos_bank: RevePositionBank
        self.electrode_set: ElectrodeSet = ElectrodeSet()
        self.channel_restricted = channel_restricted
        super().__init__(dataset, dataset_names, dataset_configs)

        self.pos_bank.load_state_dict(pos_bank_dict)

    def _setup_adapter(self):
        """Initialize Reve-specific adapter configurations."""
        self.model_name = 'reve'
        self.pos_bank = RevePositionBank()

        self._build_montage_mappings()
        self._log_adapter_info()

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by Reve (from pos_bank)."""
        if self.channel_restricted:
            return self.electrode_set.Electrodes

        if self.pos_bank is not None:
            return [ch.upper() for ch in self.pos_bank.get_all_positions()]
        return []

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample including position encoding."""
        result = super()._process_sample(sample)

        name_list = self.electrode_set.get_electrodes_name(result['chs'].tolist())

        # Get position embeddings for these channels
        pos = self.pos_bank(name_list)  # Shape: (n_channels, 3)
        result['pos'] = pos
        result.pop('chans_id')

        return result


class ReveDataLoaderFactory(AbstractDataLoaderFactory):
    """Reve DataLoader factory that inherits from AbstractDataLoaderFactory."""

    def __init__(self, pos_bank_dict, batch_size: int = 32, num_workers: int = 4, seed: int = 42, channel_restricted: bool = False):
        super().__init__(batch_size, num_workers, seed)
        self.pos_bank_dict = pos_bank_dict
        self.channel_restricted = channel_restricted

    def create_adapter(
            self,
            dataset: HFDataset,
            dataset_names: List[str],
            dataset_configs: List[str],
    ) -> ReveDatasetAdapter:
        return ReveDatasetAdapter(
            dataset, 
            dataset_names, 
            dataset_configs, 
            pos_bank_dict=self.pos_bank_dict, 
            channel_restricted=self.channel_restricted
        )

