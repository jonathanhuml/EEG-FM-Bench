"""
CSBrain Adapter that inherits from AbstractDatasetAdapter.

CSBrain requires specific data preprocessing:
1. Data shape: (batch, n_channels, n_patches, patch_size)
2. Brain region assignment for each channel
3. Topological sorting within brain regions
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory
from common.utils import ElectrodeSet

logger = logging.getLogger("baseline")


# Brain region definitions based on 10-10 system
# Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
# Extended to support all 90 electrodes from ElectrodeSet
BRAIN_REGION_MAPPING = {
    # Frontal region (0) - Prefrontal + Frontal areas
    'FP1': 0, 'FP2': 0, 'FPZ': 0,
    'AF1': 0, 'AF2': 0, 'AF3': 0, 'AF4': 0, 'AF5': 0, 'AF6': 0, 
    'AF7': 0, 'AF8': 0, 'AF9': 0, 'AF10': 0, 'AFZ': 0,
    'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 
    'F7': 0, 'F8': 0, 'F9': 0, 'F10': 0, 'FZ': 0,
    'FC1': 0, 'FC2': 0, 'FC3': 0, 'FC4': 0, 'FC5': 0, 'FC6': 0, 'FCZ': 0,
    
    # Parietal region (1)
    'P1': 1, 'P2': 1, 'P3': 1, 'P4': 1, 'P5': 1, 'P6': 1, 
    'P7': 1, 'P8': 1, 'P9': 1, 'P10': 1, 'PZ': 1,
    'PO1': 1, 'PO2': 1, 'PO3': 1, 'PO4': 1, 'PO5': 1, 'PO6': 1, 
    'PO7': 1, 'PO8': 1, 'PO9': 1, 'PO10': 1, 'POZ': 1,
    
    # Temporal region (2)
    'T1': 2, 'T2': 2,
    'T7': 2, 'T8': 2, 'T9': 2, 'T10': 2,
    'FT7': 2, 'FT8': 2, 'FT9': 2, 'FT10': 2,
    'TP7': 2, 'TP8': 2, 'TP9': 2, 'TP10': 2,
    'A1': 2, 'A2': 2,  # Ear electrodes - assign to temporal
    
    # Occipital region (3)
    'O1': 3, 'O2': 3, 'OZ': 3,
    'I1': 3, 'I2': 3, 'IZ': 3,  # Inion electrodes
    
    # Central region (4)
    'C1': 4, 'C2': 4, 'C3': 4, 'C4': 4, 'C5': 4, 'C6': 4, 'CZ': 4,
    'CP1': 4, 'CP2': 4, 'CP3': 4, 'CP4': 4, 'CP5': 4, 'CP6': 4, 'CPZ': 4,
}

# Topological order within each brain region (for sorted attention)
# Extended to include all electrodes in each region
TOPOLOGY = {
    0: [  # Frontal (anterior to posterior, left to right)
        "FP1", "FPZ", "FP2",
        "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
        "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
        "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    ],
    1: [  # Parietal
        "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
        "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    ],
    2: [  # Temporal (left then right)
        "FT9", "FT7", "T9", "T7", "TP9", "TP7", "A1", "T1",
        "T2", "A2", "TP8", "TP10", "T8", "T10", "FT8", "FT10",
    ],
    3: [  # Occipital
        "O1", "OZ", "O2",
        "I1", "IZ", "I2",
    ],
    4: [  # Central
        "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    ],
}


def get_brain_region(channel_name: str) -> int:
    """Get brain region for a channel. Returns -1 if unknown."""
    return BRAIN_REGION_MAPPING.get(channel_name.upper(), -1)


def compute_sorted_indices_and_regions(channel_names: List[str]) -> Tuple[List[int], List[int]]:
    """
    Compute sorted indices and brain region assignments for given channels.
    
    Returns:
        sorted_indices: Indices to reorder channels by brain region
        brain_regions: Brain region ID for each channel
    """
    # Assign brain regions
    brain_regions = []
    for ch in channel_names:
        region = get_brain_region(ch)
        if region == -1:
            # Default to central region for unknown channels
            region = 4
        brain_regions.append(region)
    
    # Group channels by brain region
    region_groups = defaultdict(list)
    for i, (ch, region) in enumerate(zip(channel_names, brain_regions)):
        region_groups[region].append((i, ch))
    
    # Sort within each region by topological order
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        topo_order = TOPOLOGY.get(region, [])
        
        # Sort by topological order
        def get_topo_index(item):
            idx, ch = item
            ch_upper = ch.upper()
            if ch_upper in topo_order:
                return topo_order.index(ch_upper)
            return len(topo_order)  # Unknown channels go to end
        
        sorted_electrodes = sorted(region_electrodes, key=get_topo_index)
        sorted_indices.extend([e[0] for e in sorted_electrodes])
    
    # Reorder brain_regions to match sorted indices
    sorted_regions = [brain_regions[i] for i in sorted_indices]
    
    return sorted_indices, sorted_regions


class CSBrainDatasetAdapter(AbstractDatasetAdapter):
    """CSBrain dataset adapter that handles EEG data with brain region awareness."""
    
    def __init__(
        self, 
        dataset: HFDataset, 
        dataset_names: List[str], 
        dataset_configs: List[str],
        patch_size: int = 200,
    ):
        super().__init__(dataset, dataset_names, dataset_configs)
        self.model_name = 'csbrain'
        self.electrode_set: ElectrodeSet = ElectrodeSet()
        self.patch_size = patch_size

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by CSBrain."""
        return list(BRAIN_REGION_MAPPING.keys())
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample with brain region information."""
        # Get base processed sample
        result = super()._process_sample(sample)
        result.pop('chans_id')

        chs = result['chs']
        name_list = self.electrode_set.get_electrodes_name(result['chs'].tolist())
        
        # Compute brain region sorting
        sorted_indices, brain_regions = compute_sorted_indices_and_regions(name_list)

        # Apply sorting to data
        sorted_indices_tensor = torch.tensor(sorted_indices, dtype=torch.long)
        data_sorted = result['data'][sorted_indices_tensor, :]  # (n_channels, n_timepoints)
        
        # Reshape to patches: (n_channels, n_patches, patch_size)
        n_channels, n_timepoints = data_sorted.shape
        # Calculate number of patches based on actual sample length
        n_patches = n_timepoints // self.patch_size
        
        # Truncate to fit patch structure
        data_patched = data_sorted[:, :n_patches * self.patch_size]
        # data_patched = data_patched.view(n_channels, n_patches, self.patch_size)
        
        # Build result dictionary
        result = {
            'data': data_patched,  # Shape: (n_channels, n_patches * patch_size)
            'montage': result['montage'],
            'chs': chs,
            'task': result['task'],
            'label': result['label'],
            # CSBrain specific
            'brain_regions': torch.tensor(brain_regions, dtype=torch.long),
            'sorted_indices': torch.tensor(sorted_indices, dtype=torch.long),
        }
        
        return result


class CSBrainDataLoaderFactory(AbstractDataLoaderFactory):
    """CSBrain DataLoader factory that inherits from AbstractDataLoaderFactory."""
    
    def __init__(
        self, 
        batch_size: int = 32, 
        num_workers: int = 4, 
        seed: int = 42,
        patch_size: int = 200,
        max_seq_len: Optional[int] = None  # None means use actual sample length
    ):
        super().__init__(batch_size, num_workers, seed)
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
    
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> CSBrainDatasetAdapter:
        return CSBrainDatasetAdapter(
            dataset, 
            dataset_names, 
            dataset_configs,
            patch_size=self.patch_size,
        )
