import copy
from collections import defaultdict
from typing import List, Dict

from torch import nn


def generate_area_config(brain_regions: List[int]) -> Dict[str, Dict]:
    """Generate area config from brain regions list."""
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in sorted(region_to_channels.items()):
        if channels:  # Only add non-empty regions
            area_config[f'region_{region}'] = {
                'channels': len(channels),
                'slice': slice(channels[0], channels[-1] + 1)
            }
    return area_config

def _get_clones(module, n):
    """Create N deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
