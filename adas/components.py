"""
"""
from typing import NamedTuple, List
from dataclasses import dataclass
from enum import Enum


class LayerType(Enum):
    CONV = 1
    DEPTH_CONV = 2
    FC = 3
    NON_CONV = 4


@dataclass
class LayerMetrics:
    rank: float
    KG: float
    condition: float


@dataclass
class ConvLayerMetrics:
    input_channel: LayerMetrics
    output_channel: LayerMetrics


class LRMetrics(NamedTuple):
    rank_velocity: List[float]
    r_conv: List[float]
