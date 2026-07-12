"""POTATO RGB-polarization candidate arbitration research package."""

from .e11_model import CandidateResidualTransformer
from .e12_model import RollbackTransformer
from .model import CandidateTransformerArbitrator

__all__ = [
    "CandidateTransformerArbitrator",
    "CandidateResidualTransformer",
    "RollbackTransformer",
]
__version__ = "0.3.0"
