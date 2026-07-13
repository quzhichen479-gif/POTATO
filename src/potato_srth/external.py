from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .routing import SRTHMultiScaleV1, SRTHOutput


class FeatureContractError(RuntimeError):
    pass


class SRTHExternalFeatureBridge(nn.Module):
    """Dependency-free bridge between local YOLO26 features and POTATO SRTH.

    This class deliberately imports no Ultralytics/YOLO code. Codex should adapt
    the local PythonProject2 model so that it passes P3/P4/P5 dictionaries here.
    """

    def __init__(self, router: SRTHMultiScaleV1, *, required_levels: tuple[str, ...]) -> None:
        super().__init__()
        self.router = router
        self.required_levels = required_levels

    def _validate(self, name: str, features: Mapping[str, torch.Tensor]) -> None:
        missing = [level for level in self.required_levels if level not in features]
        if missing:
            raise FeatureContractError(f"{name} is missing levels: {missing}")
        for level in self.required_levels:
            value = features[level]
            if value.ndim != 4:
                raise FeatureContractError(f"{name}[{level}] must be BCHW")

    def forward(
        self,
        rgb_features: Mapping[str, torch.Tensor],
        dif_features: Mapping[str, torch.Tensor],
        pol_features: Mapping[str, torch.Tensor],
        heuristics: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> SRTHOutput:
        self._validate("rgb_features", rgb_features)
        self._validate("dif_features", dif_features)
        self._validate("pol_features", pol_features)
        return self.router(rgb_features, dif_features, pol_features, heuristics)
