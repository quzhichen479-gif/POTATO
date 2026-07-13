from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SRTHLevelOutput:
    fused: torch.Tensor
    route_logits: torch.Tensor
    gates: torch.Tensor
    prior_logits: torch.Tensor
    dif_delta: torch.Tensor
    pol_delta: torch.Tensor


@dataclass
class SRTHOutput:
    fused_features: dict[str, torch.Tensor]
    route_logits: dict[str, torch.Tensor]
    gates: dict[str, torch.Tensor]
    prior_logits: dict[str, torch.Tensor]
    level_outputs: dict[str, SRTHLevelOutput]


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class LightweightModalityProjection(nn.Module):
    """Project an auxiliary modality to the RGB feature width with low overhead."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(16, min(out_channels, in_channels * 2))
        self.net = nn.Sequential(
            ConvNormAct(in_channels, hidden, kernel_size=1),
            ConvNormAct(hidden, hidden, kernel_size=3, groups=hidden),
            ConvNormAct(hidden, out_channels, kernel_size=1, activation=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SRTHLevelRouterV1(nn.Module):
    """RGB-preserving, heuristic-prior tri-modal router for one feature level.

    The router predicts two independent spatial gates for DIF and POL. They are
    intentionally not normalized by a softmax because both auxiliary modalities
    may be useful for the same target.
    """

    def __init__(
        self,
        *,
        rgb_channels: int,
        dif_channels: int,
        pol_channels: int,
        heuristic_channels: int,
        hidden_channels: int = 64,
        temperature: float = 1.0,
        heuristic_logit_scale: float = 1.0,
        residual_scale_init: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(rgb_channels, dif_channels, pol_channels) <= 0:
            raise ValueError("feature channels must be positive")
        if heuristic_channels < 0:
            raise ValueError("heuristic_channels must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.rgb_channels = int(rgb_channels)
        self.heuristic_channels = int(heuristic_channels)
        self.temperature = float(temperature)

        self.dif_projection = LightweightModalityProjection(dif_channels, rgb_channels)
        self.pol_projection = LightweightModalityProjection(pol_channels, rgb_channels)

        self.rgb_summary = ConvNormAct(rgb_channels, hidden_channels, kernel_size=1)
        self.dif_summary = ConvNormAct(rgb_channels, hidden_channels, kernel_size=1)
        self.pol_summary = ConvNormAct(rgb_channels, hidden_channels, kernel_size=1)

        learned_in = hidden_channels * 5 + heuristic_channels
        self.route_head = nn.Sequential(
            ConvNormAct(learned_in, hidden_channels, kernel_size=1),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.prior_head = (
            nn.Sequential(
                ConvNormAct(heuristic_channels, hidden_channels, kernel_size=1),
                nn.Conv2d(hidden_channels, 2, kernel_size=1),
            )
            if heuristic_channels > 0
            else None
        )

        self.dif_delta = nn.Sequential(
            ConvNormAct(rgb_channels, rgb_channels, kernel_size=3, groups=rgb_channels),
            ConvNormAct(rgb_channels, rgb_channels, kernel_size=1, activation=False),
        )
        self.pol_delta = nn.Sequential(
            ConvNormAct(rgb_channels, rgb_channels, kernel_size=3, groups=rgb_channels),
            ConvNormAct(rgb_channels, rgb_channels, kernel_size=1, activation=False),
        )

        self.heuristic_logit_scale = nn.Parameter(torch.tensor(float(heuristic_logit_scale)))
        self.dif_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.pol_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    @staticmethod
    def _check_spatial(reference: torch.Tensor, value: torch.Tensor, name: str) -> None:
        if reference.shape[0] != value.shape[0]:
            raise ValueError(f"{name} batch size does not match RGB")
        if reference.shape[-2:] != value.shape[-2:]:
            raise ValueError(f"{name} spatial size must match RGB before routing")

    def forward(
        self,
        rgb: torch.Tensor,
        dif: torch.Tensor,
        pol: torch.Tensor,
        heuristic: torch.Tensor | None,
    ) -> SRTHLevelOutput:
        if rgb.ndim != 4 or dif.ndim != 4 or pol.ndim != 4:
            raise ValueError("RGB, DIF and POL features must be BCHW tensors")
        self._check_spatial(rgb, dif, "DIF")
        self._check_spatial(rgb, pol, "POL")

        dif_projected = self.dif_projection(dif)
        pol_projected = self.pol_projection(pol)

        rgb_summary = self.rgb_summary(rgb)
        dif_summary = self.dif_summary(dif_projected)
        pol_summary = self.pol_summary(pol_projected)

        route_parts = [
            rgb_summary,
            dif_summary,
            pol_summary,
            torch.abs(rgb_summary - dif_summary),
            torch.abs(rgb_summary - pol_summary),
        ]

        if self.heuristic_channels > 0:
            if heuristic is None:
                raise ValueError("heuristic tensor is required by this router")
            if heuristic.ndim != 4 or heuristic.shape[1] != self.heuristic_channels:
                raise ValueError(
                    f"heuristic must be BCHW with {self.heuristic_channels} channels"
                )
            if heuristic.shape[0] != rgb.shape[0]:
                raise ValueError("heuristic batch size does not match RGB")
            heuristic_resized = F.interpolate(
                heuristic,
                size=rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            route_parts.append(heuristic_resized)
            assert self.prior_head is not None
            prior_logits = self.prior_head(heuristic_resized)
        else:
            prior_logits = rgb.new_zeros((rgb.shape[0], 2, *rgb.shape[-2:]))

        learned_logits = self.route_head(torch.cat(route_parts, dim=1))
        route_logits = learned_logits + self.heuristic_logit_scale * prior_logits
        gates = torch.sigmoid(route_logits / self.temperature)

        dif_delta = self.dif_delta(dif_projected)
        pol_delta = self.pol_delta(pol_projected)
        fused = (
            rgb
            + self.dif_residual_scale * gates[:, 0:1] * dif_delta
            + self.pol_residual_scale * gates[:, 1:2] * pol_delta
        )
        return SRTHLevelOutput(
            fused=fused,
            route_logits=route_logits,
            gates=gates,
            prior_logits=prior_logits,
            dif_delta=dif_delta,
            pol_delta=pol_delta,
        )


class SRTHMultiScaleV1(nn.Module):
    """Apply SRTH routing at selected feature levels and preserve all other RGB levels."""

    def __init__(
        self,
        *,
        channels: Mapping[str, tuple[int, int, int]],
        heuristic_channels: int,
        hidden_channels: int = 64,
        temperature: float = 1.0,
        heuristic_logit_scale: float = 1.0,
        residual_scale_init: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("at least one routed feature level is required")
        self.level_names = tuple(channels)
        self.routers = nn.ModuleDict(
            {
                level: SRTHLevelRouterV1(
                    rgb_channels=widths[0],
                    dif_channels=widths[1],
                    pol_channels=widths[2],
                    heuristic_channels=heuristic_channels,
                    hidden_channels=hidden_channels,
                    temperature=temperature,
                    heuristic_logit_scale=heuristic_logit_scale,
                    residual_scale_init=residual_scale_init,
                    dropout=dropout,
                )
                for level, widths in channels.items()
            }
        )

    def forward(
        self,
        rgb_features: Mapping[str, torch.Tensor],
        dif_features: Mapping[str, torch.Tensor],
        pol_features: Mapping[str, torch.Tensor],
        heuristics: torch.Tensor | Mapping[str, torch.Tensor] | None,
    ) -> SRTHOutput:
        fused_features = dict(rgb_features)
        route_logits: dict[str, torch.Tensor] = {}
        gates: dict[str, torch.Tensor] = {}
        prior_logits: dict[str, torch.Tensor] = {}
        level_outputs: dict[str, SRTHLevelOutput] = {}

        missing = [
            level
            for level in self.level_names
            if level not in rgb_features or level not in dif_features or level not in pol_features
        ]
        if missing:
            raise KeyError(f"missing routed feature levels: {missing}")

        for level in self.level_names:
            if isinstance(heuristics, Mapping):
                heuristic = heuristics.get(level)
            else:
                heuristic = heuristics
            output = self.routers[level](
                rgb_features[level],
                dif_features[level],
                pol_features[level],
                heuristic,
            )
            fused_features[level] = output.fused
            route_logits[level] = output.route_logits
            gates[level] = output.gates
            prior_logits[level] = output.prior_logits
            level_outputs[level] = output

        return SRTHOutput(
            fused_features=fused_features,
            route_logits=route_logits,
            gates=gates,
            prior_logits=prior_logits,
            level_outputs=level_outputs,
        )
