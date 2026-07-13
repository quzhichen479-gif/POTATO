from __future__ import annotations

import torch

from potato_srth import (
    SRTHExternalFeatureBridge,
    SRTHMultiScaleV1,
    compute_srth_v1_loss,
    detector_quality,
    oracle_route_targets,
    rasterize_route_targets,
)


def _model(residual_scale_init: float = 0.0) -> SRTHMultiScaleV1:
    return SRTHMultiScaleV1(
        channels={"p3": (16, 8, 8), "p4": (32, 16, 16)},
        heuristic_channels=5,
        hidden_channels=16,
        residual_scale_init=residual_scale_init,
    )


def _features() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rgb = {
        "p3": torch.randn(2, 16, 16, 16),
        "p4": torch.randn(2, 32, 8, 8),
        "p5": torch.randn(2, 64, 4, 4),
    }
    dif = {"p3": torch.randn(2, 8, 16, 16), "p4": torch.randn(2, 16, 8, 8)}
    pol = {"p3": torch.randn(2, 8, 16, 16), "p4": torch.randn(2, 16, 8, 8)}
    return rgb, dif, pol


def test_zero_initialized_residual_preserves_rgb_and_p5() -> None:
    model = _model(residual_scale_init=0.0).eval()
    rgb, dif, pol = _features()
    heuristic = torch.randn(2, 5, 64, 64)
    with torch.no_grad():
        output = model(rgb, dif, pol, heuristic)
    assert torch.equal(output.fused_features["p3"], rgb["p3"])
    assert torch.equal(output.fused_features["p4"], rgb["p4"])
    assert output.fused_features["p5"] is rgb["p5"]
    assert output.gates["p3"].shape == (2, 2, 16, 16)
    assert torch.all((output.gates["p3"] >= 0) & (output.gates["p3"] <= 1))


def test_router_backward_reaches_all_branches() -> None:
    model = _model(residual_scale_init=1e-3).train()
    rgb, dif, pol = _features()
    heuristic = torch.randn(2, 5, 64, 64)
    output = model(rgb, dif, pol, heuristic)
    loss = output.fused_features["p3"].square().mean() + output.gates["p4"].mean()
    loss.backward()
    assert model.routers["p3"].dif_residual_scale.grad is not None
    assert model.routers["p3"].dif_projection.net[0][0].weight.grad is not None
    assert model.routers["p4"].route_head[-1].weight.grad is not None


def test_oracle_soft_targets_follow_relative_quality() -> None:
    iou = torch.tensor([0.8, 0.2, 0.7])
    confidence = torch.tensor([0.9, 0.9, 0.4])
    rgb = detector_quality(iou, confidence)
    dif = detector_quality(torch.tensor([0.6, 0.9, 0.1]), torch.tensor([0.8, 0.8, 0.9]))
    pol = detector_quality(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([0.9, 0.8, 0.7]))
    targets = oracle_route_targets(rgb, dif, pol, temperature=0.1)
    assert targets.dif[1] > 0.5
    assert targets.pol[0] > 0.5
    assert targets.best_expert.shape == (3,)


def test_rasterization_and_auxiliary_loss_are_finite() -> None:
    model = _model(residual_scale_init=1e-3)
    rgb, dif, pol = _features()
    output = model(rgb, dif, pol, torch.randn(2, 5, 32, 32))
    boxes = torch.tensor([[0.10, 0.10, 0.25, 0.25], [0.50, 0.50, 0.80, 0.80]])
    batches = torch.tensor([0, 1])
    dif_target = torch.tensor([0.9, 0.2])
    pol_target = torch.tensor([0.3, 0.8])
    targets = {}
    masks = {}
    for level, size in {"p3": (16, 16), "p4": (8, 8)}.items():
        targets[level], masks[level] = rasterize_route_targets(
            boxes,
            batches,
            dif_target,
            pol_target,
            batch_size=2,
            feature_size=size,
        )
    losses = compute_srth_v1_loss(output, targets, masks)
    assert torch.isfinite(losses.total)
    assert losses.route.item() > 0


def test_external_bridge_checks_and_returns_feature_contract() -> None:
    bridge = SRTHExternalFeatureBridge(_model(), required_levels=("p3", "p4"))
    rgb, dif, pol = _features()
    output = bridge(rgb, dif, pol, torch.randn(2, 5, 64, 64))
    assert set(output.route_logits) == {"p3", "p4"}
