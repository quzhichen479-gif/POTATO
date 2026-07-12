from __future__ import annotations

import torch

from potato_e1.e11_arbitration import (
    E11Thresholds,
    arbitrate_e11_single,
    fixed_baseline_single,
)
from potato_e1.e11_losses import bounded_residual_score
from potato_e1.e11_model import CandidateResidualTransformer, ResidualArbitratorOutput


def _batch() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(
            [[[0.10, 0.10, 0.30, 0.30], [0.11, 0.11, 0.31, 0.31], [0.70, 0.70, 0.80, 0.80]]]
        ),
        "scores": torch.tensor([[0.90, 0.80, 0.10]]),
        "classes": torch.zeros((1, 3), dtype=torch.long),
        "modality": torch.tensor([[0, 1, 1]], dtype=torch.long),
        "appearance": torch.zeros((1, 3, 0)),
        "mask": torch.ones((1, 3), dtype=torch.bool),
    }


def test_residual_model_shapes_and_backward() -> None:
    batch = _batch()
    model = CandidateResidualTransformer(
        appearance_dim=0,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    output = model(batch)
    assert output.score_delta.shape == (1, 3)
    assert output.iou_pred.shape == (1, 3)
    assert output.rescue_logit.shape == (1, 3)
    loss = output.score_delta.sum() + output.iou_pred.sum() + output.rescue_logit.sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bounded_residual_score_cannot_exceed_alpha_in_logit_space() -> None:
    raw = torch.tensor([0.5, 0.5])
    delta = torch.tensor([1e6, -1e6])
    adjusted = bounded_residual_score(raw, delta, alpha=0.25)
    expected = torch.sigmoid(torch.tensor([0.25, -0.25]))
    assert torch.allclose(adjusted, expected, atol=1e-6)


def test_e11_never_deletes_or_moves_fixed_baseline_candidates() -> None:
    batch = _batch()
    output = ResidualArbitratorOutput(
        score_delta=torch.zeros((1, 3)),
        iou_pred=torch.tensor([[0.8, 0.7, 0.9]]),
        rescue_logit=torch.tensor([[-20.0, -20.0, 20.0]]),
        encoded=torch.zeros((1, 3, 8)),
    )
    thresholds = E11Thresholds(
        base_conf=0.25,
        base_nms_iou=0.30,
        residual_alpha=0.50,
        rescue_threshold=0.80,
        extra_quality_threshold=0.50,
        extra_score_floor=0.05,
        extra_overlap_iou=0.30,
        max_extra_per_image=1,
    )
    baseline = fixed_baseline_single(batch, 0, thresholds)
    result = arbitrate_e11_single(output, batch, 0, thresholds)

    assert len(baseline.boxes) == 1
    assert result.base_count == 1
    assert result.extra_count == 1
    assert torch.allclose(result.boxes[: result.base_count], baseline.boxes)
    assert torch.equal(result.classes[: result.base_count], baseline.classes)
    assert result.provenance.tolist() == [0, 3]


def test_e11_blocks_overlapping_pol_extra() -> None:
    batch = _batch()
    batch["scores"][0, 1] = 0.10
    output = ResidualArbitratorOutput(
        score_delta=torch.zeros((1, 3)),
        iou_pred=torch.tensor([[0.8, 0.9, 0.1]]),
        rescue_logit=torch.tensor([[-20.0, 20.0, -20.0]]),
        encoded=torch.zeros((1, 3, 8)),
    )
    thresholds = E11Thresholds(
        base_conf=0.25,
        base_nms_iou=0.30,
        rescue_threshold=0.80,
        extra_quality_threshold=0.50,
        extra_score_floor=0.05,
        extra_overlap_iou=0.30,
        max_extra_per_image=1,
    )
    result = arbitrate_e11_single(output, batch, 0, thresholds)
    assert result.base_count == 1
    assert result.extra_count == 0
