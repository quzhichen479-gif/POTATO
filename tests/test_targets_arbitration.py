import torch

from potato_e1.arbitration import ArbitrationThresholds, arbitrate_single
from potato_e1.model import ArbitratorOutput
from potato_e1.targets import build_targets


def test_four_quadrant_targets() -> None:
    gt_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.20, 0.20],
            [0.30, 0.30, 0.40, 0.40],
            [0.50, 0.50, 0.60, 0.60],
        ]
    )
    gt_classes = torch.zeros(3, dtype=torch.long)
    rgb_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.20, 0.20],  # common
            [0.30, 0.30, 0.40, 0.40],  # RGB-only
        ]
    )
    pol_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.20, 0.20],  # common
            [0.50, 0.50, 0.60, 0.60],  # POL-only
        ]
    )
    classes = torch.zeros(2, dtype=torch.long)
    targets = build_targets(
        rgb_boxes,
        classes,
        pol_boxes,
        classes,
        gt_boxes,
        gt_classes,
    )
    assert targets.valid.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert targets.protect.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert targets.rescue.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert targets.pair_same[0, 2] == 1
    assert targets.pair_same[1, 3] == 0


def test_protected_rgb_is_not_replaced() -> None:
    batch = {
        "boxes": torch.tensor(
            [[[0.10, 0.10, 0.20, 0.20], [0.11, 0.11, 0.21, 0.21]]]
        ),
        "scores": torch.tensor([[0.80, 0.95]]),
        "classes": torch.zeros((1, 2), dtype=torch.long),
        "modality": torch.tensor([[0, 1]]),
        "mask": torch.tensor([[True, True]]),
    }
    output = ArbitratorOutput(
        valid_logit=torch.tensor([[4.0, 4.0]]),
        iou_pred=torch.tensor([[0.60, 0.95]]),
        rescue_logit=torch.tensor([[-4.0, 4.0]]),
        protect_logit=torch.tensor([[4.0, -4.0]]),
        pair_same_logit=torch.tensor([[[-20.0, 4.0], [4.0, -20.0]]]),
        encoded=torch.zeros((1, 2, 8)),
    )
    result = arbitrate_single(output, batch, 0, ArbitrationThresholds())
    assert len(result.boxes) == 1
    assert int(result.provenance[0]) == 2  # merged around protected RGB, not replaced by POL


def test_unmatched_pol_rescue_is_admitted() -> None:
    batch = {
        "boxes": torch.tensor(
            [[[0.10, 0.10, 0.20, 0.20], [0.60, 0.60, 0.70, 0.70]]]
        ),
        "scores": torch.tensor([[0.80, 0.70]]),
        "classes": torch.zeros((1, 2), dtype=torch.long),
        "modality": torch.tensor([[0, 1]]),
        "mask": torch.tensor([[True, True]]),
    }
    output = ArbitratorOutput(
        valid_logit=torch.tensor([[4.0, 4.0]]),
        iou_pred=torch.tensor([[0.80, 0.80]]),
        rescue_logit=torch.tensor([[-4.0, 4.0]]),
        protect_logit=torch.tensor([[4.0, -4.0]]),
        pair_same_logit=torch.full((1, 2, 2), -4.0),
        encoded=torch.zeros((1, 2, 8)),
    )
    result = arbitrate_single(output, batch, 0, ArbitrationThresholds())
    assert len(result.boxes) == 2
    assert set(result.provenance.tolist()) == {0, 1}
