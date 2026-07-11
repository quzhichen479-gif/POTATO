import torch

from potato_e1.losses import compute_loss
from potato_e1.model import CandidateTransformerArbitrator


def make_batch() -> dict[str, torch.Tensor]:
    batch_size, tokens = 2, 7
    boxes = torch.rand(batch_size, tokens, 4)
    boxes[..., 2:] = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return {
        "boxes": boxes,
        "scores": torch.rand(batch_size, tokens).clamp(0.01, 0.99),
        "classes": torch.zeros(batch_size, tokens, dtype=torch.long),
        "modality": torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]),
        "appearance": torch.randn(batch_size, tokens, 16),
        "physics": torch.randn(batch_size, tokens, 8),
        "mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        "valid_target": torch.randint(0, 2, (batch_size, tokens)).float(),
        "iou_target": torch.rand(batch_size, tokens),
        "rescue_target": torch.randint(0, 2, (batch_size, tokens)).float(),
        "protect_target": torch.randint(0, 2, (batch_size, tokens)).float(),
        "pair_same_target": torch.randint(0, 2, (batch_size, tokens, tokens)).float(),
        "pair_mask": torch.ones(batch_size, tokens, tokens, dtype=torch.bool),
    }


def test_model_shapes_and_backward() -> None:
    batch = make_batch()
    model = CandidateTransformerArbitrator(
        appearance_dim=16,
        physics_dim=8,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pair_dim=32,
    )
    output = model(batch)
    assert output.valid_logit.shape == (2, 7)
    assert output.iou_pred.shape == (2, 7)
    assert output.pair_same_logit.shape == (2, 7, 7)
    assert torch.isfinite(output.valid_logit).all()

    losses = compute_loss(
        output,
        batch,
        {
            "valid_weight": 1,
            "iou_weight": 1,
            "pair_weight": 1,
            "rescue_weight": 1,
            "protect_weight": 1,
        },
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
