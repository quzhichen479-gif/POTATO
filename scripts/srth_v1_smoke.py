from __future__ import annotations

import argparse

import torch

from potato_srth import SRTHMultiScaleV1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    model = SRTHMultiScaleV1(
        channels={"p3": (128, 64, 64), "p4": (256, 128, 128)},
        heuristic_channels=6,
        hidden_channels=64,
        residual_scale_init=1e-3,
    ).to(device)
    rgb = {
        "p3": torch.randn(2, 128, 80, 80, device=device),
        "p4": torch.randn(2, 256, 40, 40, device=device),
        "p5": torch.randn(2, 512, 20, 20, device=device),
    }
    dif = {
        "p3": torch.randn(2, 64, 80, 80, device=device),
        "p4": torch.randn(2, 128, 40, 40, device=device),
    }
    pol = {
        "p3": torch.randn(2, 64, 80, 80, device=device),
        "p4": torch.randn(2, 128, 40, 40, device=device),
    }
    heuristic = torch.randn(2, 6, 640, 640, device=device)
    output = model(rgb, dif, pol, heuristic)
    loss = sum(feature.square().mean() for feature in output.fused_features.values())
    loss.backward()
    print(
        {
            "device": str(device),
            "p3": tuple(output.fused_features["p3"].shape),
            "p4": tuple(output.fused_features["p4"].shape),
            "p5_passthrough": output.fused_features["p5"] is rgb["p5"],
            "loss": float(loss.detach().cpu()),
        }
    )


if __name__ == "__main__":
    main()
