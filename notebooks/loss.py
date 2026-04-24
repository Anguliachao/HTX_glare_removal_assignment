"""Weighted loss helpers for notebook training.

This module intentionally keeps loss wrappers simple and extensible so more loss
types (for example perceptual/composite losses) can be added later.
"""

from __future__ import annotations



import torch
from torch import nn
from torch.nn import functional as F

# Reduction = Literal["mean", "sum", "none"]
VALID_REDUCTIONS = {"mean", "sum", "none"}


def _validate_weight_and_reduction(weight: float, reduction: str) -> None:
    """Validate common constructor arguments for weighted loss wrappers."""
    if weight < 0:
        raise ValueError(f"weight must be >= 0, got {weight!r}.")
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {sorted(VALID_REDUCTIONS)}, got {reduction!r}."
        )


class WeightedL1Loss(nn.Module):
    """Weighted L1 (MAE) loss wrapper."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        _validate_weight_and_reduction(weight=weight, reduction=reduction)
        self.weight = float(weight)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return weight * L1 loss between pred and target."""
        base = F.l1_loss(pred, target, reduction=self.reduction)
        return self.weight * base


class WeightedL2Loss(nn.Module):
    """Weighted L2 (MSE) loss wrapper."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        _validate_weight_and_reduction(weight=weight, reduction=reduction)
        self.weight = float(weight)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return weight * L2 loss between pred and target."""
        base = F.mse_loss(pred, target, reduction=self.reduction)
        return self.weight * base


def build_loss(name: str, weight: float = 1.0, reduction: str = "mean") -> nn.Module:
    """Build a weighted loss module by name.

    Supported names:
    - "l1"
    - "l2"
    """
    key = name.strip().lower()
    if key == "l1":
        return WeightedL1Loss(weight=weight, reduction=reduction)
    if key == "l2":
        return WeightedL2Loss(weight=weight, reduction=reduction)
    raise ValueError(f"Unsupported loss name '{name}'. Supported names are: ['l1', 'l2'].")


if __name__ == "__main__":
    pred = torch.tensor([[[[0.0, 0.2], [0.7, 1.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.1, 0.1], [0.9, 0.8]]]], dtype=torch.float32)

    l1 = WeightedL1Loss(weight=1.0, reduction="mean")
    l2 = WeightedL2Loss(weight=1.0, reduction="mean")
    print(f"L1 (mean): {l1(pred, target).item():.6f}")
    print(f"L2 (mean): {l2(pred, target).item():.6f}")

    l1_none = WeightedL1Loss(weight=2.0, reduction="none")
    l1_none_value = l1_none(pred, target)
    print(f"L1 (none) shape: {tuple(l1_none_value.shape)}")
    print(f"L1 (none) weighted sample value: {l1_none_value[0, 0, 0, 0].item():.6f}")

    zero_weight_l2 = WeightedL2Loss(weight=0.0, reduction="mean")
    print(f"L2 with zero weight: {zero_weight_l2(pred, target).item():.6f}")
