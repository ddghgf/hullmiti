from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def neighbor_similarity(features: Tensor, rows: int, cols: int) -> Tensor:
    """Mean cosine similarity to true four-connected spatial neighbors."""
    if features.ndim != 3 or features.shape[1] != rows * cols:
        raise ValueError("patch/grid layout mismatch")
    batch, _, dim = features.shape
    grid = F.normalize(features.float(), dim=-1).reshape(batch, rows, cols, dim)
    total = torch.zeros((batch, rows, cols), device=features.device)
    count = torch.zeros_like(total)
    vertical = (grid[:, 1:] * grid[:, :-1]).sum(-1)
    horizontal = (grid[:, :, 1:] * grid[:, :, :-1]).sum(-1)
    total[:, 1:] += vertical
    total[:, :-1] += vertical
    count[:, 1:] += 1
    count[:, :-1] += 1
    total[:, :, 1:] += horizontal
    total[:, :, :-1] += horizontal
    count[:, :, 1:] += 1
    count[:, :, :-1] += 1
    return (total / count.clamp_min(1)).reshape(batch, rows * cols)


def robust_unit_scale(values: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-image 5-95% robust scaling, avoiding dataset fitting."""
    low = torch.quantile(values.float(), 0.05, dim=1, keepdim=True)
    high = torch.quantile(values.float(), 0.95, dim=1, keepdim=True)
    return ((values.float() - low) / (high - low).clamp_min(eps)).clamp(0, 1)


@torch.no_grad()
def patch_credibility(
    visual_features: Tensor,
    rows: int,
    cols: int,
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    temperature: float = 2.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict]:
    """Compute fixed, training-free patch credibility from CLIP features."""
    if visual_features.ndim != 3 or visual_features.shape[1] != rows * cols:
        raise ValueError("visual features do not match the certified patch grid")
    work = visual_features.float()
    # Low feature dispersion is weak evidence. Scale only this unbounded statistic.
    s1 = robust_unit_scale(work.var(dim=-1, unbiased=False), eps)
    global_feature = work.mean(dim=1, keepdim=True)
    s2 = ((F.cosine_similarity(work, global_feature, dim=-1) + 1.0) / 2.0).clamp(0, 1)
    s3 = ((neighbor_similarity(work, rows, cols) + 1.0) / 2.0).clamp(0, 1)
    stacked = torch.stack((s1, s2, s3), dim=-1)
    weight = torch.tensor(weights, device=work.device, dtype=work.dtype)
    centered = ((stacked - 0.5) * weight).sum(dim=-1) / weight.abs().sum().clamp_min(eps)
    credibility = torch.sigmoid(temperature * centered)
    if not torch.isfinite(credibility).all():
        raise FloatingPointError("non-finite patch credibility")
    stats = {
        "s1_mean": float(s1.mean()), "s1_std": float(s1.std(unbiased=False)),
        "s2_mean": float(s2.mean()), "s2_std": float(s2.std(unbiased=False)),
        "s3_mean": float(s3.mean()), "s3_std": float(s3.std(unbiased=False)),
        "credibility_mean": float(credibility.mean()),
        "credibility_std": float(credibility.std(unbiased=False)),
        "credibility_min": float(credibility.min()), "credibility_max": float(credibility.max()),
    }
    return credibility, stats


@torch.no_grad()
def modulate_embeddings(
    projected: Tensor,
    credibility: Tensor,
    *,
    strength: float = 1.0,
    risk_bias: float = 0.25,
) -> tuple[Tensor, dict]:
    """Smooth unreliable local residuals toward the image-global embedding."""
    if projected.ndim != 3 or credibility.shape != projected.shape[:2]:
        raise ValueError("projected embedding/credibility layout mismatch")
    if not 0 <= strength <= 1 or not 0 <= risk_bias <= 1:
        raise ValueError("strength and risk_bias must be in [0,1]")
    original = projected
    work = projected.float()
    global_embedding = work.mean(dim=1, keepdim=True)
    c = credibility.float().unsqueeze(-1)
    mixed = c * work + (1.0 - c) * global_embedding
    # A deterministic risk-map bias: high-risk positions receive an additional
    # bounded step toward the same image-global anchor, never an arbitrary vector.
    mixed = mixed + risk_bias * (1.0 - c) * (global_embedding - mixed)
    output = work + strength * (mixed - work)
    delta = output - work
    stats = {
        "embedding_norm_mean": float(torch.linalg.vector_norm(work, dim=-1).mean()),
        "delta_norm_mean": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
        "delta_to_embedding_ratio": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(work).clamp_min(1e-6)
        ),
    }
    return output.to(original.dtype), stats

