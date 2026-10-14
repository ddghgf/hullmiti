from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from pem.core import modulate_embeddings, patch_credibility


@dataclass(frozen=True)
class PEMConfig:
    # Opt-in only: ordinary LLaVA/Baseline/HulluEdit inference keeps PEM off.
    enabled: bool = False
    strength: float = 1.0
    risk_bias: float = 0.25
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature: float = 2.0

    def to_dict(self):
        result = asdict(self)
        result["weights"] = list(self.weights)
        return result


class PEMController:
    def __init__(self, model, config: PEMConfig):
        self.model = model
        self.config = config
        tower = model.get_vision_tower()
        if getattr(tower, "select_feature", None) != "patch":
            raise RuntimeError("PEM requires patch-only visual features")
        image_size, patch_size = tower.config.image_size, tower.config.patch_size
        if not isinstance(image_size, int) or not isinstance(patch_size, int):
            raise RuntimeError("this PEM implementation requires an explicit rectangular patch grid")
        if image_size % patch_size:
            raise RuntimeError("vision input is not divisible by patch size")
        self.grid = (image_size // patch_size, image_size // patch_size)
        if tower.num_patches != self.grid[0] * self.grid[1]:
            raise RuntimeError("certified grid does not match vision tower patch count")
        self.records: list[dict] = []
        if hasattr(model, "_pem_controller"):
            raise RuntimeError("a PEM controller is already attached")
        model._pem_controller = self

    @torch.no_grad()
    def apply(self, raw_features, projected_features):
        if not self.config.enabled or self.config.strength == 0:
            return projected_features
        credibility, score_stats = patch_credibility(
            raw_features, *self.grid, weights=self.config.weights,
            temperature=self.config.temperature,
        )
        output, modulation_stats = modulate_embeddings(
            projected_features, credibility, strength=self.config.strength,
            risk_bias=self.config.risk_bias,
        )
        self.records.append({**score_stats, **modulation_stats, "batch": int(raw_features.shape[0]),
                             "patches": int(raw_features.shape[1]), "grid": list(self.grid),
                             "finite": bool(torch.isfinite(output).all())})
        return output

    def begin_sample(self):
        self.records = []

    def summary(self):
        if not self.records:
            return {"calls": 0, "grid": list(self.grid)}
        keys = [key for key, value in self.records[0].items() if isinstance(value, float)]
        return {"calls": len(self.records), "grid": list(self.grid),
                "means": {key: sum(row[key] for row in self.records) / len(self.records) for key in keys},
                "all_finite": all(row["finite"] for row in self.records)}

    def close(self):
        if getattr(self.model, "_pem_controller", None) is self:
            del self.model._pem_controller
