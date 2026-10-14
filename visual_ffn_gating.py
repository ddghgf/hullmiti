from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

import torch

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass(frozen=True)
class VisualFFNGatingConfig:
    enabled: bool = True
    layers: tuple[int, ...] = (16, 20, 24, 28)
    ratio_threshold: float = 2.0
    strength: float = 0.3
    top_fraction: float = 0.2
    visual_quantile: float = 0.9
    eps: float = 1e-6

    def to_dict(self):
        value = asdict(self)
        value["layers"] = list(self.layers)
        return value


class VisualSupportSelectiveFFNGating:
    """Per-sample, hook-only visual-support selective FFN gating.

    Prefill caches only a per-neuron visual activation quantile and the mean
    visual MLP input.  The final prefill position and each decode position are
    gated in the down-projection input; parameters and KV cache are untouched.
    """

    stats_key = "visual_ffn_gating_stats"

    def __init__(self, model, config: VisualFFNGatingConfig):
        if not 0.0 <= config.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not 0.0 <= config.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in [0, 1]")
        if not 0.0 < config.visual_quantile < 1.0:
            raise ValueError("visual_quantile must be in (0, 1)")
        if config.ratio_threshold < 0:
            raise ValueError("ratio_threshold must be non-negative")

        layers = model.get_model().layers
        if any(index < 0 or index >= len(layers) for index in config.layers):
            raise ValueError(f"layers must be within [0, {len(layers) - 1}]")

        self.model = model
        self.config = config
        self.handles = []
        self.image_start = None
        self.original_length = None
        self.visual_end = None
        self.visual_support = {}
        self.visual_reference = {}
        self.records = []

        for index in config.layers:
            mlp = layers[index].mlp
            if getattr(mlp, "pretraining_tp", 1) != 1:
                raise ValueError("Hook implementation requires pretraining_tp=1")
            self.handles.append(mlp.register_forward_pre_hook(self._mlp_input_hook(index)))
            self.handles.append(mlp.down_proj.register_forward_pre_hook(self._down_input_hook(index)))

    def begin_sample(self, input_ids=None):
        self.visual_end = None
        self.visual_support = {}
        self.visual_reference = {}
        self.records = []
        if input_ids is None:
            self.image_start = self.original_length = None
            return
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Visual FFN gating currently supports one input sample")
        positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        if positions.numel() != 1:
            raise ValueError(f"Expected exactly one image token, got {positions.numel()}")
        self.image_start = int(positions[0])
        self.original_length = int(input_ids.shape[1])

    def _mlp_input_hook(self, layer_index):
        @torch.no_grad()
        def hook(module, inputs):
            if not self.config.enabled or self.image_start is None or not inputs:
                return None
            hidden = inputs[0]
            if hidden.ndim != 3 or hidden.shape[1] <= 1:
                return None
            visual_count = hidden.shape[1] - (self.original_length - 1)
            if visual_count <= 0:
                return None
            start = self.image_start
            end = start + visual_count
            if end > hidden.shape[1]:
                return None
            self.visual_end = end
            self.visual_reference[layer_index] = hidden[:, start:end, :].float().mean(dim=1)
            return None

        return hook

    def _down_input_hook(self, layer_index):
        @torch.no_grad()
        def hook(module, inputs):
            if not self.config.enabled or self.image_start is None or not inputs:
                return None
            activation = inputs[0]
            if activation.ndim != 3:
                return None
            is_prefill = activation.shape[1] > 1
            if is_prefill:
                if self.visual_end is None:
                    return None
                visual = activation[:, self.image_start:self.visual_end, :].float().abs()
                if visual.shape[1] == 0:
                    return None
                support = torch.quantile(visual, self.config.visual_quantile, dim=1)
                if not torch.isfinite(support).all():
                    return None
                self.visual_support[layer_index] = support
            support = self.visual_support.get(layer_index)
            reference = self.visual_reference.get(layer_index)
            if support is None or reference is None:
                return None

            batch = activation.shape[0]
            support = self._expand_batch(support, batch)
            reference = self._expand_batch(reference, batch)
            if support is None or reference is None:
                return None
            current = activation[:, -1, :].float()
            support_normalized = support / (support.mean(dim=-1, keepdim=True) + self.config.eps)
            text_abs = current.abs()
            text_normalized = text_abs / (text_abs.mean(dim=-1, keepdim=True) + self.config.eps)
            ratio = text_normalized / (support_normalized + self.config.eps)

            direction = torch.matmul(reference, module.weight.detach().float())
            projection = current * direction
            top_mask = self._top_fraction_mask(text_abs)
            ratio_mask = ratio > self.config.ratio_threshold
            negative_mask = projection < 0
            selected = top_mask & ratio_mask & negative_mask

            changed = bool(self.config.strength > 0 and selected.any())
            if changed:
                edited = activation.clone()
                scale = 1.0 - self.config.strength * selected.to(current.dtype)
                edited[:, -1, :] = (current * scale).to(activation.dtype)
            else:
                edited = activation

            selected_projection_mass = (projection.abs() * selected).sum(dim=-1)
            total_projection_mass = projection.abs().sum(dim=-1) + self.config.eps
            delta_norm = torch.linalg.vector_norm((edited[:, -1, :].float() - current), dim=-1)
            self.records.append({
                "layer": layer_index,
                "phase": "prefill" if is_prefill else "decode",
                "selected_ratio": float(selected.float().mean()),
                "selected_count": float(selected.float().sum(dim=-1).mean()),
                "ratio_candidate_ratio": float(ratio_mask.float().mean()),
                "negative_ratio": float(negative_mask.float().mean()),
                "top_ratio": float(top_mask.float().mean()),
                "selected_projection_mass_ratio": float(
                    (selected_projection_mass / total_projection_mass).mean()
                ),
                "mean_ratio": float(ratio.mean()),
                "selected_mean_ratio": float(
                    ratio[selected].mean() if selected.any() else torch.zeros((), device=ratio.device)
                ),
                "visual_support_mean": float(support.mean()),
                "text_activation_mean": float(text_abs.mean()),
                "activation_delta_norm": float(delta_norm.mean()),
                "changed_ratio": float(selected.any(dim=-1).float().mean()) if self.config.strength > 0 else 0.0,
            })
            if changed:
                return (edited,)
            return None

        return hook

    def _expand_batch(self, tensor, batch):
        if tensor.shape[0] == batch:
            return tensor
        if batch % tensor.shape[0]:
            return None
        return tensor.repeat_interleave(batch // tensor.shape[0], dim=0)

    def _top_fraction_mask(self, magnitude):
        if self.config.top_fraction <= 0:
            return torch.zeros_like(magnitude, dtype=torch.bool)
        count = max(1, math.ceil(magnitude.shape[-1] * self.config.top_fraction))
        if count >= magnitude.shape[-1]:
            return torch.ones_like(magnitude, dtype=torch.bool)
        indices = torch.topk(magnitude, k=count, dim=-1, sorted=False).indices
        result = torch.zeros_like(magnitude, dtype=torch.bool)
        return result.scatter(1, indices, True)

    def summary(self):
        if not self.records:
            return {"calls": 0}
        numeric_keys = set(self.records[0]) - {"layer", "phase"}

        def aggregate(records):
            if not records:
                return {}
            return {
                key: statistics.fmean(record[key] for record in records)
                for key in sorted(numeric_keys)
            }

        return {
            "calls": len(self.records),
            "means": aggregate(self.records),
            "prefill_means": aggregate([record for record in self.records if record["phase"] == "prefill"]),
            "decode_means": aggregate([record for record in self.records if record["phase"] == "decode"]),
            "per_layer": self.records,
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
