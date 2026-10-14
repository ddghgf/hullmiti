from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass(frozen=True)
class FCCConfig:
    enabled: bool = True
    layers: tuple[int, ...] = (16, 20, 24, 28)
    strength: float = 0.25
    neuron_fraction: float = 0.05
    min_negative_mass_ratio: float = 0.0
    max_delta_norm_ratio: float = 0.0
    prefill_only: bool = False
    reference_mode: str = "mean"
    repair_mode: str = "neuron"
    retrieval_temperature: float = 0.1
    conflict_margin: float = 0.0
    eps: float = 1e-6

    def to_dict(self):
        value = asdict(self); value['layers'] = list(self.layers); return value


class FFNConceptCompetitionController:
    stats_key = 'fcc_stats'

    def __init__(self, model, config: FCCConfig):
        if config.reference_mode not in {"mean", "retrieval_innovation", "attention_output"}:
            raise ValueError(f"Unknown FCC reference mode: {config.reference_mode}")
        if config.repair_mode not in {"neuron", "conflict_neuron", "minimal_projection"}:
            raise ValueError(f"Unknown FCC repair mode: {config.repair_mode}")
        if config.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")
        self.model, self.config = model, config
        self.handles, self.visual_refs, self.visual_states, self.attention_refs, self.records = [], {}, {}, {}, []
        self.image_start = self.original_length = None
        for layer in config.layers:
            decoder_layer = model.get_model().layers[layer]
            if config.reference_mode == "attention_output":
                self.handles.append(decoder_layer.self_attn.register_forward_hook(self._attention_hook(layer)))
            mlp = decoder_layer.mlp
            self.handles.append(mlp.register_forward_hook(self._hook(layer)))

    def begin_sample(self, input_ids=None):
        self.visual_refs, self.visual_states, self.attention_refs, self.records = {}, {}, {}, []
        if input_ids is None: return
        positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        if positions.numel() != 1:
            raise ValueError(f'FCC currently expects one image token, got {positions.numel()}')
        self.image_start = int(positions[0]); self.original_length = int(input_ids.shape[1])

    def _attention_hook(self, layer):
        @torch.no_grad()
        def hook(module, inputs, output):
            if not self.config.enabled or self.image_start is None:
                return output
            attention_output = output[0] if isinstance(output, tuple) else output
            if attention_output.ndim != 3:
                return output
            self.attention_refs[layer] = attention_output[:, -1, :].detach().float()
            return output
        return hook

    def _hook(self, layer):
        @torch.no_grad()
        def hook(module, inputs, output):
            if not self.config.enabled or self.image_start is None: return output
            x = inputs[0]
            is_prefill = x.shape[1] > 1
            if is_prefill:
                visual_count = x.shape[1] - (self.original_length - 1)
                if visual_count <= 0: return output
                start, end = self.image_start, self.image_start + visual_count
                visual_states = x[:, start:end, :].float()
                visual_ref = visual_states.mean(dim=1)
                self.visual_refs[layer] = F.normalize(visual_ref, dim=-1, eps=self.config.eps)
                if self.config.reference_mode == "retrieval_innovation":
                    self.visual_states[layer] = visual_states
            elif layer not in self.visual_refs:
                return output
            if not is_prefill and self.config.prefill_only: return output
            if self.config.strength == 0 or self.config.neuron_fraction <= 0: return output
            answer = x[:, -1, :]
            ref = self.visual_refs[layer]
            reference_norm = torch.ones(ref.shape[0], device=ref.device, dtype=ref.dtype)
            retrieval_entropy = torch.zeros_like(reference_norm)
            if self.config.reference_mode == "attention_output":
                attention_ref = self.attention_refs.get(layer)
                if attention_ref is None:
                    return output
                reference_norm = torch.linalg.vector_norm(attention_ref, dim=-1)
                ref = F.normalize(attention_ref, dim=-1, eps=self.config.eps)
            elif self.config.reference_mode == "retrieval_innovation":
                visual_states = self.visual_states.get(layer)
                if visual_states is None:
                    return output
                if answer.shape[0] != visual_states.shape[0]:
                    if answer.shape[0] % visual_states.shape[0]: return output
                    visual_states = visual_states.repeat_interleave(answer.shape[0] // visual_states.shape[0], dim=0)
                query = F.normalize(answer.float(), dim=-1, eps=self.config.eps)
                visual_unit = F.normalize(visual_states, dim=-1, eps=self.config.eps)
                scores = torch.einsum("bnh,bh->bn", visual_unit, query)
                weights = torch.softmax(scores / self.config.retrieval_temperature, dim=-1)
                retrieved = torch.einsum("bn,bnh->bh", weights, visual_states)
                innovation = retrieved - visual_states.mean(dim=1)
                reference_norm = torch.linalg.vector_norm(innovation, dim=-1)
                ref = F.normalize(innovation, dim=-1, eps=self.config.eps)
                retrieval_entropy = -(weights * torch.log(weights + self.config.eps)).sum(dim=-1)
            if ref.shape[0] != x.shape[0]:
                if x.shape[0] % ref.shape[0]: return output
                ref = ref.repeat_interleave(x.shape[0] // ref.shape[0], dim=0)
            activation = (F.silu(module.gate_proj(answer)) * module.up_proj(answer)).float()
            direction = torch.matmul(ref, module.down_proj.weight.detach().float())
            projection = activation * direction
            negative = projection < -self.config.eps
            negative_mass_ratio = (projection.clamp_max(0).abs().sum(dim=-1) /
                                   (projection.abs().sum(dim=-1) + self.config.eps))
            requested = max(1, int(projection.shape[-1] * self.config.neuron_fraction))
            selected = torch.zeros_like(negative)
            for row in range(projection.shape[0]):
                if negative_mass_ratio[row] < self.config.min_negative_mass_ratio:
                    continue
                candidates = torch.nonzero(negative[row], as_tuple=False).flatten()
                if candidates.numel():
                    order = torch.argsort(projection[row, candidates])
                    selected[row, candidates[order[:min(requested, candidates.numel())]]] = True
            selected_activation = activation * selected
            removed = F.linear(selected_activation.to(module.down_proj.weight.dtype),
                               module.down_proj.weight, bias=None)
            positive_mass = projection.clamp_min(0).sum(dim=-1)
            negative_mass = projection.clamp_max(0).abs().sum(dim=-1)
            signed_projection = positive_mass - negative_mass
            conflict = (-signed_projection - self.config.conflict_margin).clamp_min(0)
            selected_negative_mass = (projection * selected).sum(dim=-1).neg().clamp_min(0)
            if self.config.repair_mode == "minimal_projection":
                # Minimum-L2 repair: preserve every component orthogonal to the
                # retrieved visual evidence and never push the FFN past neutral.
                repair_scalar = torch.minimum(
                    self.config.strength * selected_negative_mass,
                    conflict,
                )
                delta = -repair_scalar.unsqueeze(-1) * ref
            else:
                repair_scalar = torch.zeros_like(conflict)
                delta = self.config.strength * removed.float()
                if self.config.repair_mode == "conflict_neuron":
                    delta = delta * (conflict > 0).to(delta.dtype).unsqueeze(-1)
            budget_scale = torch.ones((delta.shape[0], 1), device=delta.device, dtype=delta.dtype)
            if self.config.max_delta_norm_ratio > 0:
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                output_norm = torch.linalg.vector_norm(output[:, -1, :].float(), dim=-1, keepdim=True)
                budget = self.config.max_delta_norm_ratio * output_norm
                budget_scale = torch.clamp(budget / (delta_norm + self.config.eps), max=1.0)
                delta = delta * budget_scale
            if not torch.isfinite(delta).all(): return output
            edited = output.clone(); edited[:, -1, :] -= delta.to(output.dtype)
            self.records.append({'layer':layer, 'negative_ratio':float(negative.float().mean()),
                'negative_mass_ratio':float(negative_mass_ratio.mean()),
                'selected_ratio':float(selected.float().mean()),
                'triggered_ratio':float(selected.any(dim=-1).float().mean()),
                'projection_min':float(projection.min()),
                'projection_mean':float(projection.mean()),
                'signed_projection':float(signed_projection.mean()),
                'conflict_ratio':float((negative_mass / (positive_mass + negative_mass + self.config.eps)).mean()),
                'conflict_triggered_ratio':float((conflict > 0).float().mean()),
                'repair_scalar':float(repair_scalar.mean()),
                'reference_norm':float(reference_norm.mean()),
                'retrieval_entropy':float(retrieval_entropy.mean()),
                'removed_norm':float(torch.linalg.vector_norm(removed.float())),
                'output_delta_norm':float(torch.linalg.vector_norm(delta)),
                'budget_scale':float(budget_scale.mean())})
            return edited
        return hook

    def summary(self):
        if not self.records: return {'calls':0}
        keys = set(self.records[0]) - {'layer'}
        return {'calls':len(self.records), 'means':{key:statistics.fmean(x[key] for x in self.records) for key in keys},
                'per_layer':self.records}

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
