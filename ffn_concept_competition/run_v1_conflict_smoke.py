#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

import torch
from tqdm import tqdm

from ffn_concept_competition.controller import FCCConfig, FFNConceptCompetitionController
from ffn_concept_competition.eval_utils import generate_pope
from llava.eval.pope_baseline import compute_metrics
from pem.eval_utils import ROOT, add_fpr, efficiency, load_model, pope_rows, transitions


def logit_delta(base, edited):
    norms = [float(torch.linalg.vector_norm(b["_logits"] - a["_logits"])) for a, b in zip(base, edited)]
    margins = [b["yes_no_margin"] - a["yes_no_margin"] for a, b in zip(base, edited)]
    return {
        "changed_logits": sum(value > 0 for value in norms),
        "mean_logit_l2": statistics.fmean(norms),
        "max_logit_l2": max(norms),
        "mean_yes_no_margin_delta": statistics.fmean(margins),
        "positive_margin_shifts": sum(value > 0 for value in margins),
        "negative_margin_shifts": sum(value < 0 for value in margins),
        "top1_changes": sum(int(a["_logits"].argmax()) != int(b["_logits"].argmax()) for a, b in zip(base, edited)),
    }


def aggregate_stats(rows):
    values = [row.get("fcc_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def config(strength):
    return FCCConfig(
        layers=(16, 18, 20, 22),
        strength=strength,
        neuron_fraction=0.05,
        max_delta_norm_ratio=0.05,
        reference_mode="attention_output",
        repair_mode="conflict_neuron",
    )


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_conflict"
    out.mkdir(parents=True, exist_ok=True)
    data = pope_rows("random", per_class=10)
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    baseline = [generate_pope(row, tokenizer, model, processor, capture=True) for row in tqdm(data, desc="conflict FCC baseline")]
    report = {
        "protocol": "20-sample balanced POPE Random smoke; no labels or output classes used by intervention",
        "baseline": {"metrics": add_fpr(compute_metrics(baseline)), "efficiency": efficiency(baseline)},
        "experiments": {},
    }
    for name, fcc_config in (("s025", config(0.25)), ("s050", config(0.5))):
        controller = FFNConceptCompetitionController(model, fcc_config)
        rows = [generate_pope(row, tokenizer, model, processor, controller, capture=True) for row in tqdm(data, desc=f"conflict FCC {name}")]
        controller.close()
        report["experiments"][name] = {
            "config": fcc_config.to_dict(),
            "metrics": add_fpr(compute_metrics(rows)),
            "transitions": transitions(baseline, rows),
            "logit_delta": logit_delta(baseline, rows),
            "statistics": aggregate_stats(rows),
            "efficiency": efficiency(rows),
        }
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Conflict-gated FCC changed model parameters")
    report["parameters_unchanged"] = True
    (out / "smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
