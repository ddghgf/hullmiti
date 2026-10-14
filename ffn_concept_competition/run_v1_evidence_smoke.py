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
    margin_delta = [b["yes_no_margin"] - a["yes_no_margin"] for a, b in zip(base, edited)]
    return {
        "changed_logits": sum(value > 0 for value in norms),
        "mean_logit_l2": statistics.fmean(norms),
        "max_logit_l2": max(norms),
        "mean_yes_no_margin_delta": statistics.fmean(margin_delta),
        "positive_margin_shifts": sum(value > 0 for value in margin_delta),
        "negative_margin_shifts": sum(value < 0 for value in margin_delta),
        "top1_changes": sum(int(a["_logits"].argmax()) != int(b["_logits"].argmax()) for a, b in zip(base, edited)),
    }


def aggregate_stats(rows):
    values = [row.get("fcc_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def evidence_config(layers, strength, fraction):
    return FCCConfig(
        layers=layers,
        strength=strength,
        neuron_fraction=fraction,
        reference_mode="retrieval_innovation",
        repair_mode="minimal_projection",
        retrieval_temperature=0.1,
    )


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_evidence"
    out.mkdir(parents=True, exist_ok=True)
    data = pope_rows("random", per_class=10)
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]

    baseline = [
        generate_pope(row, tokenizer, model, processor, capture=True)
        for row in tqdm(data, desc="evidence FCC smoke baseline")
    ]
    configs = {
        "mid_s050_f005": evidence_config((16, 18, 20, 22), 0.5, 0.05),
        "mid_s100_f010": evidence_config((16, 18, 20, 22), 1.0, 0.10),
        "spread_s100_f010": evidence_config((16, 20, 24, 28), 1.0, 0.10),
    }
    report = {
        "protocol": "20-sample balanced POPE Random smoke; no labels are used by the intervention",
        "baseline": {
            "metrics": add_fpr(compute_metrics(baseline)),
            "efficiency": efficiency(baseline),
        },
        "experiments": {},
    }
    predictions = []
    for name, config in configs.items():
        controller = FFNConceptCompetitionController(model, config)
        rows = [
            generate_pope(row, tokenizer, model, processor, controller, capture=True)
            for row in tqdm(data, desc=f"evidence FCC smoke {name}")
        ]
        controller.close()
        report["experiments"][name] = {
            "config": config.to_dict(),
            "metrics": add_fpr(compute_metrics(rows)),
            "transitions": transitions(baseline, rows),
            "logit_delta": logit_delta(baseline, rows),
            "statistics": aggregate_stats(rows),
            "efficiency": efficiency(rows),
        }
        predictions.extend({**row, "method": name} for row in rows)

    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Evidence-conditioned FCC changed model parameters")
    report["parameters_unchanged"] = True
    (out / "smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "smoke_predictions.jsonl").open("w") as handle:
        for row in predictions:
            clean = {key: value for key, value in row.items() if key != "_logits"}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
