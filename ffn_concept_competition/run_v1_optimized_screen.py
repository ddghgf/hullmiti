#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

import torch
from tqdm import tqdm

from ffn_concept_competition.controller import FCCConfig, FFNConceptCompetitionController
from ffn_concept_competition.eval_utils import generate_pope
from llava.eval.pope_baseline import compute_metrics
from pem.eval_utils import ROOT, SPLITS, add_fpr, efficiency, load_model, pope_rows, transitions


def aggregate_stats(rows):
    values = [row.get("fcc_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_optimized"
    out.mkdir(parents=True, exist_ok=True)
    data = {split: pope_rows(split, per_class=50) for split in SPLITS}

    saved = [json.loads(line) for line in (ROOT / "outputs/ffn_concept_competition/pope_predictions.jsonl").read_text().splitlines() if line.strip()]
    baseline = {split: [row for row in saved if row["method"] == "baseline" and row["split"] == split] for split in SPLITS}
    for split in SPLITS:
        if [row["source_index"] for row in baseline[split]] != [row["source_index"] for row in data[split]]:
            raise RuntimeError(f"Historical baseline does not match {split} data")

    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]

    # Verify the modified controller still has a strict identity setting.
    smoke_data = data["random"][:10]
    smoke_base = [generate_pope(row, tokenizer, model, processor, capture=True) for row in tqdm(smoke_data, desc="v1-opt smoke baseline")]
    zero_controller = FFNConceptCompetitionController(model, FCCConfig(strength=0.0))
    smoke_zero = [generate_pope(row, tokenizer, model, processor, zero_controller, True) for row in tqdm(smoke_data, desc="v1-opt smoke identity")]
    zero_controller.close()
    max_logit_error = max(float(torch.max(torch.abs(a["_logits"] - b["_logits"]))) for a, b in zip(smoke_base, smoke_zero))
    if max_logit_error != 0.0:
        raise RuntimeError(f"strength=0 is not exact: {max_logit_error}")

    configs = {
        "layer16": FCCConfig(layers=(16,), strength=0.5, neuron_fraction=0.05),
        "layer20": FCCConfig(layers=(20,), strength=0.5, neuron_fraction=0.05),
        "layer24": FCCConfig(layers=(24,), strength=0.5, neuron_fraction=0.05),
        "layer28": FCCConfig(layers=(28,), strength=0.5, neuron_fraction=0.05),
        "without28": FCCConfig(layers=(16, 20, 24), strength=0.5, neuron_fraction=0.05),
        "all_gated": FCCConfig(layers=(16, 20, 24, 28), strength=0.5, neuron_fraction=0.05,
                               min_negative_mass_ratio=0.55, max_delta_norm_ratio=0.05),
    }
    report = {
        "protocol": "POPE screening: historical baseline and first 50 examples per class",
        "identity_max_logit_error": max_logit_error,
        "experiments": {},
    }
    prediction_rows = []
    for name, config in configs.items():
        controller = FFNConceptCompetitionController(model, config)
        edited = {
            split: [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(rows, desc=f"{name} {split}")]
            for split, rows in data.items()
        }
        controller.close()
        report["experiments"][name] = {
            "config": config.to_dict(),
            "pope": {
                split: {
                    "metrics": add_fpr(compute_metrics(rows)),
                    "transitions": transitions(baseline[split], rows),
                    "efficiency": efficiency(rows),
                    "statistics": aggregate_stats(rows),
                }
                for split, rows in edited.items()
            },
        }
        prediction_rows.extend({**row, "method": name, "split": split} for split, rows in edited.items() for row in rows)

    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("FCC-v1 optimized controller changed model parameters")
    report["parameters_unchanged"] = True
    (out / "screening.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "pope_predictions.jsonl").open("w") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
