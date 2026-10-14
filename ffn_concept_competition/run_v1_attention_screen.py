#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

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


def metrics(rows):
    return add_fpr(compute_metrics(rows))


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_attention"
    out.mkdir(parents=True, exist_ok=True)
    data = {split: pope_rows(split, per_class=50) for split in SPLITS}

    historical = [
        json.loads(line)
        for line in (ROOT / "outputs/ffn_concept_competition/pope_predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = {split: [row for row in historical if row["method"] == "baseline" and row["split"] == split] for split in SPLITS}
    original = {split: [row for row in historical if row["method"] == "fcc_s050_f005" and row["split"] == split] for split in SPLITS}
    for split in SPLITS:
        expected = [row["source_index"] for row in data[split]]
        if [row["source_index"] for row in baseline[split]] != expected:
            raise RuntimeError(f"Historical baseline does not match {split} screening data")
        if [row["source_index"] for row in original[split]] != expected:
            raise RuntimeError(f"Historical FCC-v1 does not match {split} screening data")

    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    config = FCCConfig(
        layers=(16, 18, 20, 22),
        strength=0.5,
        neuron_fraction=0.05,
        reference_mode="attention_output",
        repair_mode="minimal_projection",
    )
    controller = FFNConceptCompetitionController(model, config)
    edited = {
        split: [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(rows, desc=f"attention FCC {split}")]
        for split, rows in data.items()
    }
    controller.close()
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Attention-conditioned FCC changed model parameters")

    report = {
        "protocol": "POPE first 50 examples per class; configuration fixed before screening",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {split: metrics(rows) for split, rows in baseline.items()},
        "original_fcc_v1": {
            split: {"metrics": metrics(rows), "transitions": transitions(baseline[split], rows)}
            for split, rows in original.items()
        },
        "attention_fcc": {
            split: {
                "metrics": metrics(rows),
                "transitions": transitions(baseline[split], rows),
                "efficiency": efficiency(rows),
                "statistics": aggregate_stats(rows),
            }
            for split, rows in edited.items()
        },
    }
    (out / "screening.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "pope_predictions.jsonl").open("w") as handle:
        for split, rows in edited.items():
            for row in rows:
                handle.write(json.dumps({**row, "method": "attention_fcc", "split": split}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
