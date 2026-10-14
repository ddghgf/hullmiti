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


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_conflict"
    out.mkdir(parents=True, exist_ok=True)
    data = {split: pope_rows(split, per_class=50) for split in SPLITS}
    historical = [json.loads(line) for line in (ROOT / "outputs/ffn_concept_competition/pope_predictions.jsonl").read_text().splitlines() if line.strip()]
    baseline = {split: [row for row in historical if row["method"] == "baseline" and row["split"] == split] for split in SPLITS}
    for split in SPLITS:
        if [row["source_index"] for row in baseline[split]] != [row["source_index"] for row in data[split]]:
            raise RuntimeError(f"Historical baseline does not match {split}")

    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    config = FCCConfig(
        layers=(16, 18, 20, 22),
        strength=0.25,
        neuron_fraction=0.05,
        max_delta_norm_ratio=0.05,
        reference_mode="attention_output",
        repair_mode="conflict_neuron",
    )
    controller = FFNConceptCompetitionController(model, config)
    edited = {
        split: [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(rows, desc=f"conflict FCC {split}")]
        for split, rows in data.items()
    }
    controller.close()
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Conflict-gated FCC changed model parameters")

    report = {
        "protocol": "POPE first 50 examples per class; configuration fixed before screening",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {split: add_fpr(compute_metrics(rows)) for split, rows in baseline.items()},
        "conflict_fcc": {
            split: {
                "metrics": add_fpr(compute_metrics(rows)),
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
                handle.write(json.dumps({**row, "method": "conflict_fcc", "split": split}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
