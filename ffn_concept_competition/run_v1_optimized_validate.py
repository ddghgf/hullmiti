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


def summarize(rows):
    return {
        "metrics": add_fpr(compute_metrics(rows)),
        "efficiency": efficiency(rows),
    }


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_optimized"
    out.mkdir(parents=True, exist_ok=True)

    # Screening used class offsets [0, 50); validation is strictly disjoint.
    data = {split: pope_rows(split, per_class=50, class_offset=50) for split in SPLITS}
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]

    baseline = {
        split: [generate_pope(row, tokenizer, model, processor) for row in tqdm(rows, desc=f"baseline-val {split}")]
        for split, rows in data.items()
    }

    config = FCCConfig(layers=(24,), strength=0.5, neuron_fraction=0.05)
    controller = FFNConceptCompetitionController(model, config)
    edited = {
        split: [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(rows, desc=f"layer24-val {split}")]
        for split, rows in data.items()
    }
    controller.close()

    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("FCC-v1 optimized validation changed model parameters")

    report = {
        "protocol": "Disjoint POPE validation: class offsets [50, 100), fixed after screening",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {split: summarize(rows) for split, rows in baseline.items()},
        "fcc_layer24": {
            split: {
                **summarize(rows),
                "transitions": transitions(baseline[split], rows),
                "statistics": aggregate_stats(rows),
            }
            for split, rows in edited.items()
        },
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "validation_predictions.jsonl").open("w") as handle:
        for method, groups in (("baseline", baseline), ("fcc_layer24", edited)):
            for split, rows in groups.items():
                for row in rows:
                    handle.write(json.dumps({**row, "method": method, "split": split}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
