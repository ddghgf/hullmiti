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
    return {"metrics": add_fpr(compute_metrics(rows)), "efficiency": efficiency(rows)}


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_attention"
    out.mkdir(parents=True, exist_ok=True)
    data = {split: pope_rows(split, per_class=50, class_offset=50) for split in SPLITS}
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    baseline = {
        split: [generate_pope(row, tokenizer, model, processor) for row in tqdm(rows, desc=f"attention baseline-val {split}")]
        for split, rows in data.items()
    }
    config = FCCConfig(
        layers=(16, 18, 20, 22),
        strength=0.5,
        neuron_fraction=0.05,
        reference_mode="attention_output",
        repair_mode="minimal_projection",
    )
    controller = FFNConceptCompetitionController(model, config)
    edited = {
        split: [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(rows, desc=f"attention FCC-val {split}")]
        for split, rows in data.items()
    }
    controller.close()
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Attention-conditioned FCC validation changed model parameters")
    report = {
        "protocol": "Disjoint POPE validation: class offsets [50, 100), configuration fixed after screening",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {split: summarize(rows) for split, rows in baseline.items()},
        "attention_fcc": {
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
        for method, groups in (("baseline", baseline), ("attention_fcc", edited)):
            for split, rows in groups.items():
                for row in rows:
                    handle.write(json.dumps({**row, "method": method, "split": split}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
