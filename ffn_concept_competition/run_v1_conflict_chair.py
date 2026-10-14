#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

from tqdm import tqdm

from ffn_concept_competition.controller import FCCConfig, FFNConceptCompetitionController
from ffn_concept_competition.eval_utils import generate_chair
from pem.eval_utils import ROOT, chair_rows, efficiency, evaluate_chair, load_model


def aggregate_stats(rows):
    values = [row.get("fcc_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def main():
    out = ROOT / "outputs" / "ffn_concept_competition" / "v1_conflict"
    out.mkdir(parents=True, exist_ok=True)
    data = chair_rows(100)
    historical = [
        json.loads(line)
        for line in (ROOT / "outputs/ffn_concept_competition/chair_captions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = [row for row in historical if row["method"] == "baseline"]
    if [row["image_id"] for row in baseline] != [row["image_id"] for row in data]:
        raise RuntimeError("Historical CHAIR baseline does not match the fixed 100-image set")

    config = FCCConfig(
        layers=(16, 18, 20, 22),
        strength=0.25,
        neuron_fraction=0.05,
        max_delta_norm_ratio=0.05,
        reference_mode="attention_output",
        repair_mode="conflict_neuron",
    )
    caption_path = out / "chair_captions.jsonl"
    completed = []
    if caption_path.exists():
        completed = [json.loads(line) for line in caption_path.read_text().splitlines() if line.strip()]
    completed_by_id = {row["image_id"]: row for row in completed}

    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    controller = FFNConceptCompetitionController(model, config)
    with caption_path.open("a") as handle:
        for row in tqdm(data, desc="conflict FCC CHAIR"):
            if row["image_id"] in completed_by_id:
                continue
            result = generate_chair(row, tokenizer, model, processor, controller)
            result["method"] = "conflict_fcc"
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed_by_id[row["image_id"]] = result
    controller.close()
    edited = [completed_by_id[row["image_id"]] for row in data]
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Conflict-gated FCC CHAIR evaluation changed model parameters")

    report = {
        "protocol": "Fixed 100-image CHAIR set; beam=3; max_new_tokens=64; same prompt as baseline",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {
            "metrics": evaluate_chair(baseline, out / "chair_baseline_scored.jsonl"),
            "efficiency": efficiency(baseline),
        },
        "conflict_fcc": {
            "metrics": evaluate_chair(edited, out / "chair_conflict_fcc_scored.jsonl"),
            "efficiency": efficiency(edited),
            "caption_changes": sum(a["caption"] != b["caption"] for a, b in zip(baseline, edited)),
            "statistics": aggregate_stats(edited),
        },
    }
    (out / "chair.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
