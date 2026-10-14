#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

from tqdm import tqdm

from ffn_concept_competition.eval_utils import generate_chair
from pem.eval_utils import ROOT, chair_rows, efficiency, evaluate_chair, load_model
from visual_ffn_gating import VisualFFNGatingConfig, VisualSupportSelectiveFFNGating


def aggregate_stats(rows):
    values = [row.get("visual_ffn_gating_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def main():
    output_dir = ROOT / "results" / "idea1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = chair_rows(100)
    historical = [
        json.loads(line)
        for line in (ROOT / "outputs/ffn_concept_competition/chair_captions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = [row for row in historical if row["method"] == "baseline"]
    if [row["image_id"] for row in baseline] != [row["image_id"] for row in data]:
        raise RuntimeError("Historical CHAIR baseline does not match fixed 100-image set")

    config = VisualFFNGatingConfig()
    caption_path = output_dir / "chair_captions.jsonl"
    completed = []
    if caption_path.exists():
        completed = [json.loads(line) for line in caption_path.read_text().splitlines() if line.strip()]
    completed_by_id = {row["image_id"]: row for row in completed}
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    controller = VisualSupportSelectiveFFNGating(model, config)
    with caption_path.open("a") as handle:
        for row in tqdm(data, desc="VFFN CHAIR"):
            if row["image_id"] in completed_by_id:
                continue
            result = generate_chair(row, tokenizer, model, processor, controller)
            result["method"] = "visual_ffn_gating"
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed_by_id[row["image_id"]] = result
    controller.close()
    edited = [completed_by_id[row["image_id"]] for row in data]
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Visual FFN gating changed model parameters")

    report = {
        "protocol": "Fixed 100-image CHAIR; matched prompt; beam=3; max_new_tokens=64",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {
            "metrics": evaluate_chair(baseline, output_dir / "chair_baseline_scored.jsonl"),
            "efficiency": efficiency(baseline),
        },
        "visual_ffn_gating": {
            "metrics": evaluate_chair(edited, output_dir / "chair_visual_ffn_gating_scored.jsonl"),
            "efficiency": efficiency(edited),
            "caption_changes": sum(a["caption"] != b["caption"] for a, b in zip(baseline, edited)),
            "statistics": aggregate_stats(edited),
        },
    }
    (output_dir / "chair.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
