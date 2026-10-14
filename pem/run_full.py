#!/usr/bin/env python3
from __future__ import annotations

import json

from tqdm import tqdm

from llava.eval.pope_baseline import compute_metrics, load_jsonl
from pem.controller import PEMConfig, PEMController
from pem.eval_utils import (
    ROOT, SPLITS, add_fpr, aggregate_pem, chair_rows, efficiency,
    evaluate_chair, generate_chair, generate_pope, load_model, transitions,
)


def main():
    out = ROOT / "outputs/pem"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor = load_model()
    config = PEMConfig(enabled=True, strength=0.5, risk_bias=0.0)
    controller = PEMController(model, config)
    report = {
        "protocol": {
            "model": "LLaVA-1.5-7B",
            "vision_encoder": "CLIP ViT-L/14@336",
            "config": config.to_dict(),
            "pope": "all 8910; same prompt/parser; greedy; max_new_tokens=4",
            "chair": "same fixed 500 images; beam=3; max_new_tokens=64",
        },
        "pope": {},
    }
    for split in SPLITS:
        annotations = load_jsonl(ROOT / "DATA/POPE" / f"coco_pope_{split}.json")
        source = [{**row, "source_index": index} for index, row in enumerate(annotations)]
        rows = [generate_pope(row, tokenizer, model, processor, controller) for row in tqdm(source, desc=f"PEM full POPE {split}")]
        baseline_document = json.loads((ROOT / "outputs/pope_llava_baseline" / f"pope_{split}.json").read_text())
        baseline_rows = baseline_document["results"]
        report["pope"][split] = {
            "metrics": add_fpr(compute_metrics(rows)),
            "baseline_metrics": add_fpr(compute_metrics(baseline_rows)),
            "transitions": transitions(baseline_rows, rows),
            "efficiency": efficiency(rows),
            "statistics": aggregate_pem(rows),
        }
        (out / f"pope_{split}_full.json").write_text(json.dumps({
            "method": "PEM core_s05", "config": config.to_dict(),
            "num_samples": len(rows), "metrics": report["pope"][split]["metrics"],
            "results": [{k: v for k, v in row.items() if k != "pem_stats"} for row in rows],
        }, indent=2, ensure_ascii=False))

    chairs = [generate_chair(row, tokenizer, model, processor, controller) for row in tqdm(chair_rows(500), desc="PEM full CHAIR")]
    controller.close()
    chair_metrics = evaluate_chair(chairs, out / "chair_captions_pem_core_s05_full.jsonl")
    baseline_rows = [json.loads(line) for line in (ROOT / "outputs/chair/chair_captions_llava15_7b_pure.jsonl").read_text().splitlines() if line.strip()]
    baseline_metrics = evaluate_chair(baseline_rows, out / "chair_baseline_eval_copy.jsonl")
    report["chair"] = {
        "metrics": chair_metrics,
        "baseline_metrics": baseline_metrics,
        "efficiency": efficiency(chairs),
        "statistics": aggregate_pem(chairs),
        "caption_changes": sum(a["caption"] != b["caption"] for a, b in zip(baseline_rows, chairs)),
    }
    report["delta"] = {
        "pope_accuracy_pp": {split: 100 * (report["pope"][split]["metrics"]["accuracy"] - report["pope"][split]["baseline_metrics"]["accuracy"]) for split in SPLITS},
        "pope_f1_pp": {split: 100 * (report["pope"][split]["metrics"]["f1"] - report["pope"][split]["baseline_metrics"]["f1"]) for split in SPLITS},
        "chair": {key: chair_metrics[key] - baseline_metrics[key] for key in ("CHAIRs", "CHAIRi", "Recall", "Len", "ObjectCount")},
    }
    (out / "full_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
