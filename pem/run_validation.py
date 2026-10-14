#!/usr/bin/env python3
from __future__ import annotations

import json

from tqdm import tqdm

from llava.eval.pope_baseline import compute_metrics
from pem.controller import PEMConfig, PEMController
from pem.eval_utils import (
    ROOT, SPLITS, add_fpr, chair_rows, efficiency, evaluate_chair,
    generate_chair, generate_pope, load_model, pope_rows, transitions,
)


def main():
    out = ROOT / "outputs/pem"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor = load_model()
    pope_data = {split: pope_rows(split, 50, class_offset=50) for split in SPLITS}
    chair_data = chair_rows(100, offset=100)
    result = {
        "protocol": {
            "purpose": "held-out confirmation after screening",
            "pope": "next non-overlapping 50 yes + 50 no per split",
            "chair": "historical seed-0 image positions 101-200",
            "selected_config": {"strength": 0.5, "risk_bias": 0.0},
        },
        "baseline": {"pope": {}},
        "pem_core_s05": {"pope": {}},
    }
    base_pope = {}
    for split in SPLITS:
        rows = [generate_pope(x, tokenizer, model, processor) for x in tqdm(pope_data[split], desc=f"PEM validation baseline POPE {split}")]
        base_pope[split] = rows
        result["baseline"]["pope"][split] = {"metrics": add_fpr(compute_metrics(rows)), "efficiency": efficiency(rows)}
    base_chair = [generate_chair(x, tokenizer, model, processor) for x in tqdm(chair_data, desc="PEM validation baseline CHAIR")]
    result["baseline"]["chair"] = {
        "metrics": evaluate_chair(base_chair, out / "chair_validation_baseline_100.jsonl"),
        "efficiency": efficiency(base_chair),
    }
    controller = PEMController(model, PEMConfig(enabled=True, strength=0.5, risk_bias=0.0))
    pem_pope = {}
    for split in SPLITS:
        rows = [generate_pope(x, tokenizer, model, processor, controller) for x in tqdm(pope_data[split], desc=f"PEM validation core_s05 POPE {split}")]
        pem_pope[split] = rows
        result["pem_core_s05"]["pope"][split] = {
            "metrics": add_fpr(compute_metrics(rows)),
            "transitions": transitions(base_pope[split], rows),
            "efficiency": efficiency(rows),
        }
    pem_chair = [generate_chair(x, tokenizer, model, processor, controller) for x in tqdm(chair_data, desc="PEM validation core_s05 CHAIR")]
    controller.close()
    result["pem_core_s05"]["chair"] = {
        "metrics": evaluate_chair(pem_chair, out / "chair_validation_pem_core_s05_100.jsonl"),
        "efficiency": efficiency(pem_chair),
        "caption_changes": sum(a["caption"] != b["caption"] for a, b in zip(base_chair, pem_chair)),
    }
    result["decision_inputs"] = {
        "pope_accuracy_delta_pp": {
            split: 100 * (result["pem_core_s05"]["pope"][split]["metrics"]["accuracy"] - result["baseline"]["pope"][split]["metrics"]["accuracy"])
            for split in SPLITS
        },
        "chair_delta": {
            key: result["pem_core_s05"]["chair"]["metrics"][key] - result["baseline"]["chair"]["metrics"][key]
            for key in ("CHAIRs", "CHAIRi", "Recall", "Len", "ObjectCount")
        },
    }
    (out / "validation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
