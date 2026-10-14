#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics

import torch
from tqdm import tqdm

from llava.eval.pope_baseline import compute_metrics
from pem.controller import PEMConfig, PEMController
from pem.eval_utils import ROOT, add_fpr, aggregate_pem, chair_rows, efficiency, evaluate_chair, generate_chair, generate_pope, load_model, pope_rows


def compare_logits(base, method, margins=False):
    norms, maximums, argmax, margin_delta = [], [], 0, []
    for a, b in zip(base, method):
        delta = b["_logits"] - a["_logits"]
        norms.append(float(torch.linalg.vector_norm(delta))); maximums.append(float(delta.abs().max()))
        argmax += int(a["_logits"].argmax()) != int(b["_logits"].argmax())
        if margins: margin_delta.append(b["yes_no_margin"] - a["yes_no_margin"])
    result = {"changed": sum(x > 0 for x in norms), "argmax_changed": argmax,
              "mean_l2": statistics.fmean(norms), "max_abs": max(maximums)}
    if margins: result.update(mean_margin_delta=statistics.fmean(margin_delta), max_abs_margin_delta=max(map(abs, margin_delta)))
    return result


def public(row): return {key: value for key, value in row.items() if key != "_logits"}


def main():
    out = ROOT / "outputs/pem"; out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor = load_model(); versions = [x._version for x in model.parameters()]
    pope_data, chair_data = pope_rows("random", 10), chair_rows(10)
    base_pope = [generate_pope(x, tokenizer, model, processor, capture=True) for x in tqdm(pope_data, desc="PEM smoke POPE baseline")]
    base_chair = [generate_chair(x, tokenizer, model, processor, capture=True) for x in tqdm(chair_data, desc="PEM smoke CHAIR baseline")]
    report = {"baseline": {"pope_metrics": add_fpr(compute_metrics(base_pope)), "pope_efficiency": efficiency(base_pope),
                            "chair_metrics": evaluate_chair(base_chair, out / "_smoke_chair_baseline.jsonl"),
                            "chair_efficiency": efficiency(base_chair)}, "configs": {}}
    configs = (("strength0", PEMConfig(enabled=True, strength=0)),
               ("pem_core", PEMConfig(enabled=True, strength=1, risk_bias=0)),
               ("pem_full", PEMConfig(enabled=True, strength=1, risk_bias=0.25)))
    for name, config in configs:
        controller = PEMController(model, config)
        pope = [generate_pope(x, tokenizer, model, processor, controller, True) for x in tqdm(pope_data, desc=f"PEM smoke POPE {name}")]
        chair = [generate_chair(x, tokenizer, model, processor, controller, True) for x in tqdm(chair_data, desc=f"PEM smoke CHAIR {name}")]
        controller.close()
        report["configs"][name] = {"config": config.to_dict(), "pope_metrics": add_fpr(compute_metrics(pope)),
            "chair_metrics": evaluate_chair(chair, out / f"_smoke_chair_{name}.jsonl"),
            "pope_prediction_changes": sum(a["prediction"] != b["prediction"] for a,b in zip(base_pope,pope)),
            "chair_caption_changes": sum(a["caption"] != b["caption"] for a,b in zip(base_chair,chair)),
            "pope_logit_change": compare_logits(base_pope, pope, True), "chair_logit_change": compare_logits(base_chair, chair),
            "pope_statistics": aggregate_pem(pope), "chair_statistics": aggregate_pem(chair),
            "pope_efficiency": efficiency(pope), "chair_efficiency": efficiency(chair),
            "pope_results": [public(x) for x in pope], "chair_results": [public(x) for x in chair]}
    if versions != [x._version for x in model.parameters()]: raise RuntimeError("PEM changed model weights")
    report["parameters_unchanged"] = True
    (out / "smoke_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"baseline": report["baseline"], "configs": {k: {x:y for x,y in v.items() if not x.endswith("results")} for k,v in report["configs"].items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
