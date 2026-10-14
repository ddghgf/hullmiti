#!/usr/bin/env python3
from __future__ import annotations

import csv
import json

from tqdm import tqdm

from llava.eval.pope_baseline import compute_metrics
from pem.controller import PEMConfig, PEMController
from pem.eval_utils import ROOT, SPLITS, add_fpr, aggregate_pem, chair_rows, efficiency, evaluate_chair, generate_chair, generate_pope, load_model, pope_rows, transitions


CONFIGS = (
    ("pem_core_s025", PEMConfig(enabled=True, strength=0.25, risk_bias=0)),
    ("pem_core_s05", PEMConfig(enabled=True, strength=0.5, risk_bias=0)),
    ("pem_full_s025", PEMConfig(enabled=True, strength=0.25, risk_bias=0.25)),
    ("pem_full_s05", PEMConfig(enabled=True, strength=0.5, risk_bias=0.25)),
)


def clean_pope(row, method, split):
    return {"method": method, "split": split, **{k: row[k] for k in ("source_index", "image", "question", "label", "prediction", "raw_output", "seconds")}}


def clean_chair(row, method):
    return {"method": method, **{k: row[k] for k in ("image_id", "caption", "seconds", "peak_gpu_memory_bytes")}}


def main():
    out = ROOT / "outputs/pem"; out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, processor = load_model(); versions = [x._version for x in model.parameters()]
    pope_data = {split: pope_rows(split, 50) for split in SPLITS}; chair_data = chair_rows(100)
    report = {"protocol": {"pope": "50 yes + 50 no per split; greedy; max_new_tokens=4",
                            "chair": "same first 100 seed-0 baseline images; beam=3; max_new_tokens=64",
                            "vision_encoder": "CLIP ViT-L/14@336", "grid": [24,24]},
              "baseline": {"pope": {}}, "experiments": {}}
    all_pope, all_chair = [], []
    base_pope = {}
    for split in SPLITS:
        rows = [generate_pope(x, tokenizer, model, processor) for x in tqdm(pope_data[split], desc=f"PEM baseline POPE {split}")]
        base_pope[split] = rows; metrics = add_fpr(compute_metrics(rows))
        report["baseline"]["pope"][split] = {"metrics": metrics, "efficiency": efficiency(rows)}
        all_pope.extend(clean_pope(x, "baseline", split) for x in rows)
    base_chair = [generate_chair(x, tokenizer, model, processor) for x in tqdm(chair_data, desc="PEM baseline CHAIR")]
    report["baseline"]["chair"] = {"metrics": evaluate_chair(base_chair, out / "chair_baseline_100.jsonl"),
                                      "efficiency": efficiency(base_chair)}
    all_chair.extend(clean_chair(x, "baseline") for x in base_chair)
    for name, config in CONFIGS:
        controller = PEMController(model, config); item = {"config": config.to_dict(), "pope": {}}
        for split in SPLITS:
            rows = [generate_pope(x, tokenizer, model, processor, controller) for x in tqdm(pope_data[split], desc=f"PEM {name} POPE {split}")]
            item["pope"][split] = {"metrics": add_fpr(compute_metrics(rows)), "transitions": transitions(base_pope[split], rows),
                                           "efficiency": efficiency(rows), "statistics": aggregate_pem(rows)}
            all_pope.extend(clean_pope(x, name, split) for x in rows)
        chairs = [generate_chair(x, tokenizer, model, processor, controller) for x in tqdm(chair_data, desc=f"PEM {name} CHAIR")]
        controller.close()
        item["chair"] = {"metrics": evaluate_chair(chairs, out / f"chair_{name}_100.jsonl"), "efficiency": efficiency(chairs),
                         "statistics": aggregate_pem(chairs),
                         "caption_changes": sum(a["caption"] != b["caption"] for a,b in zip(base_chair, chairs))}
        all_chair.extend(clean_chair(x, name) for x in chairs); report["experiments"][name] = item
    if versions != [x._version for x in model.parameters()]: raise RuntimeError("PEM changed model weights")
    base_acc = sum(report["baseline"]["pope"][s]["metrics"]["accuracy"] for s in SPLITS) / 3
    candidates = list(report["experiments"])
    best = max(candidates, key=lambda n: (sum(report["experiments"][n]["pope"][s]["metrics"]["accuracy"] for s in SPLITS)/3,
                                          -report["experiments"][n]["chair"]["metrics"]["CHAIRi"]))
    chosen = report["experiments"][best]; chosen_acc = sum(chosen["pope"][s]["metrics"]["accuracy"] for s in SPLITS)/3
    base_chair = report["baseline"]["chair"]["metrics"]; new_chair = chosen["chair"]["metrics"]
    go = ((chosen_acc > base_acc and all(chosen["pope"][s]["metrics"]["f1"] >= report["baseline"]["pope"][s]["metrics"]["f1"]-0.01 for s in SPLITS))
          or (new_chair["CHAIRs"] < base_chair["CHAIRs"] and new_chair["CHAIRi"] < base_chair["CHAIRi"]
              and new_chair["Recall"] >= base_chair["Recall"]-1 and new_chair["Len"] >= base_chair["Len"]*0.95))
    report["decision"] = {"best": best, "baseline_macro_accuracy": base_acc, "best_macro_accuracy": chosen_acc,
                          "delta_accuracy_pp": 100*(chosen_acc-base_acc), "verdict": "GO" if go else "NO-GO",
                          "run_full": go}
    (out / "screening_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "pope_predictions.jsonl").open("w") as f:
        for x in all_pope: f.write(json.dumps(x, ensure_ascii=False)+"\n")
    with (out / "chair_captions.jsonl").open("w") as f:
        for x in all_chair: f.write(json.dumps(x, ensure_ascii=False)+"\n")
    with (out / "pope_screening.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(("method","split","accuracy","f1","precision","recall","yes_ratio","fpr","changes","fixed","broken","mean_seconds"))
        for split in SPLITS:
            x=report["baseline"]["pope"][split]; m=x["metrics"]; w.writerow(("baseline",split,m["accuracy"],m["f1"],m["precision"],m["recall"],m["yes_ratio"],m["false_positive_rate"],0,0,0,x["efficiency"]["mean_seconds"]))
        for name in candidates:
            for split in SPLITS:
                x=report["experiments"][name]["pope"][split]; m=x["metrics"]; t=x["transitions"]
                w.writerow((name,split,m["accuracy"],m["f1"],m["precision"],m["recall"],m["yes_ratio"],m["false_positive_rate"],t["prediction_changes"],t["incorrect_to_correct"],t["correct_to_incorrect"],x["efficiency"]["mean_seconds"]))
    with (out / "chair_screening.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(("method","CHAIRs","CHAIRi","Recall","Len","ObjectCount","changes","mean_seconds"))
        x=report["baseline"]["chair"]; m=x["metrics"]; w.writerow(("baseline",m["CHAIRs"],m["CHAIRi"],m["Recall"],m["Len"],m["ObjectCount"],0,x["efficiency"]["mean_seconds"]))
        for name in candidates:
            x=report["experiments"][name]["chair"]; m=x["metrics"]; w.writerow((name,m["CHAIRs"],m["CHAIRi"],m["Recall"],m["Len"],m["ObjectCount"],x["caption_changes"],x["efficiency"]["mean_seconds"]))
    stats={name:{"pope":{s:report["experiments"][name]["pope"][s]["statistics"] for s in SPLITS},"chair":report["experiments"][name]["chair"]["statistics"]} for name in candidates}
    (out/"patch_statistics.json").write_text(json.dumps(stats,indent=2)); runtime={"baseline":report["baseline"],"experiments":{n:{"pope":{s:report["experiments"][n]["pope"][s]["efficiency"] for s in SPLITS},"chair":report["experiments"][n]["chair"]["efficiency"]} for n in candidates}}
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2)); print(json.dumps({"baseline":report["baseline"],"experiments":report["experiments"],"decision":report["decision"]},indent=2,ensure_ascii=False))


if __name__ == "__main__": main()
