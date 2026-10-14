#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics

import torch
from tqdm import tqdm

from ffn_concept_competition.eval_utils import generate_pope
from llava.eval.pope_baseline import compute_metrics, load_jsonl
from pem.eval_utils import ROOT, SPLITS, add_fpr, efficiency, load_model, pope_rows, transitions
from visual_ffn_gating import VisualFFNGatingConfig, VisualSupportSelectiveFFNGating


def logit_delta(baseline, edited):
    norms = [float(torch.linalg.vector_norm(b["_logits"] - a["_logits"])) for a, b in zip(baseline, edited)]
    margins = [b["yes_no_margin"] - a["yes_no_margin"] for a, b in zip(baseline, edited)]
    return {
        "changed_logits": sum(value > 0 for value in norms),
        "mean_l2": statistics.fmean(norms),
        "max_l2": max(norms),
        "mean_margin_delta": statistics.fmean(margins),
        "positive_margin_shifts": sum(value > 0 for value in margins),
        "negative_margin_shifts": sum(value < 0 for value in margins),
        "top1_changes": sum(int(a["_logits"].argmax()) != int(b["_logits"].argmax()) for a, b in zip(baseline, edited)),
    }


def aggregate_stats(rows):
    values = [row.get("visual_ffn_gating_stats", {}).get("means", {}) for row in rows]
    values = [value for value in values if value]
    keys = set().union(*(value.keys() for value in values)) if values else set()
    return {key: statistics.fmean(value[key] for value in values) for key in sorted(keys)}


def run_rows(rows, tokenizer, model, processor, config, capture, description):
    controller = VisualSupportSelectiveFFNGating(model, config)
    generated = [
        generate_pope(row, tokenizer, model, processor, controller, capture=capture)
        for row in tqdm(rows, desc=description)
    ]
    controller.close()
    return generated


def smoke(output_dir):
    data = pope_rows("random", per_class=64)
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    baseline = [generate_pope(row, tokenizer, model, processor, capture=True) for row in tqdm(data, desc="VFFN baseline")]
    configs = {
        "identity": VisualFFNGatingConfig(strength=0.0),
        "default": VisualFFNGatingConfig(),
    }
    report = {
        "protocol": "128 balanced POPE Random samples; greedy; max_new_tokens=4",
        "baseline": {"metrics": add_fpr(compute_metrics(baseline)), "efficiency": efficiency(baseline)},
        "experiments": {},
    }
    for name, config in configs.items():
        rows = run_rows(data, tokenizer, model, processor, config, True, f"VFFN smoke {name}")
        report["experiments"][name] = {
            "config": config.to_dict(),
            "metrics": add_fpr(compute_metrics(rows)),
            "transitions": transitions(baseline, rows),
            "logit_delta": logit_delta(baseline, rows),
            "statistics": aggregate_stats(rows),
            "efficiency": efficiency(rows),
        }
    if report["experiments"]["identity"]["logit_delta"]["max_l2"] != 0.0:
        raise RuntimeError("strength=0 failed exact identity check")
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Visual FFN gating changed model parameters")
    report["parameters_unchanged"] = True
    (output_dir / "smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def screen(output_dir):
    data = {split: pope_rows(split, per_class=50) for split in SPLITS}
    historical = [
        json.loads(line)
        for line in (ROOT / "outputs/ffn_concept_competition/pope_predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = {
        split: [row for row in historical if row["method"] == "baseline" and row["split"] == split]
        for split in SPLITS
    }
    for split in SPLITS:
        if [row["source_index"] for row in baseline[split]] != [row["source_index"] for row in data[split]]:
            raise RuntimeError(f"Historical baseline mismatch for {split}")
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    config = VisualFFNGatingConfig()
    edited = {
        split: run_rows(rows, tokenizer, model, processor, config, False, f"VFFN {split}")
        for split, rows in data.items()
    }
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Visual FFN gating changed model parameters")
    report = {
        "protocol": "POPE first 50/class (100/split); fixed default configuration",
        "config": config.to_dict(),
        "parameters_unchanged": True,
        "baseline": {split: add_fpr(compute_metrics(rows)) for split, rows in baseline.items()},
        "visual_ffn_gating": {
            split: {
                "metrics": add_fpr(compute_metrics(rows)),
                "transitions": transitions(baseline[split], rows),
                "statistics": aggregate_stats(rows),
                "efficiency": efficiency(rows),
            }
            for split, rows in edited.items()
        },
    }
    (output_dir / "pope_screening.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (output_dir / "pope_predictions.jsonl").open("w") as handle:
        for split in SPLITS:
            for row in baseline[split]:
                handle.write(json.dumps({**row, "method": "baseline", "split": split}, ensure_ascii=False) + "\n")
            for row in edited[split]:
                handle.write(json.dumps({**row, "method": "visual_ffn_gating", "split": split}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def ablate(output_dir):
    data = pope_rows("random", per_class=64)
    full = json.loads((ROOT / "outputs/pope_llava_baseline/pope_random.json").read_text())["results"]
    baseline = []
    for source in data:
        row = full[source["source_index"]]
        if row["image"] != source["image"] or row["question"] != source["text"]:
            raise RuntimeError(f"Baseline mismatch at source index {source['source_index']}")
        baseline.append({
            "source_index": source["source_index"], "image": row["image"],
            "question": row["question"], "label": row["label"].lower().strip(),
            "prediction": row["prediction"], "raw_output": row["raw_output"],
        })
    configs = {
        "lambda_020_p020": VisualFFNGatingConfig(strength=0.2, top_fraction=0.2),
        "lambda_040_p020": VisualFFNGatingConfig(strength=0.4, top_fraction=0.2),
        "lambda_060_p020": VisualFFNGatingConfig(strength=0.6, top_fraction=0.2),
        "lambda_030_p010": VisualFFNGatingConfig(strength=0.3, top_fraction=0.1),
        "lambda_030_p030": VisualFFNGatingConfig(strength=0.3, top_fraction=0.3),
    }
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    report = {
        "protocol": (
            "Sensitivity analysis on the same 128 balanced Random smoke samples; "
            "default configuration remains fixed independently of labels"
        ),
        "baseline": add_fpr(compute_metrics(baseline)),
        "experiments": {},
        "previously_tested": {
            "lambda_000_p020": "See smoke.json identity",
            "lambda_030_p020": "See smoke.json default",
        },
    }
    for name, config in configs.items():
        rows = run_rows(data, tokenizer, model, processor, config, False, f"VFFN ablation {name}")
        report["experiments"][name] = {
            "config": config.to_dict(),
            "metrics": add_fpr(compute_metrics(rows)),
            "transitions": transitions(baseline, rows),
            "statistics": aggregate_stats(rows),
            "efficiency": efficiency(rows),
        }
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Visual FFN gating ablation changed model parameters")
    report["parameters_unchanged"] = True
    (output_dir / "ablation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def full(output_dir):
    tokenizer, model, processor = load_model()
    versions = [parameter._version for parameter in model.parameters()]
    config = VisualFFNGatingConfig()
    report = {
        "protocol": "Complete POPE: Random 2910, Popular 3000, Adversarial 3000; fixed default configuration",
        "config": config.to_dict(),
        "baseline": {},
        "visual_ffn_gating": {},
    }
    for split in SPLITS:
        source_rows = load_jsonl(ROOT / "DATA" / "POPE" / f"coco_pope_{split}.json")
        data = [{**row, "source_index": index} for index, row in enumerate(source_rows)]
        baseline_payload = json.loads(
            (ROOT / "outputs" / "pope_llava_baseline" / f"pope_{split}.json").read_text()
        )
        baseline_rows = baseline_payload["results"]
        if len(baseline_rows) != len(data):
            raise RuntimeError(f"Full baseline length mismatch for {split}")
        prediction_path = output_dir / f"pope_full_{split}_predictions.jsonl"
        completed = []
        if prediction_path.exists():
            completed = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        completed_by_index = {row["source_index"]: row for row in completed}
        controller = VisualSupportSelectiveFFNGating(model, config)
        with prediction_path.open("a") as handle:
            for row in tqdm(data, desc=f"VFFN full {split}"):
                if row["source_index"] in completed_by_index:
                    continue
                generated = generate_pope(row, tokenizer, model, processor, controller, capture=False)
                generated["method"] = "visual_ffn_gating"
                generated["split"] = split
                handle.write(json.dumps(generated, ensure_ascii=False) + "\n")
                handle.flush()
                completed_by_index[row["source_index"]] = generated
        controller.close()
        edited = [completed_by_index[index] for index in range(len(data))]
        report["baseline"][split] = add_fpr(compute_metrics(baseline_rows))
        report["visual_ffn_gating"][split] = {
            "metrics": add_fpr(compute_metrics(edited)),
            "transitions": transitions(baseline_rows, edited),
            "statistics": aggregate_stats(edited),
            "efficiency": efficiency(edited),
        }
        (output_dir / "pope_full.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("Visual FFN gating full POPE changed model parameters")
    report["parameters_unchanged"] = True
    (output_dir / "pope_full.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "ablate", "screen", "full"), required=True)
    args = parser.parse_args()
    output_dir = ROOT / "results" / "idea1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(VisualFFNGatingConfig().to_dict(), indent=2))
    if args.mode == "smoke":
        smoke(output_dir)
    elif args.mode == "ablate":
        ablate(output_dir)
    elif args.mode == "screen":
        screen(output_dir)
    else:
        full(output_dir)


if __name__ == "__main__":
    main()
