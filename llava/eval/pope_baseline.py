#!/usr/bin/env python3
"""Evaluate the original LLaVA model on POPE.

This file intentionally has no dependency on ``hulluedit``.  It loads LLaVA
and calls ``model.generate`` directly.  Its prompt, decoding limits, early
stopping, and answer parsing match the project's HulluEdit POPE evaluation so
the two methods can be compared fairly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import StoppingCriteria


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original LLaVA-1.5 POPE baseline (no HulluEdit)"
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "llava-v1.5-7b"),
        help="Local LLaVA checkpoint",
    )
    parser.add_argument(
        "--pope-root",
        default=str(PROJECT_ROOT / "DATA" / "POPE"),
        help="Directory containing coco_pope_{split}.json",
    )
    parser.add_argument(
        "--image-folder",
        default=str(PROJECT_ROOT / "DATA" / "val2014"),
        help="COCO validation image directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "pope_llava_pure"),
    )
    parser.add_argument(
        "--split",
        choices=("all",) + SPLITS,
        default="all",
        help="Evaluate one split or all three splits",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split limit for a quick test",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"POPE annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(question: str) -> str:
    """Use exactly the same prompt as hulluedit/eval/pope_eval.py."""
    return f"USER: <image>\n{question} Answer with yes or no only.\nASSISTANT:"


def pope_prediction(text: str) -> str:
    """Match extract_yes_no in hulluedit/eval/pope_eval.py exactly."""
    if not text:
        return "no"
    prefix = text.lower().strip()[:30]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "no"


class YesNoStoppingCriteria(StoppingCriteria):
    """Stop when a standalone yes/no appears in the first 30 output chars."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, output_ids, scores, **kwargs) -> bool:
        generated_ids = output_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return re.search(r"\b(?:yes|no)\b", text[:30], flags=re.IGNORECASE) is not None


def compute_metrics(results: Iterable[Dict]) -> Dict[str, float | int]:
    rows = list(results)
    tp = sum(r["prediction"] == "yes" and r["label"] == "yes" for r in rows)
    fp = sum(r["prediction"] == "yes" and r["label"] == "no" for r in rows)
    tn = sum(r["prediction"] == "no" and r["label"] == "no" for r in rows)
    fn = sum(r["prediction"] == "no" and r["label"] == "yes" for r in rows)
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / total if total else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@torch.inference_mode()
def generate_answer(
    *,
    model,
    tokenizer,
    image_processor,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[str, str]:
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(question)

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [item.to(model.device, dtype=torch.float16) for item in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[YesNoStoppingCriteria(tokenizer, input_ids.shape[1])],
    )

    generated_ids = output_ids[0, input_ids.shape[1] :]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw_output, prompt


def evaluate_split(
    *,
    split: str,
    annotations: List[Dict],
    image_folder: Path,
    model,
    tokenizer,
    image_processor,
    args: argparse.Namespace,
) -> List[Dict]:
    results: List[Dict] = []
    for item in tqdm(annotations, desc=f"POPE {split}"):
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = image_folder / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")

        raw_output, prompt = generate_answer(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=image_path,
            question=item["text"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        results.append(
            {
                "question_id": item.get("question_id"),
                "image": item["image"],
                "question": item["text"],
                "prompt": prompt,
                "label": item["label"].lower().strip(),
                "prediction": pope_prediction(raw_output),
                "raw_output": raw_output,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be positive")

    model_path = Path(args.model_path).expanduser().resolve()
    pope_root = Path(args.pope_root).expanduser().resolve()
    image_folder = Path(args.image_folder).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the LLaVA-1.5-7B baseline")

    splits = SPLITS if args.split == "all" else (args.split,)
    split_data: Dict[str, List[Dict]] = {}
    for split in splits:
        rows = load_jsonl(pope_root / f"coco_pope_{split}.json")
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        split_data[split] = rows
        size_note = " (standard POPE uses 3000)" if args.max_samples is None and len(rows) != 3000 else ""
        print(f"[POPE] {split}: {len(rows)} samples{size_note}")

    print(f"[LLaVA] Loading original model once: {model_path}")
    disable_torch_init()
    model_name = get_model_name_from_path(str(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        None,
        model_name,
        device="cuda",
    )
    model.eval()

    combined_results: List[Dict] = []
    per_split: Dict[str, Dict[str, float | int]] = {}
    for split in splits:
        results = evaluate_split(
            split=split,
            annotations=split_data[split],
            image_folder=image_folder,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
        )
        metrics = compute_metrics(results)
        per_split[split] = metrics
        combined_results.extend(results)

        split_output = {
            "method": "Original LLaVA baseline",
            "uses_hulluedit": False,
            "model_path": str(model_path),
            "evaluation_protocol": "matched_to_hulluedit_pope",
            "prompt_template": "USER: <image>\\n{question} Answer with yes or no only.\\nASSISTANT:",
            "decoding": {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "stop_on_yes_no": True,
            },
            "split": split,
            "num_samples": len(results),
            "metrics": metrics,
            "results": results,
        }
        split_path = output_dir / f"pope_{split}.json"
        split_path.write_text(
            json.dumps(split_output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[POPE] {split}: Acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} -> {split_path}"
        )

    summary = {
        "method": "Original LLaVA baseline",
        "uses_hulluedit": False,
        "evaluation_protocol": "matched_to_hulluedit_pope",
        "num_samples": len(combined_results),
        "overall": compute_metrics(combined_results),
        "per_split": per_split,
    }
    summary_path = output_dir / "pope_all_metrics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[POPE] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
