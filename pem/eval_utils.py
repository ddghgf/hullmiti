from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hulluedit.eval.eval_chair import compute_chair_metrics, scale_chair_metrics
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.pope_baseline import YesNoStoppingCriteria, build_prompt, compute_metrics, load_jsonl, pope_prediction
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("random", "popular", "adversarial")


def setup_seeds(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    setup_seeds(); disable_torch_init()
    path = str((ROOT / "llava-v1.5-7b").resolve())
    tokenizer, model, processor, _ = load_pretrained_model(path, None, get_model_name_from_path(path), device="cuda")
    model.eval()
    return tokenizer, model, processor


def prepare_image(path, processor, model):
    image = Image.open(path).convert("RGB")
    tensor = process_images([image], processor, model.config)
    if isinstance(tensor, list):
        return [part.to(model.device, dtype=torch.float16) for part in tensor]
    return tensor.to(model.device, dtype=torch.float16)


def pope_rows(split, per_class=50, class_offset=0):
    rows = load_jsonl(ROOT / "DATA" / "POPE" / f"coco_pope_{split}.json")
    selected, counts = [], {"yes": 0, "no": 0}
    seen_counts = {"yes": 0, "no": 0}
    for index, row in enumerate(rows):
        label = row["label"].lower().strip()
        if label in counts and seen_counts[label] >= class_offset and counts[label] < per_class:
            selected.append({**row, "source_index": index}); counts[label] += 1
        if label in seen_counts:
            seen_counts[label] += 1
        if counts == {"yes": per_class, "no": per_class}: break
    if counts != {"yes": per_class, "no": per_class}: raise ValueError(str(counts))
    return selected


def token_candidates(tokenizer, words):
    return sorted({ids[0] for word in words if len(ids := tokenizer.encode(word, add_special_tokens=False)) == 1})


def margin(logits, tokenizer):
    yes = token_candidates(tokenizer, ("yes", "Yes", " yes", " Yes"))
    no = token_candidates(tokenizer, ("no", "No", " no", " No"))
    if not yes or not no: raise RuntimeError("yes/no token candidates unavailable")
    return float(logits[yes].max() - logits[no].max())


@torch.inference_mode()
def generate_pope(row, tokenizer, model, processor, controller=None, capture=False):
    images = prepare_image(ROOT / "DATA" / "val2014" / row["image"], processor, model)
    ids = tokenizer_image_token(build_prompt(row["text"]), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    if controller: controller.begin_sample()
    torch.cuda.reset_peak_memory_stats(model.device); torch.cuda.synchronize(); started = time.perf_counter()
    output = model.generate(ids, images=images, do_sample=False, num_beams=1, max_new_tokens=4, use_cache=True,
                            stopping_criteria=[YesNoStoppingCriteria(tokenizer, ids.shape[1])],
                            return_dict_in_generate=True, output_scores=capture)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    tokens = output.sequences[0, ids.shape[1]:].detach().cpu().tolist()
    raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    result = {"source_index": row["source_index"], "image": row["image"], "question": row["text"],
              "label": row["label"].lower().strip(), "prediction": pope_prediction(raw), "raw_output": raw,
              "seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(model.device))}
    if controller: result[getattr(controller, "stats_key", "pem_stats")] = controller.summary()
    if capture:
        logits = output.scores[0][0].detach().float().cpu()
        result["_logits"] = logits; result["yes_no_margin"] = margin(logits, tokenizer)
    return result


def chair_rows(limit=100, offset=0):
    historical = [json.loads(x) for x in (ROOT / "outputs/chair/chair_captions_llava15_7b_pure.jsonl").read_text().splitlines() if x.strip()]
    annotations = json.loads((ROOT / "DATA/annotations/captions_val2014.json").read_text())
    names = {int(x["id"]): x["file_name"] for x in annotations["images"]}
    return [{"image_id": int(x["image_id"]), "image_path": ROOT / "DATA/val2014" / names[int(x["image_id"])],
             "question": "Please describe this image in detail."} for x in historical[offset:offset + limit]]


def chair_prompt(question):
    conv = conv_templates["llava_v1"].copy(); conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None); return conv.get_prompt()


@torch.inference_mode()
def generate_chair(row, tokenizer, model, processor, controller=None, capture=False):
    images = prepare_image(row["image_path"], processor, model)
    ids = tokenizer_image_token(chair_prompt(row["question"]), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    if controller: controller.begin_sample()
    torch.cuda.reset_peak_memory_stats(model.device); torch.cuda.synchronize(); started = time.perf_counter()
    output = model.generate(ids, images=images, attention_mask=torch.ones_like(ids), do_sample=False, temperature=0.0,
                            top_p=None, top_k=None, num_beams=3, max_new_tokens=64, use_cache=True,
                            stopping_criteria=[KeywordsStoppingCriteria(["###"], tokenizer, ids)],
                            return_dict_in_generate=True, output_scores=capture)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    caption = tokenizer.decode(output.sequences[0, ids.shape[1]:], skip_special_tokens=True)
    if "###" in caption: caption = caption.split("###", 1)[0]
    result = {"image_id": row["image_id"], "caption": caption.strip(), "seconds": elapsed,
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(model.device))}
    if controller: result[getattr(controller, "stats_key", "pem_stats")] = controller.summary()
    if capture: result["_logits"] = output.scores[0][0].detach().float().cpu()
    return result


def add_fpr(metrics):
    result = dict(metrics); den = result["fp"] + result["tn"]
    result["false_positive_rate"] = result["fp"] / den if den else 0.0
    return result


def transitions(base, method):
    flips = fixed = broken = 0
    for a, b in zip(base, method):
        flips += a["prediction"] != b["prediction"]
        ao, bo = a["prediction"] == a["label"], b["prediction"] == b["label"]
        fixed += (not ao and bo); broken += (ao and not bo)
    return {"prediction_changes": int(flips), "incorrect_to_correct": int(fixed), "correct_to_incorrect": int(broken),
            "prediction_flip_rate": flips / len(base)}


def efficiency(rows):
    return {"total_seconds": sum(x["seconds"] for x in rows), "mean_seconds": statistics.fmean(x["seconds"] for x in rows),
            "peak_gpu_memory_bytes": max(x["peak_gpu_memory_bytes"] for x in rows)}


def aggregate_pem(rows):
    records = [record for row in rows for record in ([row.get("pem_stats", {}).get("means", {})] if row.get("pem_stats", {}).get("calls") else [])]
    keys = set().union(*(x.keys() for x in records)) if records else set()
    return {"samples": len(records), "means": {key: statistics.fmean(x[key] for x in records) for key in sorted(keys)}}


def evaluate_chair(rows, path):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps({"image_id": row["image_id"], "caption": row["caption"]}) + "\n")
    raw = compute_chair_metrics(str(path), str(ROOT / "DATA/annotations"), str(ROOT / "outputs/chair/chair_coco2014.pkl"), "")
    result = scale_chair_metrics(raw.get("overall_metrics", {})); sentences = raw.get("sentences", [])
    result["ObjectCount"] = statistics.fmean(len(x.get("mscoco_generated_words", [])) for x in sentences) if sentences else 0.0
    return result
