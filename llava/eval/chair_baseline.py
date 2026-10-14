#!/usr/bin/env python3
"""Generate CHAIR captions with the original LLaVA model only.

The dataset sampling, prompt, and decoding settings match the current
HulluEdit CHAIR run.  This file has no dependency on ``hulluedit`` and calls
the original LLaVA ``model.generate`` directly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original LLaVA-1.5 CHAIR caption generation (no HulluEdit)"
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "llava-v1.5-7b"),
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "DATA"),
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--sampling", choices=("first", "random"), default="random")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument(
        "--image-ids",
        default=None,
        help="Optional comma-separated COCO image ids; overrides sampling",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(args: argparse.Namespace) -> List[Dict]:
    data_root = Path(args.data_root).expanduser().resolve()
    annotation_file = data_root / "annotations" / f"captions_{args.split}2014.json"
    image_root = data_root / f"{args.split}2014"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO caption annotation not found: {annotation_file}")

    coco = COCO(str(annotation_file))
    all_image_ids = coco.getImgIds()

    if args.image_ids:
        image_ids = [int(value.strip()) for value in args.image_ids.split(",") if value.strip()]
        missing = sorted(set(image_ids) - set(all_image_ids))
        if missing:
            raise ValueError(f"Requested COCO image ids not found: {missing}")
    elif args.num_samples is None or args.num_samples >= len(all_image_ids):
        image_ids = all_image_ids
    elif args.sampling == "first":
        image_ids = all_image_ids[: args.num_samples]
    else:
        # setup_seeds() sets the same Python RNG used by the HulluEdit dataset.
        image_ids = random.sample(all_image_ids, args.num_samples)

    rows: List[Dict] = []
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        rows.append(
            {
                "image_id": int(image_id),
                "image_path": image_root / image_info["file_name"],
                "question": "Please describe this image in detail.",
            }
        )
    return rows


def build_prompt(question: str) -> str:
    """Match generate_chair_captions.py's LLaVA prompt exactly."""
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def generate_caption(
    *,
    model,
    tokenizer,
    image_processor,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [item.to(model.device, dtype=torch.float16) for item in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt = build_prompt(question)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        attention_mask=torch.ones_like(input_ids, device=model.device),
        do_sample=False,
        temperature=0.0,
        top_p=None,
        top_k=None,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[KeywordsStoppingCriteria(["###"], tokenizer, input_ids)],
    )

    generated_ids = output_ids[0, input_ids.shape[1] :]
    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "###" in caption:
        caption = caption.split("###", 1)[0]
    return caption.strip()


def load_processed_ids(output_file: Path) -> set[int]:
    processed: set[int] = set()
    if not output_file.is_file():
        return processed
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            processed.add(int(json.loads(line)["image_id"]))
    return processed


def main() -> None:
    args = parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for LLaVA-1.5-7B")

    setup_seeds(args.seed)
    dataset = build_dataset(args)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(output_file)

    print("[CHAIR baseline] Original LLaVA model; no hidden-state editing")
    print(
        f"[CHAIR baseline] images={len(dataset)} seed={args.seed} "
        f"beams={args.num_beams} max_new_tokens={args.max_new_tokens}"
    )
    if processed_ids:
        print(f"[CHAIR baseline] Resuming: {len(processed_ids)} captions already exist")

    model_path = Path(args.model_path).expanduser().resolve()
    disable_torch_init()
    model_name = get_model_name_from_path(str(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        None,
        model_name,
        device="cuda",
    )
    model.eval()

    mode = "a" if processed_ids else "w"
    written = len(processed_ids)
    with output_file.open(mode, encoding="utf-8") as handle:
        for item in tqdm(dataset, desc="Generating CHAIR captions"):
            image_id = item["image_id"]
            if image_id in processed_ids:
                continue
            image_path = item["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"COCO image not found: {image_path}")

            caption = generate_caption(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=image_path,
                question=item["question"],
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            handle.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
            handle.flush()
            written += 1
            if args.verbose:
                tqdm.write(f"{image_id}: {caption}")

    if written != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} captions, but output contains {written}")
    print(f"[CHAIR baseline] Captions saved: {output_file}")


if __name__ == "__main__":
    main()
