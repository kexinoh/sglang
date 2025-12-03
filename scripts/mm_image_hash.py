#!/usr/bin/env python3
"""Compute the multimodal hash/pad value exactly like `MultimodalDataItem.set_pad_value`.

The script mirrors the path that `sglang` uses inside `ScheduleBatch`:
  1. Run the HuggingFace processor for the chosen multimodal model to obtain
     tensors such as `pixel_values`.
  2. Wrap the processor outputs in `MultimodalDataItem` objects.
  3. Call `MultimodalDataItem.set_pad_value()`, which internally invokes
     `hash_feature()` and applies the modulo that Scheduler uses for padding.

Example:
    python scripts/mm_image_hash.py /path/to/image.jpg \
        --model-path Qwen/Qwen2.5-VL-7B-Instruct --trust-remote-code
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

# The CUDA-IPC transport pool is unnecessary for this offline script and can
# easily run out of GPU memory on developer machines. Disable it unless the
# caller explicitly overrides the env var before running the script.
os.environ.setdefault("SGLANG_USE_CUDA_IPC_TRANSPORT", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
for candidate in (PYTHON_DIR, REPO_ROOT):
    if str(candidate) not in sys.path and candidate.exists():
        sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="+",
        help="Path/URL/base64 string of the image(s) to hash.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model identifier or local path matching your server model.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional text prompt passed to the processor (default: empty string).",
    )
    parser.add_argument(
        "--transport-mode",
        default="default",
        choices=["default", "cuda_ipc", "auto"],
        help="Match the transport mode used in your server (default: default).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to transformers when loading the model/processor.",
    )
    parser.add_argument(
        "--show-feature-meta",
        action="store_true",
        help="Print dtype/shape information for each hashed feature.",
    )
    return parser.parse_args()


def _load_images(image_specs: List[str]) -> List[Any]:
    images: List[Any] = []
    from sglang.srt.utils.common import load_image

    for spec in image_specs:
        image, _ = load_image(spec)
        if image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)
    return images


def _describe_feature(feature) -> str:
    try:
        import torch

        if isinstance(feature, torch.Tensor):
            return f"dtype={feature.dtype}, shape={tuple(feature.shape)}"
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(feature, np.ndarray):
            return f"dtype={feature.dtype}, shape={feature.shape}"
    except Exception:
        pass

    return f"type={type(feature)}"


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "This script requires the `transformers` package. "
            "Install it via `pip install transformers` and try again."
        ) from exc

    try:
        from sglang.srt.managers.multimodal_processor import (
            import_processors,
            get_mm_processor,
        )
        from sglang.srt.server_args import ServerArgs
    except ImportError as exc:
        raise SystemExit(
            "Unable to import sglang. Run this script from the repository root "
            "or ensure that /path/to/python is on PYTHONPATH."
        ) from exc

    hf_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    hf_processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        enable_multimodal=True,
        keep_mm_feature_on_device=False,
    )

    import_processors("sglang.srt.multimodal.processors")

    mm_processor = get_mm_processor(
        hf_config=hf_config,
        server_args=server_args,
        processor=hf_processor,
        transport_mode=args.transport_mode,
    )

    images = _load_images(args.image)

    processor_output = mm_processor.process_mm_data(
        input_text=args.prompt,
        images=images,
    )

    mm_items = mm_processor.collect_mm_items_from_processor_output(processor_output)
    if not mm_items:
        raise RuntimeError(
            "Processor did not produce any multimodal items. "
            "Ensure the selected model supports image inputs and that the prompt "
            "contains the proper image token if required."
        )

    for idx, item in enumerate(mm_items):
        item.set_pad_value()
        feature_meta = _describe_feature(item.feature or item.precomputed_embeddings)
        print(f"[Item {idx}] modality={item.modality.name}")
        if args.show_feature_meta:
            print(f"  feature: {feature_meta}")
        print(f"  hash: {item.hash} (0x{item.hash:016x})")
        print(f"  pad_value: {item.pad_value}")


if __name__ == "__main__":
    main()
