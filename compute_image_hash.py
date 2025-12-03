#!/usr/bin/env python3
"""
Standalone script to compute the same image hash as SGLang does.

This script replicates the hash computation in:
  sglang/srt/managers/schedule_batch.py#L244 (set_pad_value)
  sglang/srt/managers/mm_utils.py (hash_feature, tensor_hash, data_hash)

Usage:
  python compute_image_hash.py <image_path> [--model <model_name>]

Example:
  python compute_image_hash.py /path/to/image.jpg --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import hashlib
from io import BytesIO
from typing import Union

import numpy as np
import torch
from PIL import Image


# ==============================================================================
# Hash functions (from sglang/srt/managers/mm_utils.py)
# ==============================================================================

def data_hash(data) -> int:
    """Compute hash from bytes data using SHA256."""
    if isinstance(data, tuple):
        # Handle tuple case - convert to bytes
        data = str(data).encode()
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def flatten_nested_list(nested_list):
    """Flatten a nested list structure."""
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


def tensor_hash(tensor_list) -> int:
    """Hash a tensor or a tensor list (CPU tensors only)."""
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    
    tensor = tensor.detach().contiguous()

    if tensor.dtype == torch.bfloat16:
        # memoryview() doesn't support PyTorch's BFloat16 dtype
        tensor = tensor.float()

    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()

    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def hash_feature(f) -> int:
    """
    Hash feature data. This is the entry point used by SGLang.
    Supports: list of tensors, np.ndarray, single tensor.
    """
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        arr_bytes = arr.tobytes()
        return data_hash(arr_bytes)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    return data_hash(f)


# ==============================================================================
# Image loading (from sglang/srt/utils/common.py)
# ==============================================================================

def load_image(image_file: Union[str, bytes]) -> Image.Image:
    """Load an image from file path, URL, or base64 string."""
    import requests
    import pybase64
    
    if isinstance(image_file, Image.Image):
        return image_file
    elif isinstance(image_file, bytes):
        return Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw)
        image.load()
        return image
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        return Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        return Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        return Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image_file}")


# ==============================================================================
# Main processing
# ==============================================================================

def compute_image_hash_with_processor(
    image_path: str,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cpu"
) -> int:
    """
    Compute the hash of an image using a HuggingFace processor.
    
    This replicates the exact processing done by SGLang when computing the
    pad_value hash.
    
    Args:
        image_path: Path to the image file, URL, or base64 string
        model_name: HuggingFace model name (for loading the processor)
        device: Device for processing ("cpu" or "cuda")
    
    Returns:
        The hash value (same as SGLang's hash computation)
    """
    from transformers import AutoProcessor
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Convert to RGB if needed (SGLang does this)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    print(f"Image size: {image.size}, mode: {image.mode}")
    
    # Load the processor
    print(f"Loading processor for: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Process the image
    # SGLang uses a placeholder text with image token
    # The exact text varies by model, but we need something that includes an image token
    if "qwen" in model_name.lower():
        # For Qwen2-VL models
        text = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        # Generic placeholder - adjust based on your model
        text = "<image>"
    
    print(f"Processing image with text: {text}")
    
    # Call the processor
    result = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    
    # Get pixel_values - this is what SGLang hashes
    if "pixel_values" in result:
        pixel_values = result["pixel_values"]
        print(f"pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
        
        # Move to specified device if needed
        if device != "cpu":
            pixel_values = pixel_values.to(device)
        else:
            # Ensure on CPU for hashing
            pixel_values = pixel_values.cpu()
        
        # Compute hash (same as SGLang)
        feature_hash = hash_feature(pixel_values)
        pad_value = feature_hash % (1 << 30)
        
        print(f"\n=== Results ===")
        print(f"Hash: {feature_hash}")
        print(f"Pad Value (hash % 2^30): {pad_value}")
        
        return feature_hash
    else:
        print("No pixel_values in processor output!")
        print(f"Available keys: {result.keys()}")
        return None


def compute_image_hash_raw(image_path: str) -> int:
    """
    Compute hash directly from raw image bytes (without processor).
    This is useful if you want to compute a hash without model-specific processing.
    """
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    image_array = np.array(image)
    print(f"Image array shape: {image_array.shape}, dtype: {image_array.dtype}")
    
    # Compute hash
    feature_hash = hash_feature(image_array)
    pad_value = feature_hash % (1 << 30)
    
    print(f"\n=== Results (raw) ===")
    print(f"Hash: {feature_hash}")
    print(f"Pad Value (hash % 2^30): {pad_value}")
    
    return feature_hash


def main():
    parser = argparse.ArgumentParser(
        description="Compute image hash using SGLang's algorithm"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file, URL, or base64 string"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HuggingFace model name for processor (default: Qwen/Qwen2-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Compute hash from raw image without processor"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing (default: cpu)"
    )
    
    args = parser.parse_args()
    
    if args.raw:
        compute_image_hash_raw(args.image_path)
    else:
        compute_image_hash_with_processor(
            args.image_path,
            model_name=args.model,
            device=args.device
        )


if __name__ == "__main__":
    main()
