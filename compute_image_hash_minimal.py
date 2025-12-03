#!/usr/bin/env python3
"""
Minimal script to compute image hash matching SGLang's algorithm.

This extracts just the core hash logic from:
  sglang/srt/managers/mm_utils.py (hash_feature, tensor_hash, data_hash)

If you already have the pixel_values tensor from the processor,
you can use this directly.
"""

import hashlib
import numpy as np
import torch


def data_hash(data) -> int:
    """Compute hash from bytes using SHA256 (first 8 bytes as int)."""
    if isinstance(data, tuple):
        data = str(data).encode()
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def flatten_nested_list(nested_list):
    """Flatten nested list structure."""
    if isinstance(nested_list, list):
        return [item for sublist in nested_list for item in flatten_nested_list(sublist)]
    return [nested_list]


def tensor_hash(tensor_list) -> int:
    """Hash a tensor or list of tensors."""
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list]
        tensor = torch.concat(tensor_list)
    
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    tensor_cpu = tensor.cpu()
    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def hash_feature(f) -> int:
    """
    Hash feature data - this is SGLang's main entry point.
    
    Args:
        f: Can be torch.Tensor, list of tensors, or np.ndarray
        
    Returns:
        int: The hash value
    """
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        return data_hash(arr.tobytes())
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    return data_hash(f)


# ===== Example Usage =====
if __name__ == "__main__":
    # Example 1: Hash a torch tensor (simulating pixel_values)
    print("=== Example with random tensor ===")
    fake_pixel_values = torch.randn(1, 3, 224, 224)
    h = hash_feature(fake_pixel_values)
    pad_value = h % (1 << 30)
    print(f"Hash: {h}")
    print(f"Pad Value: {pad_value}")
    
    # Example 2: Hash a numpy array
    print("\n=== Example with numpy array ===")
    fake_array = np.random.rand(3, 224, 224).astype(np.float32)
    h2 = hash_feature(fake_array)
    pad_value2 = h2 % (1 << 30)
    print(f"Hash: {h2}")
    print(f"Pad Value: {pad_value2}")
    
    # Example 3: Hash from actual image processing
    print("\n=== To use with actual image ===")
    print("""
    # Load your image with PIL
    from PIL import Image
    image = Image.open("your_image.jpg").convert("RGB")
    
    # Process with your model's processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    result = processor(text=["<image>"], images=[image], return_tensors="pt")
    
    # Get pixel_values and compute hash
    pixel_values = result["pixel_values"]
    feature_hash = hash_feature(pixel_values)
    pad_value = feature_hash % (1 << 30)
    print(f"Hash: {feature_hash}, Pad Value: {pad_value}")
    """)
