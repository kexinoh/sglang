#!/usr/bin/env python3
"""
精确复制 SGLang 的图片处理流程 - 最终版本
关键：使用 device="cuda" 在 GPU 上处理图片
"""

import hashlib
from io import BytesIO
import os
from typing import Union

import numpy as np
import torch
from PIL import Image

# ==============================================================================
# 配置
# ==============================================================================
IMAGE_PATH = "/root/workspace/hello1.png"
MODEL_PATH = "/root/.cache/modelscope/models/Qwen/Qwen2___5-VL-7B-Instruct"

# ==============================================================================
# Hash 函数 (完全复制自 SGLang)
# ==============================================================================

def data_hash(data) -> int:
    if isinstance(data, tuple):
        data = str(data).encode()
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [item for sublist in nested_list for item in flatten_nested_list(sublist)]
    return [nested_list]


def tensor_hash(tensor_list) -> int:
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


# ==============================================================================
# Main
# ==============================================================================

def main():
    from transformers import AutoProcessor
    
    print("=" * 70)
    print("精确复制 SGLang 图片处理流程 (GPU 版本)")
    print("=" * 70)
    
    # 1. 加载 Processor
    print(f"\n[Step 1] 加载 Processor: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"  Processor type: {type(processor).__name__}")
    
    # 2. 加载图片
    print(f"\n[Step 2] 加载图片: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH)
    if image.mode != "RGB":
        print(f"  Converting from {image.mode} to RGB")
        image = image.convert("RGB")
    print(f"  Size: {image.size}, Mode: {image.mode}")
    
    # 3. 处理图片 - 关键：使用 device="cuda"！
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    
    print(f"\n[Step 3] 处理图片 (device='cuda')")
    print(f"  text: {repr(text)}")
    
    # SGLang 的处理方式 (base_processor.py:258-276)
    result = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
        device="cuda",  # ← 关键！SGLang 使用 GPU 处理
    )
    
    if "pixel_values" not in result:
        print("  ERROR: No pixel_values!")
        return
    
    pixel_values = result["pixel_values"]
    print(f"  pixel_values device after processing: {pixel_values.device}")
    
    # SGLang 会将 tensor 移到 CPU (keep_mm_feature_on_device 默认为 False)
    pixel_values = pixel_values.cpu()
    print(f"  pixel_values device after .cpu(): {pixel_values.device}")
    
    # 4. 打印详细信息
    print(f"\n[Step 4] pixel_values 详细信息:")
    print("=" * 60)
    print(f"  shape: {pixel_values.shape}")
    print(f"  dtype: {pixel_values.dtype}")
    print(f"  device: {pixel_values.device}")
    print(f"  is_contiguous: {pixel_values.is_contiguous()}")
    print(f"  sum: {pixel_values.sum().item()}")
    print(f"  mean: {pixel_values.mean().item()}")
    print(f"  first 10: {pixel_values.flatten()[:10].tolist()}")
    print(f"  last 10: {pixel_values.flatten()[-10:].tolist()}")
    
    # SHA256
    tensor_bytes = pixel_values.detach().contiguous().cpu().numpy().tobytes()
    sha = hashlib.sha256(tensor_bytes).hexdigest()[:16]
    print(f"  direct_sha: {sha}")
    print("=" * 60)
    
    # 5. 计算 hash
    print(f"\n[Step 5] 计算 hash")
    feature_hash = hash_feature(pixel_values)
    pad_value = feature_hash % (1 << 30)
    
    print(f"  Hash: {feature_hash}")
    print(f"  Pad Value: {pad_value}")
    
    # 6. 对比 CPU 处理
    print(f"\n[对比] CPU 处理的结果:")
    result_cpu = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
        # 不传 device，使用 CPU
    )
    pv_cpu = result_cpu["pixel_values"].cpu()
    sha_cpu = hashlib.sha256(pv_cpu.detach().contiguous().numpy().tobytes()).hexdigest()[:16]
    hash_cpu = hash_feature(pv_cpu)
    
    print(f"  CPU direct_sha: {sha_cpu}")
    print(f"  CPU Hash: {hash_cpu}")
    print(f"  CPU Pad Value: {hash_cpu % (1 << 30)}")
    
    print(f"\n[结论]")
    print(f"  GPU sha: {sha}")
    print(f"  CPU sha: {sha_cpu}")
    print(f"  SHA 相同: {sha == sha_cpu}")


if __name__ == "__main__":
    main()
