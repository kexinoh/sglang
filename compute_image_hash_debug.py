#!/usr/bin/env python3
"""
Debug version - 精确复制 SGLang 的图片处理流程
"""

import hashlib
from io import BytesIO
import os
from typing import Union

import numpy as np
import torch
from PIL import Image

# ==============================================================================
# ⚙️ 配置
# ==============================================================================

IMAGE_PATH: str = "/root/workspace/hello1.png"
MODEL_PATH: str = "/root/.cache/modelscope/models/Qwen/Qwen2___5-VL-7B-Instruct"
DEVICE: str = "cpu"  # 或 "cuda"

# ==============================================================================
# Hash functions (完全复制自 sglang/srt/managers/mm_utils.py)
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
    
    if tensor.is_cuda:
        # GPU tensor - 需要特殊处理
        # SGLang 使用 gpu_tensor_hash，但我们这里简化为移到 CPU
        tensor = tensor.cpu()
    
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
# Image loading (复制自 sglang/srt/utils/common.py)
# ==============================================================================

def load_image(image_file: Union[str, bytes]) -> Image.Image:
    import requests
    import pybase64
    
    if isinstance(image_file, Image.Image):
        return image_file
    elif isinstance(image_file, bytes):
        return Image.open(BytesIO(image_file))
    elif isinstance(image_file, str) and (image_file.startswith("http://") or image_file.startswith("https://")):
        response = requests.get(image_file, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw)
        image.load()
        return image
    elif isinstance(image_file, str) and image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        return Image.open(image_file)
    elif isinstance(image_file, str) and image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        return Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        try:
            return Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
        except Exception:
            return Image.open(image_file)
    else:
        raise ValueError(f"Invalid image: {image_file}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    from transformers import AutoProcessor, BaseImageProcessorFast
    
    print("=" * 70)
    print("SGLang Image Hash Debug Script")
    print("=" * 70)
    
    # 1. 加载图片
    print(f"\n[Step 1] Loading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    
    # SGLang 默认 discard_alpha_channel=True
    if image.mode != "RGB":
        print(f"  Converting from {image.mode} to RGB")
        image = image.convert("RGB")
    
    print(f"  Image size: {image.size}")
    print(f"  Image mode: {image.mode}")
    
    # 2. 加载 Processor
    print(f"\n[Step 2] Loading processor from: {MODEL_PATH}")
    
    # 检查是否有 fast processor
    # SGLang 默认: use_fast = not server_args.disable_fast_image_processor (即默认 True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        # use_fast=True  # 默认值，可以尝试设置为 False 对比
    )
    
    # 检查是否是 fast processor
    is_fast = hasattr(processor, "image_processor") and isinstance(processor.image_processor, BaseImageProcessorFast)
    print(f"  Processor type: {processor.__class__.__name__}")
    print(f"  Is fast image processor: {is_fast}")
    
    # 3. 处理图片
    print(f"\n[Step 3] Processing image...")
    
    # SGLang 对 Qwen2-VL 的 text 处理:
    # 实际的 input_text 是用户的完整 prompt，但对于 pixel_values 计算，
    # 关键是 images 参数
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    
    # 关键：检查 SGLang 是否在 GPU 上处理
    # SGLang: if isinstance(processor.image_processor, BaseImageProcessorFast): kwargs["device"] = "cuda"
    process_kwargs = {
        "text": [text],
        "images": [image],
        "padding": True,
        "return_tensors": "pt",
    }
    
    if is_fast and DEVICE == "cuda":
        process_kwargs["device"] = "cuda"
        print(f"  Using device=cuda for fast processor")
    
    result = processor(**process_kwargs)
    
    # 4. 获取 pixel_values 并打印详细信息
    print(f"\n[Step 4] Analyzing pixel_values...")
    
    if "pixel_values" not in result:
        print("  ERROR: No pixel_values in result!")
        print(f"  Available keys: {result.keys()}")
        return
    
    pixel_values = result["pixel_values"]
    
    # 打印详细信息（与 SGLang hook 对应）
    print("=" * 60)
    print(f"[DEBUG] pixel_values type: {type(pixel_values)}")
    print(f"[DEBUG] shape: {pixel_values.shape}")
    print(f"[DEBUG] dtype: {pixel_values.dtype}")
    print(f"[DEBUG] device: {pixel_values.device}")
    print(f"[DEBUG] is_contiguous: {pixel_values.is_contiguous()}")
    
    flat = pixel_values.flatten()[:10]
    print(f"[DEBUG] first 10 values: {flat.tolist()}")
    
    # 额外打印一些统计信息
    print(f"[DEBUG] min: {pixel_values.min().item():.6f}")
    print(f"[DEBUG] max: {pixel_values.max().item():.6f}")
    print(f"[DEBUG] mean: {pixel_values.mean().item():.6f}")
    print("=" * 60)
    
    # 5. SGLang 可能会将 tensor 移回 CPU
    # if not self.server_args.keep_mm_feature_on_device:
    #     result[feature_name] = result[feature_name].to("cpu")
    if pixel_values.is_cuda:
        print("\n[Step 5] Moving pixel_values to CPU (SGLang default behavior)...")
        pixel_values = pixel_values.cpu()
    
    # 6. 计算 hash
    print(f"\n[Step 6] Computing hash...")
    feature_hash = hash_feature(pixel_values)
    pad_value = feature_hash % (1 << 30)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Hash: {feature_hash}")
    print(f"Pad Value (hash % 2^30): {pad_value}")
    print("=" * 70)
    
    # 7. 额外测试：不同配置的 hash
    print("\n[Extra Tests] Trying different configurations...")
    
    # Test A: 直接从 numpy array 计算
    img_array = np.array(image)
    hash_from_numpy = hash_feature(img_array)
    print(f"  Hash from raw numpy array: {hash_from_numpy}")
    
    # Test B: 尝试不同的 contiguous 处理
    if not pixel_values.is_contiguous():
        pv_contig = pixel_values.contiguous()
        hash_contig = hash_feature(pv_contig)
        print(f"  Hash after explicit contiguous(): {hash_contig}")


if __name__ == "__main__":
    main()
