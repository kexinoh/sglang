#!/usr/bin/env python3
"""
SGLang 图片 Hash 计算 - 支持 GPU 和 CPU 两种算法
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

IMAGE_PATH: str = "/root/workspace/hello1.png"
MODEL_PATH: str = "/root/.cache/modelscope/models/Qwen/Qwen2___5-VL-7B-Instruct"

# 关键配置：选择使用哪种 hash 算法
# 请根据 SGLang hook 打印的 is_cuda 值来设置
USE_GPU_HASH: bool = False  # True = 使用 gpu_tensor_hash, False = 使用 SHA256

# 如果 USE_GPU_HASH=True，需要确保有 GPU 可用
DEVICE: str = "cuda" if USE_GPU_HASH else "cpu"

# ==============================================================================
# CPU Hash functions (SHA256)
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


def cpu_tensor_hash(tensor_list) -> int:
    """CPU 版本的 tensor hash (SHA256)"""
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


# ==============================================================================
# GPU Hash function (Triton kernel - 需要从 SGLang 导入)
# ==============================================================================

def gpu_tensor_hash_wrapper(tensor: torch.Tensor) -> int:
    """GPU 版本的 tensor hash (使用 SGLang 的 Triton kernel)"""
    try:
        from sglang.srt.layers.multimodal import gpu_tensor_hash
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        return gpu_tensor_hash(tensor)
    except ImportError:
        print("Warning: Cannot import gpu_tensor_hash from sglang.")
        print("Falling back to CPU hash after moving tensor to CPU.")
        return cpu_tensor_hash([tensor.cpu()])


# ==============================================================================
# Unified hash_feature function
# ==============================================================================

def hash_feature(f, use_gpu: bool = False) -> int:
    """
    Hash feature data - 统一入口
    
    Args:
        f: 要 hash 的数据 (tensor, list of tensors, ndarray)
        use_gpu: 是否使用 GPU hash
    """
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            if use_gpu or f[0].is_cuda:
                # 合并所有 tensor
                flat_list = flatten_nested_list(f)
                flat_list = [x.flatten() for x in flat_list if isinstance(x, torch.Tensor)]
                combined = torch.concat(flat_list)
                return gpu_tensor_hash_wrapper(combined)
            return cpu_tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        return data_hash(arr.tobytes())
    elif isinstance(f, torch.Tensor):
        if use_gpu or f.is_cuda:
            return gpu_tensor_hash_wrapper(f)
        return cpu_tensor_hash([f])
    return data_hash(f)


# ==============================================================================
# Image loading
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
    print("SGLang Image Hash Calculator v2")
    print(f"Hash Algorithm: {'GPU (Triton)' if USE_GPU_HASH else 'CPU (SHA256)'}")
    print("=" * 70)
    
    # 1. 加载图片
    print(f"\n[1] Loading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    if image.mode != "RGB":
        image = image.convert("RGB")
    print(f"    Size: {image.size}, Mode: {image.mode}")
    
    # 2. 加载 Processor
    print(f"\n[2] Loading processor: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    is_fast = hasattr(processor, "image_processor") and isinstance(processor.image_processor, BaseImageProcessorFast)
    print(f"    Processor: {processor.__class__.__name__}")
    print(f"    Fast processor: {is_fast}")
    
    # 3. 处理图片
    print(f"\n[3] Processing image...")
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    
    process_kwargs = {
        "text": [text],
        "images": [image],
        "padding": True,
        "return_tensors": "pt",
    }
    
    # 如果使用 GPU hash 且有 fast processor，在 GPU 上处理
    if USE_GPU_HASH and is_fast and torch.cuda.is_available():
        process_kwargs["device"] = "cuda"
        print("    Processing on: CUDA")
    else:
        print("    Processing on: CPU")
    
    result = processor(**process_kwargs)
    
    if "pixel_values" not in result:
        print("    ERROR: No pixel_values!")
        return
    
    pixel_values = result["pixel_values"]
    
    # 4. 打印详细信息（用于对比）
    print("\n[4] pixel_values info:")
    print("=" * 60)
    print(f"    shape: {pixel_values.shape}")
    print(f"    dtype: {pixel_values.dtype}")
    print(f"    device: {pixel_values.device}")
    print(f"    is_cuda: {pixel_values.is_cuda}")
    print(f"    is_contiguous: {pixel_values.is_contiguous()}")
    
    flat = pixel_values.flatten()[:10]
    if pixel_values.is_cuda:
        flat = flat.cpu()
    print(f"    first 10 values: {flat.tolist()}")
    print("=" * 60)
    
    # 5. 计算 hash
    print(f"\n[5] Computing hash (use_gpu={USE_GPU_HASH})...")
    
    # 根据配置决定使用哪种 hash
    if USE_GPU_HASH:
        if not pixel_values.is_cuda:
            print("    Moving tensor to CUDA...")
            pixel_values = pixel_values.cuda()
        feature_hash = hash_feature(pixel_values, use_gpu=True)
    else:
        if pixel_values.is_cuda:
            print("    Moving tensor to CPU...")
            pixel_values = pixel_values.cpu()
        feature_hash = hash_feature(pixel_values, use_gpu=False)
    
    pad_value = feature_hash % (1 << 30)
    
    # 6. 结果
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Hash: {feature_hash}")
    print(f"Pad Value (hash % 2^30): {pad_value}")
    print("=" * 70)
    
    # 7. 同时计算另一种 hash 用于对比
    print("\n[Extra] Computing hash with the other algorithm for comparison...")
    if USE_GPU_HASH:
        # 也计算 CPU hash
        pv_cpu = result["pixel_values"].cpu()
        cpu_hash = hash_feature(pv_cpu, use_gpu=False)
        print(f"    CPU hash (SHA256): {cpu_hash}")
        print(f"    CPU pad_value: {cpu_hash % (1 << 30)}")
    else:
        # 也计算 GPU hash (如果可用)
        if torch.cuda.is_available():
            pv_gpu = result["pixel_values"].cuda()
            gpu_hash = hash_feature(pv_gpu, use_gpu=True)
            print(f"    GPU hash (Triton): {gpu_hash}")
            print(f"    GPU pad_value: {gpu_hash % (1 << 30)}")
        else:
            print("    GPU not available, skipping GPU hash")


if __name__ == "__main__":
    main()
