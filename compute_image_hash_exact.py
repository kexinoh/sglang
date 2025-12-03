#!/usr/bin/env python3
"""
精确复制 SGLang 的图片处理流程
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
# 复制 SGLang 的 get_processor 函数
# ==============================================================================

def get_processor_like_sglang(model_path: str, use_fast: bool = True):
    """
    精确复制 SGLang 的 get_processor 逻辑
    来源: python/sglang/srt/utils/hf_transformers_utils.py:457
    """
    from transformers import AutoConfig, AutoProcessor
    
    print("\n[get_processor] Loading config...")
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    print(f"[get_processor] model_type: {config.model_type}")
    
    # SGLang 的特殊处理 (hf_transformers_utils.py:480-483)
    kwargs = {}
    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}
        print(f"[get_processor] Injecting size: {kwargs['size']}")
    
    # use_fast 参数 (hf_transformers_utils.py:485-486)
    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast
        print(f"[get_processor] use_fast: {use_fast}")
    
    print(f"[get_processor] Final kwargs: {kwargs}")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        **kwargs,
    )
    
    return processor, config


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("精确复制 SGLang 图片处理流程")
    print("=" * 70)
    
    # 1. 使用 SGLang 相同的方式加载 processor
    print("\n[Step 1] 加载 Processor (SGLang 方式)")
    processor, config = get_processor_like_sglang(MODEL_PATH, use_fast=True)
    
    print(f"\nProcessor type: {type(processor).__name__}")
    
    # 2. 打印 image_processor 的所有配置
    print("\n[Step 2] Image Processor 配置:")
    print("-" * 50)
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        print(f"type: {type(ip).__name__}")
        
        # 打印所有属性
        for attr in dir(ip):
            if not attr.startswith('_') and not callable(getattr(ip, attr)):
                try:
                    val = getattr(ip, attr)
                    if not isinstance(val, (dict, list)) or len(str(val)) < 200:
                        print(f"  {attr}: {val}")
                except:
                    pass
    print("-" * 50)
    
    # 3. 加载图片
    print(f"\n[Step 3] 加载图片: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH)
    if image.mode != "RGB":
        print(f"  Converting from {image.mode} to RGB")
        image = image.convert("RGB")
    print(f"  Size: {image.size}, Mode: {image.mode}")
    
    # 4. 使用 SGLang 相同的 text
    # SGLang 中 input_text 来自 base_output.input_text
    # 对于 Qwen2-VL: image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    
    print(f"\n[Step 4] 处理图片")
    print(f"  text: {repr(text)}")
    
    # 5. 调用 processor (复制 process_mm_data 的逻辑)
    # 来源: base_processor.py:271-276
    from transformers import BaseImageProcessorFast
    
    call_kwargs = {
        "text": [text],
        "images": [image],
        "padding": True,
        "return_tensors": "pt",
    }
    
    # SGLang 可能添加 device="cuda" (base_processor.py:258-270)
    if hasattr(processor, "image_processor") and isinstance(processor.image_processor, BaseImageProcessorFast):
        # 默认情况下 SGLang 会用 device="cuda"，但我们这里用 CPU 保持一致
        # call_kwargs["device"] = "cuda"
        print(f"  Fast processor detected, but using CPU for consistency")
    
    print(f"  Calling processor with kwargs: {list(call_kwargs.keys())}")
    
    result = processor(**call_kwargs)
    
    # 6. 获取 pixel_values
    if "pixel_values" not in result:
        print("  ERROR: No pixel_values!")
        return
    
    pixel_values = result["pixel_values"]
    
    # SGLang 可能将 tensor 移到 CPU (base_processor.py:277-286)
    # keep_mm_feature_on_device 默认是 False
    pixel_values = pixel_values.cpu()
    
    # 7. 打印详细信息
    print(f"\n[Step 5] pixel_values 详细信息:")
    print("=" * 60)
    print(f"  shape: {pixel_values.shape}")
    print(f"  dtype: {pixel_values.dtype}")
    print(f"  device: {pixel_values.device}")
    print(f"  is_contiguous: {pixel_values.is_contiguous()}")
    print(f"  sum: {pixel_values.sum().item()}")
    print(f"  mean: {pixel_values.mean().item()}")
    print(f"  std: {pixel_values.std().item()}")
    print(f"  min: {pixel_values.min().item()}")
    print(f"  max: {pixel_values.max().item()}")
    print(f"  first 10: {pixel_values.flatten()[:10].tolist()}")
    print(f"  last 10: {pixel_values.flatten()[-10:].tolist()}")
    
    # SHA256
    tensor_bytes = pixel_values.detach().contiguous().cpu().numpy().tobytes()
    sha = hashlib.sha256(tensor_bytes).hexdigest()[:16]
    print(f"  direct_sha: {sha}")
    print("=" * 60)
    
    # 8. 计算 hash
    print(f"\n[Step 6] 计算 hash")
    feature_hash = hash_feature(pixel_values)
    pad_value = feature_hash % (1 << 30)
    
    print(f"  Hash: {feature_hash}")
    print(f"  Pad Value: {pad_value}")
    
    # 9. 额外测试：尝试不同的 text
    print("\n[Extra] 测试不同的 text 是否影响 pixel_values:")
    
    test_texts = [
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n描述这张图片<|im_end|>",
    ]
    
    for t in test_texts:
        try:
            r = processor(
                text=[t],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            pv = r["pixel_values"].cpu()
            tb = pv.detach().contiguous().numpy().tobytes()
            s = hashlib.sha256(tb).hexdigest()[:16]
            print(f"  text={repr(t[:40])}... -> sha={s}, shape={pv.shape}")
        except Exception as e:
            print(f"  text={repr(t[:40])}... -> ERROR: {e}")


if __name__ == "__main__":
    main()
