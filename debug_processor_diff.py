#!/usr/bin/env python3
"""
调试 Processor 差异
"""

import hashlib
from PIL import Image
import numpy as np
import torch

# ==============================================================================
# 配置
# ==============================================================================
IMAGE_PATH = "/root/workspace/hello1.png"
MODEL_PATH = "/root/.cache/modelscope/models/Qwen/Qwen2___5-VL-7B-Instruct"

def main():
    from transformers import AutoProcessor
    
    print("=" * 70)
    print("Processor 配置对比")
    print("=" * 70)
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        local_files_only=True
    )
    
    # 打印 image_processor 配置
    print("\n[1] Image Processor 配置:")
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        print(f"    type: {type(ip).__name__}")
        
        attrs = [
            'image_mean', 'image_std', 'rescale_factor', 'do_rescale',
            'do_normalize', 'min_pixels', 'max_pixels', 'size', 'resample',
            'patch_size', 'temporal_patch_size', 'merge_size'
        ]
        for attr in attrs:
            if hasattr(ip, attr):
                val = getattr(ip, attr)
                print(f"    {attr}: {val}")
    
    # 加载图片
    print(f"\n[2] 加载图片: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH)
    if image.mode != "RGB":
        image = image.convert("RGB")
    print(f"    原始尺寸: {image.size}")
    
    # 测试不同的 text 输入
    print("\n[3] 测试不同的 text 输入对 pixel_values 的影响:")
    
    texts_to_test = [
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述图片<|im_end|>",
        "Hello",
    ]
    
    for text in texts_to_test:
        try:
            result = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            if "pixel_values" in result:
                pv = result["pixel_values"]
                pv_bytes = pv.detach().contiguous().cpu().numpy().tobytes()
                sha = hashlib.sha256(pv_bytes).hexdigest()[:16]
                
                print(f"\n    text: {repr(text[:50])}...")
                print(f"    pixel_values shape: {pv.shape}")
                print(f"    pixel_values SHA256: {sha}")
                print(f"    first 5 values: {pv.flatten()[:5].tolist()}")
        except Exception as e:
            print(f"\n    text: {repr(text[:50])}...")
            print(f"    ERROR: {e}")
    
    # 测试手动设置参数
    print("\n[4] 测试手动设置 min_pixels/max_pixels:")
    
    # 检查是否支持这些参数
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    
    # 默认调用
    result_default = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    pv_default = result_default["pixel_values"]
    sha_default = hashlib.sha256(pv_default.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
    
    print(f"\n    默认参数:")
    print(f"    shape: {pv_default.shape}")
    print(f"    SHA256: {sha_default}")
    
    # 尝试传入额外参数
    try:
        # 有些 processor 支持在调用时传入这些参数
        result_custom = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            min_pixels=4 * 28 * 28,  # SGLang 默认值
            max_pixels=16384 * 28 * 28,  # SGLang 默认值
        )
        pv_custom = result_custom["pixel_values"]
        sha_custom = hashlib.sha256(pv_custom.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
        
        print(f"\n    带 min_pixels/max_pixels 参数:")
        print(f"    shape: {pv_custom.shape}")
        print(f"    SHA256: {sha_custom}")
        print(f"    SHA256 相同: {sha_default == sha_custom}")
    except Exception as e:
        print(f"\n    带参数调用失败: {e}")
    
    # 检查 processor 的 __call__ 签名
    print("\n[5] Processor __call__ 方法签名:")
    import inspect
    sig = inspect.signature(processor.__call__)
    print(f"    参数: {list(sig.parameters.keys())}")


if __name__ == "__main__":
    main()
