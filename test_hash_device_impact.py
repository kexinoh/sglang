#!/usr/bin/env python3
"""
测试脚本：验证 device 参数对 hash 值的影响

这个脚本演示了为什么 processor 中的 device="cuda" 会影响 hash 值的计算。
"""

import torch
import numpy as np

# 导入hash计算函数
try:
    from sglang.srt.managers.mm_utils import hash_feature, tensor_hash
    from sglang.srt.layers.multimodal import gpu_tensor_hash
except ImportError:
    print("Warning: Could not import sglang modules. This script should be run in the sglang environment.")
    exit(1)


def test_hash_consistency():
    """测试相同数据在不同device下的hash一致性"""
    print("=" * 60)
    print("测试1: 相同数据在CPU和CUDA上的hash值")
    print("=" * 60)
    
    # 创建测试数据
    data = torch.randn(3, 224, 224, dtype=torch.float32)
    
    # CPU版本
    cpu_tensor = data.clone().cpu()
    cpu_hash = hash_feature(cpu_tensor)
    
    # CUDA版本（如果可用）
    if torch.cuda.is_available():
        cuda_tensor = data.clone().cuda()
        cuda_hash = hash_feature(cuda_tensor)
        
        print(f"CPU hash:  {cpu_hash}")
        print(f"CUDA hash: {cuda_hash}")
        print(f"Hash值是否相等: {cpu_hash == cuda_hash}")
        
        if cpu_hash != cuda_hash:
            print("\n❌ Hash值不一致！这证实了device参数会影响hash值。")
            print("\n原因分析：")
            print("1. CPU路径使用SHA256 hash，只依赖数据内容")
            print("2. CUDA路径使用位置相关的GPU hash算法")
            print("3. GPU hash算法将元素索引纳入计算，导致不同内存布局产生不同hash")
        else:
            print("\n✅ Hash值一致（这可能是巧合，取决于数据的具体值）")
    else:
        print("CUDA不可用，跳过CUDA测试")


def test_tensor_hash_directly():
    """直接测试tensor_hash函数"""
    print("\n" + "=" * 60)
    print("测试2: 直接测试tensor_hash函数")
    print("=" * 60)
    
    data = torch.randn(10, 10, dtype=torch.float32)
    
    # CPU tensor
    cpu_tensor = data.clone().cpu()
    cpu_hash = tensor_hash(cpu_tensor)
    print(f"CPU tensor hash: {cpu_hash}")
    
    # CUDA tensor
    if torch.cuda.is_available():
        cuda_tensor = data.clone().cuda()
        cuda_hash = tensor_hash(cuda_tensor)
        print(f"CUDA tensor hash: {cuda_hash}")
        print(f"Hash值是否相等: {cpu_hash == cuda_hash}")


def test_gpu_hash_position_sensitivity():
    """测试GPU hash的位置敏感性"""
    print("\n" + "=" * 60)
    print("测试3: GPU hash的位置敏感性")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过此测试")
        return
    
    # 创建相同的数据，但使用不同的内存布局
    data1 = torch.randn(100, dtype=torch.float32).cuda()
    data2 = data1.clone().cuda()  # 相同数据，但可能是不同的内存位置
    
    hash1 = gpu_tensor_hash(data1)
    hash2 = gpu_tensor_hash(data2)
    
    print(f"相同数据的hash1: {hash1}")
    print(f"相同数据的hash2: {hash2}")
    print(f"Hash值是否相等: {hash1 == hash2}")
    
    # 测试contiguous的影响
    data3 = data1[::2].contiguous()  # 非连续tensor转为连续
    hash3 = gpu_tensor_hash(data3)
    print(f"\n非连续tensor的hash: {hash3}")
    print(f"与原始hash是否相等: {hash1 == hash3}")


def test_padding_impact():
    """测试padding对hash的影响"""
    print("\n" + "=" * 60)
    print("测试4: Padding对hash的影响")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过此测试")
        return
    
    from sglang.srt.layers.multimodal import _as_uint32_words
    
    # 创建不同大小的tensor（可能导致不同的padding）
    sizes = [100, 101, 102, 103]  # 不同的字节数，可能导致不同的padding
    
    print("不同大小的tensor在转换为uint32时的padding情况：")
    for size in sizes:
        tensor = torch.randn(size, dtype=torch.float32).cuda()
        u32 = _as_uint32_words(tensor)
        original_bytes = tensor.numel() * tensor.element_size()
        padded_bytes = u32.numel() * 4
        padding = padded_bytes - original_bytes
        print(f"  Size {size}: {original_bytes} bytes -> {padded_bytes} bytes (padding: {padding} bytes)")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Hash值计算中device参数影响的分析测试")
    print("=" * 60)
    print("\n这个测试脚本验证了为什么processor中的device='cuda'会影响hash值。")
    print("主要原因是CPU和CUDA使用了不同的hash算法。\n")
    
    test_hash_consistency()
    test_tensor_hash_directly()
    test_gpu_hash_position_sensitivity()
    test_padding_impact()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n建议：")
    print("1. 在计算hash之前，统一将tensor转换为CPU")
    print("2. 这样可以确保相同的数据产生相同的hash值")
    print("3. 详见 analysis_device_impact_on_hash.md 文件")


if __name__ == "__main__":
    main()
