#!/usr/bin/env python3
"""
示例：如何从sglang中提取图片feature并计算hash

这个示例展示了如何：
1. 从sglang中获取处理后的图片feature
2. 保存为文件
3. 使用 calculate_image_hash.py 计算hash

注意：这只是一个示例，实际使用需要根据你的具体模型和processor来调整。
"""

import torch
import numpy as np
from PIL import Image

# 示例1: 如果你已经有了处理好的tensor（例如从sglang中获取的）
def example_with_tensor():
    """
    假设你从sglang中获取了图片的feature tensor
    """
    # 这里应该是从sglang中获取的feature
    # 例如: feature = mm_item.feature
    # 为了演示，我们创建一个示例tensor
    feature = torch.randn(1, 3, 224, 224)  # 示例：batch=1, channels=3, height=224, width=224
    
    # 保存为pytorch tensor文件
    torch.save(feature, 'feature.pt')
    print(f"Saved feature tensor to feature.pt")
    print(f"Shape: {feature.shape}")
    print(f"Device: {feature.device}")
    print(f"Dtype: {feature.dtype}")
    
    # 现在可以使用 calculate_image_hash.py 来计算hash:
    # python calculate_image_hash.py --tensor feature.pt


def example_with_numpy():
    """
    如果你有numpy array格式的feature
    """
    # 创建示例numpy array
    feature = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 保存为numpy文件
    np.save('feature.npy', feature)
    print(f"Saved feature array to feature.npy")
    print(f"Shape: {feature.shape}")
    print(f"Dtype: {feature.dtype}")
    
    # 现在可以使用 calculate_image_hash.py 来计算hash:
    # python calculate_image_hash.py --tensor feature.npy


def example_from_sglang_code():
    """
    示例：如何在sglang代码中提取feature并保存
    
    在你的sglang代码中，可以这样做：
    
    ```python
    from sglang.srt.managers.schedule_batch import MultimodalDataItem
    
    # 假设你有一个 MultimodalDataItem
    mm_item = ...  # 从sglang中获取
    
    # 获取feature
    feature = mm_item.feature  # 或者 mm_item.precomputed_embeddings
    
    # 保存为文件
    if isinstance(feature, torch.Tensor):
        torch.save(feature, 'feature.pt')
    elif isinstance(feature, np.ndarray):
        np.save('feature.npy', feature)
    
    # 然后使用 calculate_image_hash.py 计算hash
    ```
    """
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("示例：如何准备feature文件用于hash计算")
    print("=" * 60)
    print()
    
    print("示例1: 使用PyTorch tensor")
    example_with_tensor()
    print()
    
    print("示例2: 使用NumPy array")
    example_with_numpy()
    print()
    
    print("=" * 60)
    print("使用保存的文件计算hash:")
    print("  python calculate_image_hash.py --tensor feature.pt")
    print("  或")
    print("  python calculate_image_hash.py --tensor feature.npy")
    print("=" * 60)
