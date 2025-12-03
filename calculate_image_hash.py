#!/usr/bin/env python3
"""
独立的Python脚本，用于计算图片的hash值。
这个hash值与 sglang/srt/managers/schedule_batch.py#L244 处计算的hash值相同。

使用方法:
    # 从图片路径计算hash
    python calculate_image_hash.py <image_path>
    
    # 从numpy array文件计算hash
    python calculate_image_hash.py --tensor <tensor_file.npy>
    
    # 从pytorch tensor文件计算hash
    python calculate_image_hash.py --tensor <tensor_file.pt>
    
    # 如果tensor在CUDA上，使用--cuda选项（需要sglang支持）
    python calculate_image_hash.py --tensor <tensor_file.pt> --cuda

注意:
    - 如果输入是图片路径，脚本会使用PIL加载图片并转换为tensor。
      但是，为了得到与sglang完全相同的hash值，你应该使用与sglang相同的processor
      来处理图片，然后将处理后的tensor保存为文件，再使用--tensor选项。
    
    - 对于CUDA tensor，如果sglang可用，会使用GPU加速的hash计算。
      如果sglang不可用，会自动将tensor移到CPU再计算（结果可能略有不同）。
    
示例:
    # 1. 如果你有处理好的tensor文件
    python calculate_image_hash.py --tensor feature.pt
    
    # 2. 如果你有numpy array文件
    python calculate_image_hash.py --tensor feature.npy
    
    # 3. 直接从图片计算（注意：可能与sglang的processor结果不同）
    python calculate_image_hash.py image.jpg
"""

import argparse
import hashlib
import sys
from typing import Any, List, Union

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install it with: pip install numpy")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is required. Install it with: pip install torch")
    sys.exit(1)

try:
    from sglang.srt.layers.multimodal import gpu_tensor_hash
    from sglang.srt.utils import flatten_nested_list
    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    # 简单的flatten实现
    def flatten_nested_list(l):
        result = []
        for item in l:
            if isinstance(item, list):
                result.extend(flatten_nested_list(item))
            else:
                result.append(item)
        return result
    print("Warning: sglang not available, GPU tensor hashing will use fallback method")


def data_hash(data: bytes) -> int:
    """计算数据的SHA256 hash，返回前8字节作为整数"""
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list: Union[torch.Tensor, List[torch.Tensor]]) -> int:
    """
    hash一个tensor或tensor列表
    这与 sglang/srt/managers/mm_utils.py 中的 tensor_hash 函数相同
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        # 展平嵌套列表
        tensor_list = flatten_nested_list(tensor_list)
        
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    
    if tensor.is_cuda:
        if HAS_SGLANG:
            return gpu_tensor_hash(tensor.cuda())
        else:
            # Fallback: 将CUDA tensor移到CPU
            # 注意：这可能导致hash值与sglang中的不完全相同
            # 因为GPU tensor hash使用了特殊的CUDA kernel
            print("Warning: GPU tensor detected but sglang not available.")
            print("Moving tensor to CPU for hashing. The hash value may differ from sglang's GPU hash.")
            tensor = tensor.cpu()
    
    tensor = tensor.detach().contiguous()
    
    if tensor.dtype == torch.bfloat16:
        # memoryview() 不支持 PyTorch 的 BFloat16 dtype
        tensor = tensor.float()
    
    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()
    
    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def hash_feature(f: Union[torch.Tensor, np.ndarray, List, bytes]) -> int:
    """
    计算特征的hash值
    这与 sglang/srt/managers/mm_utils.py 中的 hash_feature 函数相同
    """
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        # 展平嵌套列表
        flattened = flatten_nested_list(f)
        return data_hash(tuple(flattened))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        arr_bytes = arr.tobytes()
        return data_hash(arr_bytes)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, bytes):
        return data_hash(f)
    else:
        # 尝试转换为bytes
        try:
            return data_hash(bytes(f))
        except TypeError:
            raise TypeError(f"Unsupported type for hashing: {type(f)}")


def load_image_as_tensor(image_path: str, use_processor: bool = False):
    """
    加载图片并转换为tensor
    
    注意: 这个方法只是示例。实际使用中，你需要使用与sglang相同的processor
    来处理图片，以确保得到的tensor与sglang中使用的完全一致。
    
    如果 use_processor=True，会尝试使用sglang的processor（如果可用）
    """
    if use_processor:
        try:
            # 尝试使用sglang的processor
            # 这需要根据你使用的具体模型来调整
            from PIL import Image
            from transformers import AutoProcessor
            
            # 这里需要根据你的模型来设置processor
            # 例如: processor = AutoProcessor.from_pretrained("your-model-name")
            print("Error: Please specify the processor/model name")
            print("For now, using basic image loading...")
            use_processor = False
        except ImportError:
            print("Warning: transformers not available, using basic image loading")
            use_processor = False
    
    if not use_processor:
        # 基本加载：使用PIL加载图片并转换为numpy array
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            # 转换为numpy array，然后转换为tensor
            img_array = np.array(img)
            # 转换为CHW格式 (如果原始是HWC)
            if len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            return torch.from_numpy(img_array)
        except ImportError:
            print("Error: PIL (Pillow) is required for image loading")
            print("Install it with: pip install Pillow")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="计算图片的hash值（与sglang中计算的hash值相同）"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="图片路径或tensor文件路径（.npy或.pt）"
    )
    parser.add_argument(
        "--tensor",
        action="store_true",
        help="输入是tensor文件（.npy或.pt）而不是图片"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="将tensor移到CUDA设备（如果可用）"
    )
    parser.add_argument(
        "--processor",
        action="store_true",
        help="尝试使用sglang的processor处理图片（需要配置）"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # 加载数据
    if args.tensor:
        # 从文件加载tensor
        if args.input.endswith('.npy'):
            feature = np.load(args.input)
            if args.cuda and torch.cuda.is_available():
                feature = torch.from_numpy(feature).cuda()
        elif args.input.endswith('.pt') or args.input.endswith('.pth'):
            feature = torch.load(args.input)
            if args.cuda and torch.cuda.is_available() and isinstance(feature, torch.Tensor):
                feature = feature.cuda()
        else:
            print(f"Error: Unsupported tensor file format: {args.input}")
            print("Supported formats: .npy, .pt, .pth")
            sys.exit(1)
    else:
        # 从图片路径加载
        feature = load_image_as_tensor(args.input, use_processor=args.processor)
        if args.cuda and torch.cuda.is_available():
            feature = feature.cuda()
    
    # 计算hash
    try:
        hash_value = hash_feature(feature)
        print(f"Hash value: {hash_value}")
        print(f"Pad value (hash % (1 << 30)): {hash_value % (1 << 30)}")
        return hash_value
    except Exception as e:
        print(f"Error calculating hash: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
