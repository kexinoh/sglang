# 图片Hash计算工具

这个工具用于计算图片的hash值，与 `sglang/srt/managers/schedule_batch.py#L244` 处计算的hash值相同。

## 文件说明

- `calculate_image_hash.py` - 主脚本，用于计算hash值
- `calculate_image_hash_example.py` - 使用示例

## 依赖

```bash
pip install numpy torch
# 可选：如果要从图片路径直接计算
pip install Pillow
# 可选：如果要使用GPU tensor hash（需要sglang）
# sglang会自动提供gpu_tensor_hash函数
```

## 使用方法

### 1. 从图片路径计算hash（基础方法）

```bash
python calculate_image_hash.py image.jpg
```

**注意**: 这种方法使用PIL加载图片，可能与sglang的processor处理结果不同。为了得到完全相同的hash值，应该使用方法2或3。

### 2. 从PyTorch tensor文件计算hash

```bash
# 首先，从sglang中提取feature并保存
# 在你的代码中：
#   torch.save(mm_item.feature, 'feature.pt')

# 然后计算hash
python calculate_image_hash.py --tensor feature.pt

# 如果是CUDA tensor
python calculate_image_hash.py --tensor feature.pt --cuda
```

### 3. 从NumPy array文件计算hash

```bash
# 首先，从sglang中提取feature并保存
# 在你的代码中：
#   np.save('feature.npy', mm_item.feature)

# 然后计算hash
python calculate_image_hash.py --tensor feature.npy
```

## 如何从sglang中提取feature

在你的sglang代码中，可以这样提取feature：

```python
from sglang.srt.managers.schedule_batch import MultimodalDataItem
import torch
import numpy as np

# 假设你有一个 MultimodalDataItem 对象
mm_item = ...  # 从sglang中获取

# 获取feature（优先使用feature，如果没有则使用precomputed_embeddings）
if mm_item.feature is not None:
    feature = mm_item.feature
else:
    feature = mm_item.precomputed_embeddings

# 保存为文件
if isinstance(feature, torch.Tensor):
    torch.save(feature, 'feature.pt')
elif isinstance(feature, np.ndarray):
    np.save('feature.npy', feature)
else:
    print(f"Unsupported feature type: {type(feature)}")
```

## 输出说明

脚本会输出：
- `Hash value`: 计算得到的hash值（整数）
- `Pad value`: `hash % (1 << 30)`，这是sglang中使用的pad值

## 注意事项

1. **GPU tensor**: 如果tensor在CUDA上，且sglang可用，会使用GPU加速的hash计算。如果sglang不可用，会自动将tensor移到CPU再计算（结果可能略有不同）。

2. **图片处理**: 直接从图片路径计算hash时，使用的是PIL的基础处理，可能与sglang的processor结果不同。为了得到完全相同的hash值，应该使用与sglang相同的processor处理图片，然后保存tensor文件。

3. **数据类型**: 脚本支持以下数据类型：
   - `torch.Tensor` (CPU或CUDA)
   - `numpy.ndarray`
   - `list` of tensors/arrays
   - `bytes`

## 示例

查看 `calculate_image_hash_example.py` 获取更多使用示例。
