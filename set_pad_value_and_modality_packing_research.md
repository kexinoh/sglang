# set_pad_value 和模态打包调研报告

## 1. pad_values 和 set_pad_value 的引入时间

### pad_values 的引入（最初版本）
`pad_values` 概念是在 **commit fd9ad817e** (2024年9月28日) 引入的，这个 commit 的标题是：
```
Organize image inputs (#1531)
```

#### 最初的设计
在 commit fd9ad817e 中，引入了 `ImageInputs` 类：

```python
@dataclass
class ImageInputs:
    pixel_values: torch.Tensor
    image_hash: int
    image_sizes: Optional[list] = None
    image_offsets: Optional[list] = None
    pad_values: Optional[list] = None  # 列表，包含4个值
    modalities: Optional[list] = None
```

#### pad_values 的计算方式（最初版本）
```python
@staticmethod
def from_dict(obj, vocab_size):
    # 将所有图片的哈希值组合成一个元组，然后计算哈希
    ret = ImageInputs(
        pixel_values=obj["pixel_values"],
        image_hash=hash(tuple(obj["image_hashes"])),  # 所有图片哈希的组合
    )
    image_hash = ret.image_hash
    # 从 image_hash 生成4个 pad_values
    ret.pad_values = [
        (image_hash) % vocab_size,
        (image_hash >> 16) % vocab_size,
        (image_hash >> 32) % vocab_size,
        (image_hash >> 64) % vocab_size,
    ]
```

**关键观察**：
- 在引入 pad_values 时，**所有图片的哈希值被组合成一个 `image_hash`**（通过 `hash(tuple(obj["image_hashes"]))`）
- 从这个单一的 `image_hash` 生成4个 `pad_values`
- `pixel_values` 是一个 `torch.Tensor`，多张图片被组织在一个张量中
- **这意味着在引入 pad_values 时，所有图片被组织在一起，共享同一个 image_hash 和 pad_values**

### set_pad_value 的引入（重构版本）
`set_pad_value` 方法是在 **commit 5cb552b1d** (2025年4月1日) 引入的，这个 commit 的标题是：
```
refactor: multimodal data (#4754)
```

#### 引入背景
在引入 `set_pad_value` 之前：
- `pad_values` 是 `MultimodalInputs`（之前是 `ImageInputs`）类的一个列表字段
- 所有图片共享一个 `image_hash`，从这个 hash 生成多个 `pad_values`

在引入 `set_pad_value` 之后：
- 引入了 `MultimodalDataItem` 类，每个 item 代表一种模态的所有输入
- 每个 `MultimodalDataItem` 有自己的 `pad_value` 字段（单个值，不再是列表）
- 通过 `set_pad_value()` 方法为每个 item 单独计算 pad value

### set_pad_value 的实现
```python
def set_pad_value(self):
    """
    Set the pad value after first hashing the data
    """
    from sglang.srt.managers.mm_utils import hash_feature

    if self.hash is None:
        if self.feature is not None:
            hashed_feature = self.feature
        else:
            hashed_feature = self.precomputed_embeddings
        self.hash = hash_feature(hashed_feature)
    assert self.hash is not None
    self.pad_value = self.hash % (1 << 30)
```

pad_value 的计算方式：`pad_value = hash % (1 << 30)`，即对特征数据的哈希值取模。

## 2. 模态打包机制

### pad_values 引入时的打包情况（2024年9月）
在最初引入 `pad_values` 时（commit fd9ad817e）：
- **所有图片被组织在一起**：`pixel_values` 是一个 `torch.Tensor`，包含所有图片
- **共享一个 image_hash**：所有图片的哈希值被组合成一个元组，然后计算哈希
- **共享 pad_values**：从单一的 `image_hash` 生成4个 `pad_values`，所有图片使用相同的 pad_values
- **支持 modalities 字段**：虽然存在 `modalities` 字段，但当时主要处理图片，多张图片共享同一个 hash 和 pad_values

**结论**：在 pad_values 引入时，**多张图片是打包在一起的**，共享同一个 image_hash 和 pad_values。

### MultimodalDataItem 引入时的打包情况（2025年4月）
根据代码注释和实现，**sglang 在引入 MultimodalDataItem 时，确实默认将同一种模态打包在一起**。

### 证据

1. **MultimodalDataItem 的注释**（`schedule_batch.py:197-199`）：
```python
"""
One MultimodalDataItem contains all inputs for one modality.
For example, if there are 3 images and 1 audio inputs, there will be 2 MultimodalDataItem.
One for images and one for audio.
```

2. **embed_mm_inputs 函数的实现**（`mm_utils.py:540-543`）：
```python
# 2. Get multimodal embedding separately
# Try get mm embedding if any
for modality in Modality.all():
    items = [
        item for item in item_flatten_list if item.is_modality(modality=modality)
    ]
```
代码按模态类型分组处理，将同一种模态的所有 items 收集在一起。

3. **MultimodalInputs 的结构**：
- `mm_items: List[MultimodalDataItem]` - 每个 item 代表一种模态
- 在 `from_dict` 方法中（`schedule_batch.py:331-332`），会为每个 item 调用 `set_pad_value()`

### 打包逻辑
- **同一种模态的数据会被打包到一个 MultimodalDataItem 中**
- 例如：如果有 3 张图片和 1 个音频，会创建 2 个 MultimodalDataItem：
  - 一个包含所有 3 张图片（modality=IMAGE）
  - 一个包含音频（modality=AUDIO）

### 调用时机
`set_pad_value()` 在以下位置被调用：
1. `MultimodalInputs.from_dict()` - 创建 MultimodalInputs 时（`schedule_batch.py:332`）
2. `MultimodalDataItem.merge()` - 合并多个 items 时（`schedule_batch.py:292`）

## 3. 相关 Git 历史

### 关键 Commits
- `fd9ad817e` (2024-09-28): Organize image inputs (#1531) - **最早引入 pad_values 概念**
- `5cb552b1d` (2025-04-01): refactor: multimodal data (#4754) - **引入 set_pad_value 和 MultimodalDataItem**
- `f50a6cf44`: Fix hash collision for multi modal models (#2256) - 修复多模态模型的哈希冲突
- `3c79ad35c`: [Fix] Fix the padded hash value for image tokens (#2309) - 修复图像 token 的填充哈希值

### 相关文件变更
在 refactor commit 中，主要修改了：
- `python/sglang/srt/managers/schedule_batch.py` - 引入 MultimodalDataItem
- `python/sglang/srt/managers/mm_utils.py` - 更新 pad_values 的使用方式
- 多个模型文件 - 适配新的数据结构

## 4. 代码示例

### 模态打包示例
在 `embed_mm_inputs` 函数中（`mm_utils.py:540-567`），可以看到模态打包的处理逻辑：

```python
# 1. 首先将所有 mm_items 展平
item_flatten_list = []
for mm_inputs in mm_inputs_list:
    item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

# 2. 按模态类型分组处理
for modality in Modality.all():  # IMAGE, VIDEO, AUDIO
    items = [
        item for item in item_flatten_list if item.is_modality(modality=modality)
    ]
    # 处理同一种模态的所有 items
    if len(items) != 0:
        placeholder_tensor = torch.as_tensor(
            [item.pad_value for item in items],  # 收集该模态所有 items 的 pad_value
            device=input_ids.device,
        )
        # ... 后续处理
```

### set_pad_value 调用链
1. **创建时**：`MultimodalInputs.from_dict()` → 遍历 `mm_items` → 调用 `item.set_pad_value()`
2. **合并时**：`MultimodalDataItem.merge()` → 合并 hash → 调用 `self.set_pad_value()`

## 5. 总结

### pad_values 引入时（2024年9月28日，commit fd9ad817e）
1. **打包方式**：
   - ✅ **是的**，在引入 pad_values 时，**所有图片被组织在一起**
   - 多张图片的哈希值被组合成一个 `image_hash`：`hash(tuple(obj["image_hashes"]))`
   - 从这个单一的 `image_hash` 生成4个 `pad_values`，所有图片共享这些 pad_values
   - `pixel_values` 是一个 `torch.Tensor`，包含所有图片数据

2. **设计特点**：
   - 所有图片共享同一个 image_hash 和 pad_values
   - 支持 `modalities` 字段，但当时主要处理图片
   - pad_values 是一个包含4个值的列表

### set_pad_value 引入时（2025年4月1日，commit 5cb552b1d）
1. **打包方式**：
   - ✅ **是的**，sglang 在引入 MultimodalDataItem 时，默认将同一种模态的所有输入打包到一个 MultimodalDataItem 中
   - 例如：3 张图片 + 1 个音频 = 2 个 MultimodalDataItem（1个IMAGE类型，1个AUDIO类型）
   - 每种模态有自己独立的 `pad_value`（单个值，不再是列表）

2. **设计改进**：
   - 从共享的 image_hash 改为每个 MultimodalDataItem 独立计算 hash 和 pad_value
   - 支持真正的多模态（IMAGE、VIDEO、AUDIO），每种模态独立处理
   - pad_value 从列表改为单个值：`pad_value = hash % (1 << 30)`

### 演进过程
- **2024年9月**：引入 pad_values，所有图片打包在一起，共享 hash 和 pad_values
- **2025年4月**：引入 MultimodalDataItem 和 set_pad_value，按模态类型打包，每种模态独立计算 pad_value

### 设计目的
- 简化多模态数据的组织和管理
- 为每种模态单独计算和存储 pad_value（基于特征数据的哈希值）
- 支持按模态类型进行批量处理（在 embed_mm_inputs 中按模态分组处理）
- 提高 Radix Attention 的缓存效率（通过 pad_value 进行前缀匹配）
