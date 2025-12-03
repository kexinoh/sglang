# 解决方案：在CPU上进行图像处理

## 问题背景

在 `python/sglang/srt/multimodal/processors/base_processor.py` 的 `process_mm_data` 方法中（第264行），代码会自动设置 `device="cuda"`，这会导致：
1. 图像处理在GPU上进行
2. Hash值计算使用GPU hash算法，导致不一致性

## 解决方案

### 方案1：修改代码强制使用CPU（推荐）

修改 `base_processor.py` 文件，将 device 设置为 "cpu"：

**文件位置**: `python/sglang/srt/multimodal/processors/base_processor.py`

**修改位置**: 第258-270行

**修改前**:
```python
if (
    hasattr(processor, "image_processor")
    and isinstance(processor.image_processor, BaseImageProcessorFast)
    and not self.server_args.disable_fast_image_processor
):
    if not _is_npu:
        kwargs["device"] = "cuda"  # ← 这里强制使用CUDA
    elif processor.__class__.__name__ not in {
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
    }:
        kwargs["device"] = "npu"
```

**修改后**:
```python
if (
    hasattr(processor, "image_processor")
    and isinstance(processor.image_processor, BaseImageProcessorFast)
    and not self.server_args.disable_fast_image_processor
):
    # 强制使用CPU进行图像处理，确保hash值一致性
    kwargs["device"] = "cpu"  # ← 改为CPU
    # 注释掉原来的CUDA/NPU逻辑
    # if not _is_npu:
    #     kwargs["device"] = "cuda"
    # elif processor.__class__.__name__ not in {
    #     "Qwen2_5_VLProcessor",
    #     "Qwen3VLProcessor",
    # }:
    #     kwargs["device"] = "npu"
```

### 方案2：添加配置选项（更灵活）

如果你希望保留灵活性，可以添加一个配置选项：

**步骤1**: 在 `server_args.py` 中添加新配置

**文件位置**: `python/sglang/srt/server_args.py`

在 `ServerArgs` 类中添加（约第530行附近）:
```python
mm_processor_device: str = "cpu"  # 可选: "cpu", "cuda", "auto"
```

**步骤2**: 修改 `base_processor.py`

**文件位置**: `python/sglang/srt/multimodal/processors/base_processor.py`

**修改**:
```python
if (
    hasattr(processor, "image_processor")
    and isinstance(processor.image_processor, BaseImageProcessorFast)
    and not self.server_args.disable_fast_image_processor
):
    # 使用配置的device，默认为CPU
    if self.server_args.mm_processor_device == "auto":
        # 自动选择：非NPU使用CUDA，NPU使用NPU
        if not _is_npu:
            kwargs["device"] = "cuda"
        elif processor.__class__.__name__ not in {
            "Qwen2_5_VLProcessor",
            "Qwen3VLProcessor",
        }:
            kwargs["device"] = "npu"
    else:
        # 使用配置的device
        kwargs["device"] = self.server_args.mm_processor_device
```

**步骤3**: 在启动时设置

```python
# 在启动SGLang服务器时
server_args = ServerArgs(
    # ... 其他参数 ...
    mm_processor_device="cpu",  # 设置为CPU
)
```

### 方案3：使用环境变量（最简单，无需修改代码）

如果你不想修改代码，可以通过设置环境变量来禁用fast image processor，这会间接影响device设置：

```bash
# 方法1: 禁用fast image processor（会回退到慢速版本，但可能在CPU上）
export SGLANG_DISABLE_FAST_IMAGE_PROCESSOR=1

# 或者通过命令行参数
python -m sglang.launch_server_http \
    --disable-fast-image-processor \
    # ... 其他参数
```

**注意**: 这个方法可能不会完全解决问题，因为即使禁用了fast processor，某些processor可能仍然会在GPU上处理。

## 推荐方案

**推荐使用方案1**，因为：
1. 简单直接，确保在CPU上处理
2. 解决hash值不一致的问题
3. 不需要额外的配置

## 验证修改

修改后，可以通过以下方式验证：

1. **检查processor调用**:
   - 在 `base_processor.py` 的 `process_mm_data` 方法中添加日志：
   ```python
   logger.info(f"Processing with device: {kwargs.get('device', 'not set')}")
   ```

2. **检查hash值一致性**:
   - 运行之前创建的测试脚本 `test_hash_device_impact.py`
   - 确认CPU和CUDA路径产生相同的hash值（如果都使用CPU）

3. **性能考虑**:
   - CPU处理可能比GPU慢，但对于hash计算的一致性更重要
   - 处理后的tensor会在第277-286行自动移动到CPU（如果 `keep_mm_feature_on_device=False`）

## 注意事项

1. **性能影响**: 在CPU上处理图像可能比GPU慢，但确保了hash值的一致性
2. **内存**: CPU处理不会占用GPU内存，但会占用CPU内存
3. **兼容性**: 某些processor可能有GPU特定的优化，在CPU上可能表现不同

## 相关代码位置

- **主要修改点**: `python/sglang/srt/multimodal/processors/base_processor.py` 第258-270行
- **配置选项**: `python/sglang/srt/server_args.py` 第530行附近
- **Hash计算**: `python/sglang/srt/managers/mm_utils.py` 第797-820行
