# 修改说明：强制在CPU上进行图像处理

## 修改内容

**文件**: `python/sglang/srt/multimodal/processors/base_processor.py`

**位置**: 第257-270行

**修改**: 将图像处理的device从"cuda"改为"cpu"，确保hash值计算的一致性。

### 修改前
```python
if (
    hasattr(processor, "image_processor")
    and isinstance(processor.image_processor, BaseImageProcessorFast)
    and not self.server_args.disable_fast_image_processor
):
    if not _is_npu:
        kwargs["device"] = "cuda"  # ← 使用CUDA
    elif processor.__class__.__name__ not in {
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
    }:
        kwargs["device"] = "npu"
```

### 修改后
```python
if (
    hasattr(processor, "image_processor")
    and isinstance(processor.image_processor, BaseImageProcessorFast)
    and not self.server_args.disable_fast_image_processor
):
    # Force CPU processing to ensure hash value consistency
    # This prevents hash differences between CPU and CUDA processing paths
    kwargs["device"] = "cpu"  # ← 强制使用CPU
    # Original code (commented out):
    # if not _is_npu:
    #     kwargs["device"] = "cuda"
    # elif processor.__class__.__name__ not in {
    #     "Qwen2_5_VLProcessor",
    #     "Qwen3VLProcessor",
    # }:
    #     kwargs["device"] = "npu"
```

## 效果

1. **图像处理在CPU上进行**: 所有图像处理操作现在都在CPU上执行
2. **Hash值一致性**: 由于使用CPU路径，hash值计算将使用SHA256算法，确保一致性
3. **解决pad_value问题**: 相同的图像数据现在会产生相同的hash值和pad_value

## 验证方法

### 1. 检查代码修改
```bash
# 查看修改后的代码
grep -A 10 "Force CPU processing" python/sglang/srt/multimodal/processors/base_processor.py
```

### 2. 运行时验证
在代码中添加日志（可选）：
```python
# 在 base_processor.py 的 process_mm_data 方法中，第271行之前添加：
logger.info(f"Processing multimodal data with device: {kwargs.get('device', 'not set')}")
```

### 3. 测试hash一致性
运行测试脚本：
```bash
python test_hash_device_impact.py
```

现在所有处理都应该在CPU上进行，hash值应该是一致的。

## 注意事项

1. **性能**: CPU处理可能比GPU慢，但确保了hash值的一致性
2. **内存**: 不会占用GPU内存进行图像预处理
3. **后续处理**: 处理后的tensor会在第277-286行自动移动到CPU（如果`keep_mm_feature_on_device=False`）

## 如果需要恢复CUDA处理

如果需要恢复原来的CUDA处理逻辑，可以：
1. 取消注释原来的代码
2. 删除 `kwargs["device"] = "cpu"` 这一行
3. 恢复原来的if-elif逻辑

## 相关文件

- **修改文件**: `python/sglang/srt/multimodal/processors/base_processor.py`
- **分析文档**: `analysis_device_impact_on_hash.md`
- **解决方案文档**: `solution_cpu_image_processing.md`
- **测试脚本**: `test_hash_device_impact.py`
