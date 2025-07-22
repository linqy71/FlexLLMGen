# FlexGen LM_Eval Integration

这个集成允许您使用 FlexGen 后端来运行 lm_eval 评估任务。

## 概述

我们修改了原有的 `run_lm_eval_harness.py` 脚本，使其能够支持两种后端：
1. **HuggingFace 后端**：使用标准的 transformers 库
2. **FlexGen 后端**：使用 FlexGen 进行高效的大模型推理

## 主要特性

- ✅ **无缝集成**：复用现有的 loglikelihood 计算逻辑
- ✅ **双后端支持**：可以在 HuggingFace 和 FlexGen 之间切换
- ✅ **完整流程**：支持从数据生成到结果评估的完整流程
- ✅ **兼容现有代码**：不破坏原有的评估流程

## 核心实现

### 关键文件

1. **`run_lm_eval_harness.py`** - 主要脚本，支持 FlexGen 和 HuggingFace 后端
2. **`generate_task_data.py`** - 生成测试数据
3. **`evaluate_task_result.py`** - 评估结果
4. **`demo_integration.py`** - 演示脚本
5. **`test_integration.sh`** - 测试脚本

### 关键函数

```python
def get_flexgen_logits(flexgen_model, input_ids, tokenizer):
    """从 FlexGen 模型获取 logits"""
    # 实现 FlexGen 前向传播
    # 返回与 HuggingFace 兼容的 logits
```

## 使用方法

### 快速开始

1. **运行演示**：
```bash
python demo_integration.py
```

2. **生成测试数据**：
```bash
python generate_task_data.py --output-file input.jsonl --task-name hellaswag
```

3. **使用 HuggingFace 后端**：
```bash
python run_lm_eval_harness.py \
    --input-path input.jsonl \
    --output-path hf_output.jsonl \
    --model-name facebook/opt-350m \
    --model-type opt
```

4. **使用 FlexGen 后端**：
```bash
python run_lm_eval_harness.py \
    --input-path input.jsonl \
    --output-path flexgen_output.jsonl \
    --model-name facebook/opt-350m \
    --model-type opt \
    --use-flexgen
```

5. **评估结果**：
```bash
python evaluate_task_result.py \
    --result-file flexgen_output.jsonl \
    --task-name hellaswag \
    --model-type opt
```

### 高级用法

#### 大模型评估（需要内存卸载）

```bash
python run_lm_eval_harness.py \
    --input-path input.jsonl \
    --output-path output.jsonl \
    --model-name facebook/opt-6.7b \
    --model-type opt \
    --use-flexgen
```

#### 启用小缓存优化

```bash
python run_lm_eval_harness.py \
    --input-path input.jsonl \
    --output-path output.jsonl \
    --model-name facebook/opt-1.3b \
    --model-type opt \
    --use-flexgen \
    --enable_small_cache \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1
```

#### 多任务评估

```bash
# 生成多个任务的数据
for task in hellaswag piqa winogrande; do
    python generate_task_data.py --output-file ${task}_input.jsonl --task-name $task
    python run_lm_eval_harness.py \
        --input-path ${task}_input.jsonl \
        --output-path ${task}_flexgen_output.jsonl \
        --model-name facebook/opt-1.3b \
        --model-type opt \
        --use-flexgen
    python evaluate_task_result.py \
        --result-file ${task}_flexgen_output.jsonl \
        --task-name $task \
        --model-type opt
done
```

## 命令行参数

### 通用参数
- `--input-path`: 输入 JSONL 文件路径
- `--output-path`: 输出 JSONL 文件路径  
- `--model-name`: 模型名称（如 facebook/opt-1.3b）
- `--model-type`: 模型类型（opt, llama, gpt_neox）
- `--cache-dir`: 模型缓存目录

### 后端选择
- `--use-flexgen`: 使用 FlexGen 后端（默认使用 HuggingFace）

### 优化选项
- `--enable_small_cache`: 启用小缓存优化
- `--heavy_ratio`: Heavy hitter 缓存比例
- `--recent_ratio`: Recent 缓存比例

## 工作原理

### 1. 逻辑复用
原有的 `run_lm_eval_harness.py` 已经实现了完整的 loglikelihood 计算：

```python
# 原有代码（第67行）
logits = model(input_ids).logits.log_softmax(dim=-1)

# 我们的修改
if use_flexgen:
    logits = get_flexgen_logits(model, input_ids, tokenizer).log_softmax(dim=-1)
else:
    logits = model(input_ids).logits.log_softmax(dim=-1)
```

### 2. FlexGen 集成
`get_flexgen_logits()` 函数：
1. 设置 FlexGen Task
2. 执行前向传播
3. 从最后一层提取 logits
4. 返回与 HuggingFace 兼容的张量

### 3. 无缝切换
通过 `--use-flexgen` 参数在两种后端之间切换，其余逻辑完全相同。

## 测试和验证

### 运行测试
```bash
bash test_integration.sh
```

### 验证输出格式
两种后端应该产生相同格式的输出：
```json
{
  "request": {...},
  "result": {
    "choices": [{
      "text": "prompt text",
      "logprobs": {
        "tokens": ["token1", "token2", ...],
        "token_logprobs": [null, -1.23, -2.45, ...],
        "top_logprobs": [null, {"token1": -1.23}, ...]
      }
    }]
  }
}
```

## 故障排除

### 常见问题

1. **FlexGen 导入失败**
   ```
   Warning: FlexGen not available, will use HuggingFace only
   ```
   - 解决：安装 FlexGen 或使用 HuggingFace 后端

2. **CUDA 内存不足**
   - 解决：使用更小的模型或启用内存卸载

3. **模型下载失败**
   - 解决：检查网络连接或使用本地模型路径

### 调试技巧

1. **从小模型开始**：先用 `facebook/opt-350m` 测试
2. **检查日志**：观察初始化和处理过程的输出
3. **对比结果**：比较 HuggingFace 和 FlexGen 的输出

## 性能对比

### 内存使用
- **HuggingFace**: 需要完整模型加载到 GPU
- **FlexGen**: 支持 CPU/磁盘卸载，可处理更大模型

### 速度
- **小模型**: HuggingFace 可能更快
- **大模型**: FlexGen 在内存受限情况下更优

## 扩展和自定义

### 支持新模型类型
1. 在 `ENABLE_Heavy_Hitter_FUNCTIONS` 中添加新模型
2. 更新 `get_flexgen_logits` 函数
3. 测试兼容性

### 添加新的优化
1. 修改 FlexGen Policy 配置
2. 调整内存分配策略
3. 优化批处理逻辑

## 结论

这个集成成功地将 FlexGen 与 lm_eval 结合，提供了：
- 简单易用的接口
- 完整的评估流程  
- 高效的大模型推理
- 良好的扩展性

通过复用现有的 loglikelihood 计算逻辑，我们避免了重新实现复杂的评估逻辑，同时获得了 FlexGen 的所有优势。
