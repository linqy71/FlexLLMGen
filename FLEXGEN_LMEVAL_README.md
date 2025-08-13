# FlexGen LM_Eval Integration
### 关键文件

1. **`run_lm_eval_harness.py`** - 主要脚本，支持 FlexGen 和 HuggingFace 后端
2. **`generate_task_data.py`** - 生成测试数据
3. **`evaluate_task_result.py`** - 评估结果
5. **`compare_backends.sh`** - 测试脚本

### 关键函数

```python
def get_flexgen_logits(flexgen_model, input_ids, tokenizer):
    """从 FlexGen 模型获取 logits"""
    # 实现 FlexGen 前向传播
    # 返回与 HuggingFace 兼容的 logits
```

## 使用方法

## 生成测试数据
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

## 命令行参数

### 通用参数
- `--input-path`: 输入 JSONL 文件路径
- `--output-path`: 输出 JSONL 文件路径  
- `--model-name`: 模型名称（如 facebook/opt-1.3b）
- `--model-type`: 模型类型（opt, llama, gpt_neox）
- `--cache-dir`: 模型缓存目录

### 后端选择
- `--use-flexgen`: 使用 FlexGen 后端（默认使用 HuggingFace）


## 工作原理

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

## 测试和验证

### 运行测试
```bash
bash eval_long_tasks.sh
```
