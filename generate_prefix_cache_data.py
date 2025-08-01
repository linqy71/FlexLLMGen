import json
import numpy as np
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import os
import sys

# 从命令行参数读取配置
# python generate_prefix_cache_data.py openbookqa input.jsonl 200 facebook/opt-13b

if len(sys.argv) != 5:
    print("错误: 参数数量不正确。")
    print("使用方法: python your_script_name.py <TASK_NAME> <INPUT_FILE> <TARGET_TOKEN_COUNT> <TOKENIZER_NAME>")
    sys.exit(1)

TASK_NAME = sys.argv[1]
INPUT_FILE = sys.argv[2]
try:
    TARGET_TOKEN_COUNT = int(sys.argv[3])
except ValueError:
    print(f"错误: TARGET_TOKEN_COUNT '{sys.argv[3]}' 必须是一个有效的整数。")
    sys.exit(1)
TOKENIZER_NAME = sys.argv[4]

OUTPUT_FILE_TEMPLATE = f'prefix_caching_dataset_max2k_{TASK_NAME}.jsonl'
NUM_UNIQUE_PREFIXES = 100
NORMAL_DIST_SCALE = NUM_UNIQUE_PREFIXES / 4

def create_prefixes(prompts, tokenizer, target_token_count):
    """根据给定的提示列表，创建指定数量的、达到目标 token 数的唯一前缀。"""
    prefixes = []
    print(f"正在为目标 {target_token_count} tokens 创建 {NUM_UNIQUE_PREFIXES} 个唯一前缀...")
    
    prompts_pool = list(prompts)
    if not prompts_pool:
        raise ValueError("输入提示列表为空。")

    for _ in tqdm(range(NUM_UNIQUE_PREFIXES), desc="创建前缀"):
        current_prefix_prompts = []
        current_token_count = 0
        # 循环构建前缀，直到达到或超过目标 token 数量
        while current_token_count < target_token_count:
            prompt_to_add = random.choice(prompts_pool)
            current_prefix_prompts.append(prompt_to_add)
            # 使用 encode 方法来精确计算 token 数量
            current_token_count = len(tokenizer.encode("\n\n".join(current_prefix_prompts)))
        
        prefix = "\n\n".join(current_prefix_prompts)
        prefixes.append(prefix)
    return prefixes

def generate_dataset(input_file, output_file, tokenizer, target_token_count):
    """生成一个数据集，其中共享前缀遵循正态分布，并保持4行一组的结构。"""
    print(f"\n正在为 {output_file} 生成数据集...")
    
    if not os.path.exists(input_file):
        print(f"错误：在 '{input_file}' 未找到输入文件")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records or len(records) % 4 != 0:
        print(f"错误：'{input_file}' 中的记录为空或总行数不是4的倍数。当前行数: {len(records)}")
        return

    record_groups = [records[i:i + 4] for i in range(0, len(records), 4)]
    num_question_groups = len(record_groups) 
    
    prompts_for_prefix_creation = [rec['prompt'] for rec in records]
    
    prefixes = create_prefixes(prompts_for_prefix_creation, tokenizer, target_token_count)
    
    loc = NUM_UNIQUE_PREFIXES / 2
    
    # 这将确保每个问题组都能分配到一个按正态分布抽取的前缀
    print(f"正在为 {num_question_groups} 个问题组生成正态分布的前缀索引...")
    prefix_indices = np.random.normal(loc=loc, scale=NORMAL_DIST_SCALE, size=num_question_groups)
    prefix_indices = np.clip(prefix_indices, 0, NUM_UNIQUE_PREFIXES - 1).astype(int)
    
    with open(output_file, 'w') as f:
        print(f"正在为所有 {len(record_groups)} 个问题组生成数据并写入到 {output_file}...")
        for group in tqdm(record_groups, desc="生成问题组"):
            chosen_prefix = random.choice(prefixes)
            
            template_group = group 
            
            for record in template_group:
                final_prompt = chosen_prefix + "\n\n" + record['prompt']
                output_record = record.copy()
                output_record['prompt'] = final_prompt
                
                f.write(json.dumps(output_record) + '\n') 
    print(f"成功将数据集写入 {output_file}")

def main():
    """主函数，初始化并生成数据集。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except OSError:
        print(f"错误：无法加载分词器 '{TOKENIZER_NAME}'。")
        print("请确保您已连接到互联网并且分词器名称正确。")
        return

    generate_dataset(INPUT_FILE, OUTPUT_FILE_TEMPLATE, tokenizer, TARGET_TOKEN_COUNT)

if __name__ == '__main__':
    main()