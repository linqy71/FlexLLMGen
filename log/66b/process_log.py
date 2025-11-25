import re
import os 
import glob

def process_log_file(log_lines):
    max_common_lens = []
    ratios = []

    i = 0
    n = len(log_lines)
    while i < n:
        line = log_lines[i].strip()
        
        # 匹配 max common len 行，例如：max common len 1714 for prefix 1
        max_common_match = re.search(r'max common len (\d+)', line)
        if max_common_match:
            max_common_len = int(max_common_match.group(1))
            max_common_lens.append(max_common_len)

            # 检查下一行是否为 get important tokens 行
            if i + 1 < n:
                next_line = log_lines[i + 1].strip()
                important_match = re.search(r'get ([\d.]+) important tokens', next_line)
                if important_match:
                    important_tokens = float(important_match.group(1))
                    ratio = important_tokens / max_common_len
                    ratios.append(ratio)

            i += 1  # 移动到下一行
        else:
            i += 1

    # 统计 1. max common len 平均值
    avg_max_common_len = sum(max_common_lens) / len(max_common_lens) if max_common_lens else 0

    # 统计 2. important tokens / max common len 比值平均值
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

    return avg_max_common_len, avg_ratio, max_common_lens, ratios


log_files = glob.glob("*rte*.log")
log_files.sort()
res = {}

for log_file in log_files:
    print(log_file)
    # 如果你是从文件读取，可以用下面这行代替上面的 log_lines 定义：
    with open(log_file, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()

    avg_max_len, avg_ratio, maxlens, ratios = process_log_file(log_lines)
    # ds, name = log_file.split('_')[3:5]
    # print(f"1. 所有 max common len 的平均值: {avg_max_len:.2f}")
    print(f"2. important tokens / max common len 的平均比值: {avg_ratio:.6f}")
    print("-----------------")
    # print(f"   （共处理了 {len(maxlens)} 个 max common len，{len(ratios)} 组比值）")

# print(res)