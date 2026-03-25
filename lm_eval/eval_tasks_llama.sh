#!/bin/bash
#
# LM Eval Harness for FlexLLMGen Llama
#
# Usage:
#   1. Generate task data:  bash eval_tasks_llama.sh generate <task_name>
#   2. Run inference:       bash eval_tasks_llama.sh infer <task_name> [K] [L] [threshold]
#   3. Evaluate results:    bash eval_tasks_llama.sh eval <task_name> [K] [L]
#   4. Full pipeline:       bash eval_tasks_llama.sh all <task_name> [K] [L] [threshold]

set -e

# ---- Configuration ----
HF_HOME="/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/HF_HOME"
TOKENIZER_PATH="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_WEIGHT_PATH="${HF_HOME}/hub"
OFFLOAD_DIR="/HOME/nsccgz_qylin/nsccgz_qylinxy_1/HDD_POOL/lqy/offload_dir/llama"
MODEL_NAME="meta/llama3.1-8b"

SEQ=4096
NUM_FEWSHOT=0
LIMIT_SAMPLES=200
GEN_LEN=8
PERCENT="100 0 100 0 100 0"

# Output directories
mkdir -p datasets logits acc ../log

# ---- Parse arguments ----
ACTION=${1:-"all"}
TASK_NAME=${2:-"copa"}
K=${3:-10}
L=${4:-150}
THRESHOLD=${5:-2}

echo "Action: $ACTION, Task: $TASK_NAME, K=$K, L=$L, threshold=$THRESHOLD"

# ---- Step 1: Generate task data ----
generate_data() {
    echo "=== Generating task data for $TASK_NAME ==="
    INPUT_FILE="datasets/${TASK_NAME}_fs${NUM_FEWSHOT}.jsonl"

    cd "$(dirname "$0")"
    MODEL_NAME_ENV="$TOKENIZER_PATH" python3 generate_task_data.py \
        --output-file "$INPUT_FILE" \
        --task-name "$TASK_NAME" \
        --num-fewshot "$NUM_FEWSHOT" \
        --limit "$LIMIT_SAMPLES" \
        --seq "$SEQ" \
        --tokenizer-path "$TOKENIZER_PATH"

    echo "Generated $(wc -l < "$INPUT_FILE") samples to $INPUT_FILE"
}

# ---- Step 2: Run FlexLLMGen inference ----
run_inference() {
    echo "=== Running FlexLLMGen inference: $TASK_NAME K=${K} L=${L} threshold=${THRESHOLD} ==="
    INPUT_FILE="datasets/${TASK_NAME}_fs${NUM_FEWSHOT}.jsonl"
    LOGITS_FILE="logits/${TASK_NAME}_K${K}L${L}T${THRESHOLD}.jsonl"
    LOG_FILE="../log/lm_eval_${TASK_NAME}_K${K}L${L}T${THRESHOLD}.log"

    cd "$(dirname "$0")"
    python3 -u run_lm_eval_harness.py \
        --input-path "$INPUT_FILE" \
        --output-path "$LOGITS_FILE" \
        --use-flexgen \
        --model "$MODEL_NAME" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --percent $PERCENT \
        --path "$MODEL_WEIGHT_PATH" \
        --prompt-len "$SEQ" \
        --gen-len "$GEN_LEN" \
        --strategy seq \
        --overlap=False \
        --K="$K" \
        --L="$L" \
        --collision-threshold "$THRESHOLD" \
        --offload-dir "$OFFLOAD_DIR" > "$LOG_FILE" 2>&1

    if [ -f "$LOGITS_FILE" ]; then
        echo "Inference complete: $(wc -l < "$LOGITS_FILE") results in $LOGITS_FILE"
    else
        echo "ERROR: Inference failed. Check $LOG_FILE"
        exit 1
    fi
}

# ---- Step 3: Evaluate results ----
evaluate_results() {
    echo "=== Evaluating results: $TASK_NAME K=${K} L=${L} ==="
    LOGITS_FILE="logits/${TASK_NAME}_K${K}L${L}T${THRESHOLD}.jsonl"
    ACC_FILE="acc/${TASK_NAME}_K${K}L${L}T${THRESHOLD}.json"

    cd "$(dirname "$0")"
    python3 evaluate_task_result.py \
        --result-file "$LOGITS_FILE" \
        --task-name "$TASK_NAME" \
        --model-type llama \
        --num-fewshot "$NUM_FEWSHOT" \
        --limit "$LIMIT_SAMPLES" \
        --seq "$SEQ" \
        --acc-file "$ACC_FILE" \
        --tokenizer-path "$TOKENIZER_PATH"

    echo "Results saved to $ACC_FILE"
    echo "--- Accuracy ---"
    python3 -c "
import json
with open('$ACC_FILE') as f:
    r = json.load(f)
if 'results' in r:
    for task, metrics in r['results'].items():
        print(f'{task}:')
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')
"
}

# ---- Run selected action ----
case "$ACTION" in
    generate)
        generate_data
        ;;
    infer)
        run_inference
        ;;
    eval)
        evaluate_results
        ;;
    all)
        generate_data
        run_inference
        evaluate_results
        ;;
    sweep)
        # Run a sweep over multiple threshold values
        generate_data
        for T in 2 3 4 5; do
            THRESHOLD=$T
            echo ""
            echo "========== Threshold=$T =========="
            run_inference
            evaluate_results
        done
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 {generate|infer|eval|all|sweep} <task_name> [K] [L] [threshold]"
        exit 1
        ;;
esac

echo "=== Done ==="
