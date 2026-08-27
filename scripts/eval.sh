#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jobs/%x-%A_%a.out
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --array=0-0%1

# === Setup ===
conda activate s

cd lm-evaluation-harness
START_TIME=$(date +%s)
echo "START TIME: $(date)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# === Parameter grids ===
# Testing
SW_VALUES=(4096)
# Full suite
#SW_VALUES=(4096 8192 16384)

# AIME/GPQA/MATH500 ; Note current  --array=0-0%1 only runs aime; increase for more
TASKS=(aime25_nofigures_tok524288_agg64 gpqa_diamond_openai_tok524288_agg64 openai_math_tok524288_agg64)
# If you have memory issues, use the below & adjust max_gen_toks to 262144 & max_position_embeddings to ~263000 (slightly larger to fit prompt)
# TASKS=(aime25_nofigures_tok262144_agg64 gpqa_diamond_openai_tok262144_agg64 openai_math_tok262144_agg64)
# You can also go further down to 131072 & 65536 though it may slightly reduce scores
# HealthBench
# TASKS=(healthbench_tok32768)

# Compute indices
NUM_TASKS=${#TASKS[@]}
SW_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_TASKS))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_TASKS))

SW=${SW_VALUES[$SW_INDEX]}
TASK=${TASKS[$TASK_INDEX]}

echo "Running SW=$SW, TASK=$TASK"

# === Environment variables ===
# Decide on API provider; This is needed for verification; cost should be very small maybe ~<1$ for one run
# export PORTKEY_API_KEY="YOUR_TOKEN"
# export OPENAI_API_KEY=YOUR_TOKEN

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Needed for GPQA as gated dataset on HF
# export HF_TOKEN=YOUR_TOKEN

# === Run ===
### Running Full Attention
# lm_eval \
#   --model vllm \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"max_position_embeddings\":530000}}" \
#   --tasks $TASK \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path Qwen3-1.7B/toks524288/${TASK} \
#   --log_samples \
#   --gen_kwargs max_gen_toks=524288,temperature=0.6,top_p=0.95

### Running Prefix Sliding
# Comment out for no Prefix Sliding
export SWF=$SW
lm_eval \
  --model vllm \
  --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"sliding_window\":${SW},\"max_position_embeddings\":530000,\"use_sliding_window_attention\":true}}" \
  --tasks $TASK \
  --batch_size auto \
  --apply_chat_template \
  --output_path ./Qwen3-1.7B/swf${SW}/toks524288/${TASK} \
  --log_samples \
  --gen_kwargs max_gen_toks=524288,temperature=0.6,top_p=0.95

### Running Sliding Window
# unset SWF
# lm_eval \
#   --model vllm \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"sliding_window\":${SW},\"max_position_embeddings\":530000,\"use_sliding_window_attention\":true}}" \
#   --tasks $TASK \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/sw${SW}/toks524288/${TASK} \
#   --log_samples \
#   --gen_kwargs max_gen_toks=524288,temperature=0.6,top_p=0.95

### Running Last k
# export MAX_SAVING="${MAX_SAVING:-63}"
# export SAVE_LAST_K="${SAVE_LAST_K:-256}"
# export SUBTRACT="${SUBTRACT:-1}"
# export USE_MAX="${USE_MAX:-true}"
# export MAX_TOK_PER_WINDOW="${SW:-4096}"

# lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"max_position_embeddings\":270000}}" \
#   --tasks "${TASK}" \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/lastk${SW}/toks262144/${TASK} \
#   --log_samples \
#   --gen_kwargs max_gen_toks=262144,temperature=0.6,top_p=0.95

### Running Summary
SYSTEM_PROMPT='# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "pass", "description": "Passes the task to the next model to solve the problem from where you left off. Useful when thinking gets too long.", "parameters": {"type": "object", "properties": {"context": {"type": "string", "description": "Information to pass to the next model. E.g., ideas already tried, key results, next steps to perform..."}}, "required": ["context"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Below is an example of using the pass tool:
<|im_start|>user
There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.<|im_end|>
<|im_start|>assistant
<think>
Okay, so I need to solve this problem where there are real numbers x and y, both greater than 1, such that log base x of (y^x) equals log base y of (x^{4y}) equals 10. And I need to find the product xy. Hmm, let me think.

First, let me recall some logarithm properties. The equation says that two different logarithms are equal to 10. So, log_x(y^x) = 10 and log_y(x^{4y}) = 10. Maybe I can convert these logarithmic equations into exponential form to make them easier to handle.

Starting with the first one: log_x(y^x) = 10. Using the logarithm power rule, which says that log_b(a^c) = c*log_b(a). So, applying that here, log_x(y^x) = x * log_x(y) = 10. So, x * log_x(y) = 10. 

But log_x(y) is the same as 1 / log_y(x), by the change of base formula. Wait, but maybe I can express log_x(y) in terms of something else. Let me
<tool_call>
{"name": "pass", "arguments": {"context": "Maybe convert logarithmic equations into exponential form to make them easier to handle. Via logarithm power rule log_x(y^x) = x * log_x(y) = 10. Noting that log_x(y) is the same as 1 / log_y(x). Maybe try express log_x(y) in terms of something else."}}
</tool_call>

When context is passed to you, evaluate its correctness first and then use it if correct.
<|im_start|>user
There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.<|im_end|>
<|im_start|>assistant
<context>
Maybe convert logarithmic equations into exponential form to make them easier to handle. Via logarithm power rule log_x(y^x) = x * log_x(y) = 10. Noting that log_x(y) is the same as 1 / log_y(x). Maybe try express log_x(y) in terms of something else.
</context>
<think>
Okay, so I need to first evaluate whether the context is correct and then use it to solve the problem.

The context suggests converting logarithmic equations into exponential form and using the logarithm power rule, which is a valid approach. It also notes that log_x(y) can be expressed in terms of log_y(x), which is correct. It suggests expressing log_x(y) in terms of something else.

Wait, do we really need to express log_x(y) in terms of something else? Maybe we can instead first use the logarithm power rule for the second equation as well. Let me do that...'

# export MAX_SAVING="${MAX_SAVING:-63}"
# export USE_MAX="${USE_MAX:-true}"
# export SAVING_PROMPT="context"
# export BUDGET_FORCE_SAVING="${BUDGET_FORCE_SAVING:-256}"
# export MAX_TOK_PER_WINDOW=4096

# lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"max_position_embeddings\":270000}}" \
#   --tasks $TASK \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/summary${SW}/toks262144/${TASK} \
#   --log_samples \
#   --gen_kwargs max_gen_toks=262144,temperature=0.6,top_p=0.95 \
#   --system_instruction "${SYSTEM_PROMPT}"







### quick tests to run locally

# SWF=128 lm_eval \
#   --model vllm \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"sliding_window\":128,\"max_position_embeddings\":100000,\"use_sliding_window_attention\":true}}" \
#   --tasks aime25_nofigures_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/swf128/toks100/aime25_nofigures_agg64 \
#   --log_samples \
#   --gen_kwargs max_gen_toks=100,temperature=0.6,top_p=0.95 --limit 1

# MAX_SAVING=2 SAVE_LAST_K=128 SUBTRACT=1 USE_MAX=true MAX_TOK_PER_WINDOW=256 lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\"}" \
#   --tasks aime25_nofigures_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/lastk128/toks262144/aime \
#   --log_samples \
#   --gen_kwargs max_gen_toks=256,temperature=0.6,top_p=0.95 --limit 1

# MAX_TOK_PER_WINDOW=256 MAX_SAVING=2 FORCE_SAVING=true SAVING_PROMPT=context USE_MAX=true lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\"}" \
#   --tasks aime25_nofigures_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/summary256/toks262144/aime \
#   --log_samples \
#   --gen_kwargs max_gen_toks=256,temperature=0.6,top_p=0.95 \
#   --system_instruction "${SYSTEM_PROMPT}" --limit 1

### real runs to run locally

# SWF=4096 lm_eval \
#   --model vllm \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"sliding_window\":4096,\"max_position_embeddings\":263000,\"use_sliding_window_attention\":true}}" \
#   --tasks aime25_nofigures_tok262144_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/swf4096/toks262144/aime25_nofigures_tok262144_agg64 \
#   --log_samples \
#   --gen_kwargs max_gen_toks=262144,temperature=0.6,top_p=0.95

# MAX_TOK_PER_WINDOW=4096 MAX_SAVING=64 BUDGET_FORCE_SAVING=256 SAVING_PROMPT=context USE_MAX=true lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"max_position_embeddings\":270000}}" \
#   --tasks aime25_nofigures_tok262144_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/w4096s256context/toks262144/aime \
#   --log_samples \
#   --gen_kwargs max_gen_toks=262144,temperature=0.6,top_p=0.95 \
#   --system_instruction "${SYSTEM_PROMPT}"

# MAX_TOK_PER_WINDOW=4096 MAX_SAVING=64 SAVE_LAST_K=256 SUBTRACT=1 USE_MAX=true lm_eval \
#   --model vllm-tool \
#   --model_args "{\"pretrained\":\"Qwen/Qwen3-1.7B\",\"tensor_parallel_size\":1,\"data_parallel_size\":1,\"dtype\":\"auto\",\"hf_overrides\":{\"max_position_embeddings\":270000}}" \
#   --tasks aime25_nofigures_tok262144_agg64 \
#   --batch_size auto \
#   --apply_chat_template \
#   --output_path ./Qwen3-1.7B/lastk4096/toks262144/aime \
#   --log_samples \
#   --gen_kwargs max_gen_toks=262144,temperature=0.6,top_p=0.95
