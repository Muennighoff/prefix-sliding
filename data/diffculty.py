import json
from functools import partial

from simpleverify import verify_code, verify_generic, verify_math
from simpleverify import guessability, verifiability, difficulty
import datasets

from simpleverify.data_filtering import guessability_boxed

### Difficulty ###

ds = datasets.load_dataset("prefixsliding/train_v1", split="train")
ds_math = ds.filter(lambda x: x['domain'] == "math")
ds_math_no_difficulty = ds_math.filter(lambda x: x['difficulty'] is None)

o_math = difficulty(
    ds_math_no_difficulty['problem'], # [:1], # for testing
    ds_math_no_difficulty['solution'], # [:1],
    verify_fn=verify_math,
    m="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    p_start="<｜begin▁of▁sentence｜>You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<｜User｜>",
    p_end="\nPlease reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n",
)

ds_math_no_difficulty = ds_math_no_difficulty.remove_columns("difficulty")
# 16 = got none correct i.e. very difficulty; 0 = got all correct i.e. very easy
ds_math_no_difficulty = ds_math_no_difficulty.add_column("difficulty", [json.dumps({"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": str(int(16 - sum([y[0] for y in x if y[0] is not None]))) + "/16"}, ensure_ascii=False) for x in o_math])
ds_math_no_difficulty = ds_math_no_difficulty.add_column("difficulty_samples", [json.dumps({"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": [y[1] for y in x]}, ensure_ascii=False) for x in o_math])

ds_math_difficulty = ds.filter(lambda x: x['difficulty'] is not None)
ds_math_difficulty = ds_math_difficulty.add_column("difficulty_samples", [None for _ in range(len(ds_math_difficulty))])
ds_math = datasets.concatenate_datasets([ds_math_no_difficulty, ds_math_difficulty])

ds.push_to_hub("prefixsliding/train_v4_math")

