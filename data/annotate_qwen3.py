import datasets
import json
import os
from functools import partial
from tqdm import tqdm
from litellm import text_completion
import argparse

from simpleverify import verify_math
from simpleverify import guessability, verifiability, difficulty
from simpleverify.data_filtering import guessability_boxed

from simpleverify.prompts import GUESSABILITY_PROMPT, GUESSABILITY_PROMPT_BOXED, GUESSABILITY_PROMPT_CROSSWORD_BOXED, VERIFIABLE_PROMPT, VERIFIABLE_PROMPT_NOANSWER

MODEL = "Qwen/Qwen3-1.7B"
P_START = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
P_END = "<|im_end|>\n<|im_start|>assistant\nThe answer is \\boxed"
ROLLOUT_NUM = 8

def load_ds(data_name: str, split: str, domain: str):
    ds = datasets.load_dataset(data_name, split=split)
    ds_math = ds.filter(lambda x: x['domain'] == domain)
    return ds_math

def make_requests(prompt_lst: list[str]):
    response = text_completion(
        prompt=prompt_lst,
        api_base=os.environ.get("LITELLM_API_BASE", "EMPTY"),
        api_key=os.environ.get("LITELLM_API_KEY", "EMPTY"),
        model="litellm_proxy/Qwen3-1.7B",
        temperature=0.6,
        top_p=0.95,
        max_tokens=64,
        n=ROLLOUT_NUM,
    )
    return response


def main(limit: int):
    ds = load_ds("prefixsliding/train_v1", "train", "math")
    if limit is not None:
        ds = ds.select(range(limit))
    problems = ds['problem']
    solutions = ds['solution']

    prompt_lst = []
    for p_mid in problems:
        prompt = P_START + GUESSABILITY_PROMPT_BOXED + p_mid + P_END
        prompt_lst.append(prompt)
    print(f"Total requests: {len(prompt_lst)} 🚀")

    # Get guessability
    outputs = make_requests(prompt_lst)

    # Get verifyability
    o_v = []
    for i in tqdm(range(len(prompt_lst))):
        answer_lst = []
        sub_ret = []
        for j in range(ROLLOUT_NUM):
            index = i*ROLLOUT_NUM + j
            answer_lst.append("\\boxed" + outputs.choices[index].text)
        
        # print(verify_math(answer_lst, solutions[i]))
        for k, v in enumerate(verify_math(answer_lst, solutions[i])):
            sub_ret.append((v[0], outputs.choices[i*ROLLOUT_NUM + k].text.split('\n')[0].strip("{}.")))
        o_v.append(sub_ret)

    # Write to dataset
    ds = ds.add_column("guessability", [json.dumps({MODEL: str(int(sum([y[0] for y in x if y[0] is not None]))) + "/8"}, ensure_ascii=False) for x in o_v])
    ds = ds.add_column("guessability_samples", [json.dumps({MODEL: [y[1] for y in x]}, ensure_ascii=False) for x in o_v])
    # ds.push_to_hub("prefixsliding/train_debug_qwen3_1B7")
    ds.to_json("train_debug_qwen3_1B7.jsonl", orient="records", lines=True, force_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.limit)