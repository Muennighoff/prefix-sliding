import datasets
import json
from functools import partial
from tqdm import tqdm

from simpleverify import verify_code, verify_generic, verify_math
from simpleverify import guessability, verifiability, difficulty
from simpleverify.data_filtering import guessability_boxed


### Guessability Annotations ###
ds = datasets.load_dataset("prefixsliding/train_v1", split="train")

ds_code = ds.filter(lambda x: x['domain'] == "code")
ds_math = ds.filter(lambda x: x['domain'] == "math")
ds_generic = ds.filter(lambda x: x['domain'] not in ["math", "code"])
ds_generic_crossword = ds_generic.filter(lambda x: x['domain'] == "crossword")
ds_generic_other = ds_generic.filter(lambda x: x['domain'] != "crossword")

m: str = "Qwen/Qwen2.5-1.5B-Instruct"
p_start: str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
p_end: str = "<|im_end|>\n<|im_start|>assistant\nThe answer is \\boxed" # Not including `{` as `{-` is one token and don't want to prevent model from generating -
n: int = 8
tp: int = 1

from vllm import LLM, SamplingParams
from simpleverify.prompts import GUESSABILITY_PROMPT, GUESSABILITY_PROMPT_BOXED, GUESSABILITY_PROMPT_CROSSWORD_BOXED, VERIFIABLE_PROMPT, VERIFIABLE_PROMPT_NOANSWER

p, a = ds_math['problem'], ds_math['solution']

m = LLM(m, tensor_parallel_size=tp)
s = SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=64)
o = m.generate([p_start + GUESSABILITY_PROMPT_BOXED + p_mid + p_end for p_mid in p], sampling_params=s)

o_v = [
    [
        (v[0], o[i].outputs[k].text.strip("{}."))
        for k, v in enumerate(
            verify_math(
                ["\\boxed" + o[i].outputs[j].text for j in range(n)],
                a[i]
            )
        )
    ]
    for i in tqdm(range(len(p)))
]
# Sometimes the above hangs in which case a for loop like below worked
# o_v = []
# for i in tqdm(range(len(p))):
#     outputs = []
#     verified = verify_math(
#         ["\\boxed" + o[i].outputs[j].text for j in range(n)],
#         a[i]
#     )
#     for k, v in enumerate(verified):
#         text = o[i].outputs[k].text.strip("}.")
#         outputs.append((v[0], text))
#     o_v.append(outputs)

ds_math = ds_math.add_column("guessability", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": str(int(sum([y[0] for y in x if y[0] is not None]))) + "/8"}, ensure_ascii=False) for x in o_v])
ds_math = ds_math.add_column("guessability_samples", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": [y[1] for y in x]}, ensure_ascii=False) for x in o_v])

# Check how many
# tmp = ds_math.filter(lambda x: json.loads(x['guessability'])['Qwen/Qwen2.5-1.5B-Instruct'].split("/")[0] != "0")

# Save quickly
ds_nomath = ds.filter(lambda x: x['domain'] != "math")
ds_nomath = ds_nomath.add_column("guessability", [None for _ in range(len(ds_nomath))])
ds_nomath = ds_nomath.add_column("guessability_samples", [None for _ in range(len(ds_nomath))])
ds = datasets.concatenate_datasets([ds_math, ds_nomath])
ds.push_to_hub("prefixsliding/train_v2")


p, a = ds_generic_crossword['problem'], ds_generic_crossword['solution']
o = m.generate([p_start + GUESSABILITY_PROMPT_CROSSWORD_BOXED + p_mid + p_end for p_mid in p], sampling_params=s)

from simpleverify.verify_generic import last_boxed_only_string, remove_boxed

o_v = []
for i in tqdm(range(len(p))):
    outputs = []
    for j in range(n):
        if box := last_boxed_only_string("\\boxed" + o[i].outputs[j].text):
            nobox = remove_boxed(box)
            outputs.append((a[i] == nobox.upper(), nobox))
        else:
            outputs.append((False, o[i].outputs[j].text.lstrip("{")))
    o_v.append(outputs)

ds_generic_crossword = ds_generic_crossword.add_column("guessability", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": str(int(sum([y[0] for y in x]))) + "/8"}, ensure_ascii=False) for x in o_v])
ds_generic_crossword = ds_generic_crossword.add_column("guessability_samples", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": [y[1] for y in x]}, ensure_ascii=False) for x in o_v])

# Check how many
# tmp = ds_generic_crossword.filter(lambda x: json.loads(x['guessability'])['Qwen/Qwen2.5-1.5B-Instruct'].split("/")[0] != "0")

# Save quickly
ds = datasets.load_dataset("prefixsliding/train_v2", split="train")
ds_no_crossword = ds.filter(lambda x: x['domain'] != "crossword")
ds = datasets.concatenate_datasets([ds_no_crossword, ds_generic_crossword])
ds.push_to_hub("prefixsliding/train_v2")

# Already verifiable
ds_generic_other = ds_no_crossword.filter(lambda x: x['domain'] not in ["math", "code"])
ds_generic_gr = ds_generic_other.filter(lambda x: x['source'].startswith('GeneralReasoning/GeneralThought-430K') and x['source'] != 'GeneralReasoning/GeneralThought-430K/FreedomIntelligence/medical-o1-verifiable-problem')

# 'explain' 20/20 were indeed unverifiable
# 'explain' removes 12409 samples
# for 'GeneralReasoning/GeneralThought-430K/FreedomIntelligence/medical-o1-verifiable-problem' only 2/5 with 'explain' in the problem are indeed unverifiable hence ignoring it
# 'discuss' 20/20 were indeed unverifiable
# 'discuss' removes an additional 7079 samples
# 'describe' 10/10 were indeed unverifiable
# 'describe' removes an additional 2305 samples
# 'prove ' 10/10 were indeed unverifiable
# 'prove ' removes an additional 1687 samples
# 'explanation' 10/10 were indeed unverifiable
# 'explanation' removes an additional 2875 samples
# 'show ' 10/10 were indeed unverifiable
# 'show ' removes an additional 2559 samples
# 'compare and contrast' 10/10 were indeed unverifiable
# 'compare and contrast' removes an additional 158 samples
# 'analyze' 9.5/10 were indeed unverifiable
# 'analyze' removes an additional 686 samples
# 'how d' 15/15 were indeed unverifiable
# 'how d' removes an additional 2780 samples
# 'why' 10/10 were indeed unverifiable
# 'why' removes an additional 703 samples
# 'derive' 2/5 were indeed unverifiable
# 'elaborate' 0/3 were indeed unverifiable
# 'provide' 3/5 were indeed unverifiable
# 'provide ' 8/10 were indeed unverifiable
# 'detailed' 10/10 were indeed unverifiable
# 'detailed' removes an additional 397 samples
# 'what are' 10/10 were indeed unverifiable
# 'what are' removes an additional 1400 samples
# 'how would' 6/6 were indeed unverifiable
# 'how would' removes an additional 210 samples
# 'justify 6/6 were indeed unverifiable
# 'justify' removes an additional 116 samples

def is_verifiable(x):
    # if 'explain' in x['problem'].lower():
    if any(y in x['problem'].lower() for y in ['explain', 'discuss', 'describe', 'prove ', 'explanation', 'show ', 'compare and contrast', 'analyze', 'how d', 'why', 'detailed', 'what are', 'how would', 'justify']):
        x['verifiable'] = False
    else:
        x['verifiable'] = None
    return x

ds_generic_gr = ds_generic_gr.map(is_verifiable)

# Save quickly
ds_nogr = ds.filter(lambda x: (not x['source'].startswith('GeneralReasoning/GeneralThought-430K')) or (x['domain'] in ["math", "code", "crossword"]))
ds_generic_medical = ds.filter(lambda x: x['source'] == 'GeneralReasoning/GeneralThought-430K/FreedomIntelligence/medical-o1-verifiable-problem')

ds_nogr = ds_nogr.add_column("verifiable", [None for _ in range(len(ds_nogr))])
ds_generic_medical = ds_generic_medical.add_column("verifiable", [None for _ in range(len(ds_generic_medical))])

ds = datasets.concatenate_datasets([ds_nogr, ds_generic_medical, ds_generic_gr])
ds.push_to_hub("prefixsliding/train_v2")

ds_generic_gr_v = ds_generic_gr.filter(lambda x: x['verifiable'] is None)
ds_generic_nogr = ds.filter(lambda x: not x['source'].startswith('GeneralReasoning/GeneralThought-430K') and x['domain'] not in ["math", "code", "crossword"])
ds_guess = datasets.concatenate_datasets([ds_generic_gr_v, ds_generic_medical, ds_generic_nogr])

p, a = ds_guess['problem'], ds_guess['solution']
o = m.generate([p_start + GUESSABILITY_PROMPT_BOXED + p_mid + p_end for p_mid in p], sampling_params=s)


# gpt-4.1-nano-2025-04-14 made 2 mistake in 15 * 8 samples
# gpt-4.1-mini-2025-04-14 made 1 mistake in 25 * 8 samples
o_v = []
for i in tqdm(range(len(p))):
    verified = verify_generic(
        ["\\boxed" + o[i].outputs[j].text for j in range(n)],
        a[i],
        m="gpt-4.1-mini-2025-04-14", # nano made some odd mistakes
    )
    o_v.append([(v[0], v[1]) for v in verified])


# for k in tqdm(range(len(p[i:]))):
#     verified = verify_generic(
#         ["\\boxed" + o[k].outputs[j].text for j in range(n)],
#         a[i],
#         m="gpt-4.1-mini-2025-04-14", # nano made some odd mistakes
#     )
#     o_v.append([(v[0], v[1]) for v in verified])

# for k in tqdm(range(len(p[i:]))):
#     verified = verify_generic(
#         ["\\boxed" + o[i+k].outputs[j].text for j in range(n)],
#         a[i+k],
#         m="gpt-4.1-mini-2025-04-14", # nano made some odd mistakes
#     )
#     o_v_fixed.append([(v[0], v[1]) for v in verified])



# p, a = ds_generic_other['problem'], ds_generic_other['solution']

# from simpleverify.verify_generic import ChatCompletionSampler, get_answer
# sampler = ChatCompletionSampler("gpt-4.1-mini-2025-04-14")
# r = []
# for p_str in p[5:15]:
#     p_str = VERIFIABLE_PROMPT_NOANSWER(p_str)
#     match = get_answer(sampler, prompt=p_str)[-3:].lower() == "yes"
#     r.append(match)


ds_guess = ds_guess.remove_columns(["guessability", "guessability_samples"])
ds_guess = ds_guess.add_column("guessability", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": str(int(sum([y[0] for y in x]))) + "/8"}, ensure_ascii=False) for x in o_v])
ds_guess = ds_guess.add_column("guessability_samples", [json.dumps({"Qwen/Qwen2.5-1.5B-Instruct": [y[1] for y in x]}, ensure_ascii=False) for x in o_v])

# Check how many
# tmp = ds_guess.filter(lambda x: json.loads(x['guessability'])['Qwen/Qwen2.5-1.5B-Instruct'].split("/")[0] != "0")

# Save quickly
ds_generic_gr_noverif = ds_generic_gr.filter(lambda x: x['verifiable'] is False)
ds_nogeneric = ds.filter(lambda x: x['domain'] in ["math", "code", "crossword"])

ds = datasets.concatenate_datasets([ds_nogeneric, ds_generic_gr_noverif, ds_guess])
ds.push_to_hub("prefixsliding/train_v3")

# Now already done in data.py
# import ast
# ds = ds.filter(lambda x: not(x['source'].startswith("GAIR/OlympicArena")) or (ast.literal_eval(x['metadata'])["answer_type"] in ["NV", "SET", "IN", "EX", "EQ", "TUP", "MPV", "MA"]))

ds = ds.rename_column("verifiable", "verifiability")
ds.push_to_hub("prefixsliding/train_v4")

# TODO: Coding guessability
# ds_code = ds.filter(lambda x: x['domain'] == "code")
# o = m.generate([p_start + GUESSABILITY_PROMPT + p_mid + p_end for p_mid in p], sampling_params=s)

### Difficulty ###

# External script added difficulty samples; merged it into train_v4 and uploaded as train_v5
# ds.push_to_hub("prefixsliding/train_v5")


# Verified all difficulty samples
import json
from simpleverify import verify_crossword, verify_generic, verify_math
from transformers import AutoTokenizer

# t = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-32B')
from simpleverify.verify_generic import last_boxed_only_string, remove_boxed, ANSWER_PATTERN
import re
from typing import List, Tuple, Union, Optional
def custom_clean_fn(x: str, sep: str = None, max_char: int = 2000) -> Tuple[str, bool]:
    """Quick cleanup before verifying"""
    is_valid, has_sep = True, False
    if sep is not None:
        if len(x_sep := x.split(sep)) > 1:
            has_sep = True
        x = x_sep[-1]
    # Small issue is that e.g. `The positive integers \( n \) are \( \boxed{1} \) and \( \boxed{9} \).` will be just a `9`
    if (box := last_boxed_only_string(x)) is not None:
        x = remove_boxed(box)
    # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
    elif (matches := re.findall(ANSWER_PATTERN, x, re.DOTALL)) != []:
        x = matches[-1] # Get the last match
    elif has_sep is False:
        is_valid = len(t.tokenize(x)) < 32768
    # Limit to last 2000 characters (4300 is the default max for string to int conversion (sys.get_int_max_str_digits))
    # anything before the final 2K chars is likely not the answer, saves some API credits, & makes it faster
    return x[-max_char:], is_valid

m = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' # "Qwen/Qwen3-1.7B" # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
m = 'Qwen/Qwen3-32B'
def verify_map(x):
    d = json.loads(x['difficulty_samples'])
    assert len(d) == 1, "Expected only one model in difficulty_samples"
    d = d[m][0]
    if x['domain'] == "math":
        r = verify_math(
            a=d,
            s=x['solution'],
            sep="</think>",
            fix_empty=True,
        )
    elif x['domain'] == "crossword":
        r = verify_crossword(
            a=d,
            s=x['solution'],
            sep="</think>",
            force_sep=True,  # Must have </think>
            tokenizer=t,  # Use the tokenizer to check length
            max_tokens=32768,  # Max tokens for the model
        )
    else:
        r = verify_generic(
            a=d,
            s=x['solution'],
            sep="</think>",
            m="gpt-4.1-mini-2025-04-14",
            clean_fn=custom_clean_fn,
            force_boxed=True,  # Only verify if clean_fn returns True for is_valid
        )
    x['difficulty_ordered'] = [y[0] for y in r]
    d = json.loads(x['difficulty']) if x['difficulty'] else {}
    n_wrong = str(int(len(r) - sum([y[0] for y in r]))) + "/" + str(len(r))
    if m in d:
        # If the model is already in the difficulty, we update it
        a, b = d[m].split('/')
        n_wrong = str(int(a) + int(n_wrong.split('/')[0])) + "/" + str(int(b) + len(r))
    else:
        d[m] = n_wrong
    x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x

import datasets
# ds = datasets.load_dataset("prefixsliding/Qwen3-32B-0701")['train']
# prefixsliding/DeepSeek-R1-Distill-Qwen-1.5-0530
ds = datasets.load_dataset("prefixsliding/DeepSeek-R1-Distill-Qwen-1.5-0530")['train']

ds_math = ds.filter(lambda x: x['domain'] == 'math')
ds_cross = ds.filter(lambda x: x['domain'] == 'crossword')
ds_code = ds.filter(lambda x: x['domain'] == 'code')
ds_generic = ds.filter(lambda x: x['domain'] not in ['math', 'crossword', 'code'])

ds_cross = ds_cross.map(verify_map, num_proc=32, desc="Verifying difficulty")
ds_math = ds_math.map(verify_map, num_proc=8, desc="Verifying difficulty")

# Due to API costs only apply to samples that will be used
mg = "Qwen/Qwen2.5-1.5B-Instruct"
def filter_fn(x):
    if x['verifiability'] is False: return False
    if x["guessability"] is None: return True
    g = json.loads(x["guessability"])
    if g[mg][0] != '0': return False
    return True

ds_generic_filter = ds_generic.filter(filter_fn)
ds_generic_rest = ds_generic.filter(lambda x: not filter_fn(x))
ds_generic_filter = ds_generic_filter.map(verify_map, num_proc=16, desc="Verifying difficulty")

# Alternatively, do it via for loop in case of error
o_gf = []
for i in tqdm(range(len(ds_generic_filter))):
    o_gf.append(verify_map(ds_generic_filter[i]))

ds_generic_filter = datasets.Dataset.from_list(o_gf)

from tqdm import tqdm
o_gf = []
for i in tqdm(range(len(ds_generic_filter.select(list(range(10)))))):
    o_gf.append(verify_map(ds_generic_filter[i]))

tmp = datasets.Dataset.from_list(o_gf)
dsv6.push_to_hub("prefixsliding/tmp")

missing = ds_generic_filter.select(list(range(i, len(ds_generic_filter))))
missing = missing.map(verify_map, num_proc=16, desc="Verifying difficulty (missing)")


dsqwen = datasets.concatenate_datasets([ds_math, ds_code, ds_cross, ds_generic_filter])


dsqwen.push_to_hub("prefixsliding/Qwen3-1.7B-0608")

dsdeepseek = dsdeepseek.remove_columns(["difficulty_ordered"])

def merge_difficulty(x):
    if x['difficulty'] is None:
        print("No difficulty for", x['problem'])
        return x
    elif x['difficulty_qwen'] is None:
        return x
    dss = json.loads(x['difficulty'])
    dq = json.loads(x['difficulty_qwen'])
    dss['Qwen/Qwen3-1.7B'] = dq['Qwen/Qwen3-1.7B']
    x['difficulty'] = json.dumps(dss, ensure_ascii=False)
    return x


merged_ds.push_to_hub("prefixsliding/train_v6")



dsv6 = datasets.concatenate_datasets([ds_math, ds_code, ds_cross, ds_generic_filter, ds_generic_rest])
dsv6 = dsv6.remove_columns(["difficulty_samples"])
dsv6.push_to_hub("prefixsliding/train_v6")




ds_generic_filter_half = ds_generic_filter.select(list(range(0, len(ds_generic_filter) // 2)))
ds_generic_filter_otherhalf = ds_generic_filter.select(list(range(len(ds_generic_filter) // 2, len(ds_generic_filter))))

ds_generic_filter_new = datasets.concatenate_datasets([ds_generic_filter_half, ds_generic_filter_otherhalf])


def verify_map(x):
    d = json.loads(x['difficulty_samples'])
    d = d[m][0]
    r = verify_generic(
        a=d,
        s=x['solution'],
        sep="</think>",
        m="gpt-4.1-mini-2025-04-14",
        clean_fn=custom_clean_fn,
        force_boxed=True,  # Only verify if clean_fn returns True for is_valid
    )
    x['difficulty_ordered'] = [y[0] for y in r]
    d = json.loads(x['difficulty']) if x['difficulty'] else {}
    n_wrong = str(int(len(r) - sum([y[0] for y in r]))) + "/" + str(len(r))
    if m in d:
        # If the model is already in the difficulty, we update it
        a, b = d[m].split('/')
        n_wrong = str(int(a) + int(n_wrong.split('/')[0])) + "/" + str(int(b) + len(r))
    else:
        d[m] = n_wrong
    x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x





def revert_diff(x):
    if x['source'].startswith('GeneralReasoning/GeneralThought-430K') and x['difficulty'] is not None:
        d = json.loads(x['difficulty'])
        for k, v in d.items():
            if 'Qwen/Qwen3-1.7B' == k:
                assert d['Qwen/Qwen3-1.7B'][-2:] == '16'
            elif 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' == k:
                assert d['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'][-2:] == '16'
            else:
                a,b = v.split('/')
                rev = str(int(a) - int(b)) + '/' + b
                print(f"Reverting {k} from {v} to {rev}")
                d[k] = rev
        x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x


rename_keys = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek/DeepSeek-R1-Zero": "deepseek-ai/DeepSeek-R1-Zero",
    "DeepSeek/deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "GAIR-NLP/LIMO": "GAIR/LIMO"
}

def rename(x):
    if x['difficulty'] is not None:
        d = json.loads(x['difficulty'])
        for k, v in list(d.items()):
            if k in rename_keys:
                new_k = rename_keys[k]
                if new_k in d:
                    print(f"Warning: {new_k} already exists in {d}, merging {k} into it")
                    a, b = d[new_k].split('/')
                    a2, b2 = v.split('/')
                    d[new_k] = str(int(a) + int(a2)) + '/' + str(int(b) + int(b2))
                else:
                    d[new_k] = v
                del d[k]
        x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x

        


# Rename AI-MO/NuminaMath-COT -> AI-MO/NuminaMath-1.5
def rename_source(x):
    if "AI-MO/NuminaMath-COT" in x['source']:
        x['source'] = x['source'].replace("AI-MO/NuminaMath-COT", "AI-MO/NuminaMath-1.5")
    return x

# Maybe not needed - kind of done by guessability and if it passes that then it is probably hard enough
# def filter_which_of(x):
#     # Filter out samples that contain 'which of' in the problem
#     return 'which of' in x['problem'].lower()


def rmv_diff_meta(x):
    if "GeneralReasoning/GeneralThought-430K" in x['source']:
        meta = json.loads(x['metadata'])
        meta.pop('difficulty', None)  # Remove difficulty key
        x['metadata'] = json.dumps(meta, ensure_ascii=False)
    return x

from tqdm import tqdm
ds_math_list = []
failed = []
z = 200000# i+1
z = 253251
for i in tqdm(range(z,len(ds_math))):
    try:
        ds_math_list.append(verify_map(ds_math[i]))
    except KeyboardInterrupt as e:
        print(f"Error verifying sample {i}: {e}")
        x = ds_math[i]
        x['difficuly_ordered'] = None
        ds_math_list.append(x)
        failed.append(i)

ds_math_2_25 = datasets.Dataset.from_list(ds_math_list)



m = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' # "Qwen/Qwen3-1.7B" # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
m = 'Qwen/Qwen3-32B'
def verify_map(x):
    d = json.loads(x['difficulty_samples'])
    assert len(d) == 1, "Expected only one model in difficulty_samples"
    d = d[m][0]
    try:
        r = verify_math(
            a=d,
            s=x['solution'],
            sep="</think>",
            fix_empty=True,
        )
    except (KeyboardInterrupt, Exception) as e:
        print("Excepting")
        rs = []
        for answer in d:
            try:
                r = verify_math(
                    a=[answer],
                    s=x['solution'],
                    sep="</think>",
                    fix_empty=True,
                )
                rs.append(r[0])
            except (KeyboardInterrupt, Exception) as e:
                print(f"Error verifying sample {x['problem']} with answer {answer[-500:]}")
                rs.append([0])
        print(rs)
        r = rs
    x['difficulty_ordered'] = [y[0] for y in r]
    d = json.loads(x['difficulty']) if x['difficulty'] else {}
    n_wrong = str(int(len(r) - sum([y[0] for y in r]))) + "/" + str(len(r))
    if m in d:
        # If the model is already in the difficulty, we update it
        a, b = d[m].split('/')
        n_wrong = str(int(a) + int(n_wrong.split('/')[0])) + "/" + str(int(b) + len(r))
    else:
        d[m] = n_wrong
    x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x

ds_math = ds_math.map(verify_map, num_proc=1, desc="Verifying difficulty")





from simpleverify import verify_code
ds = datasets.load_dataset("prefixsliding/train_v6_deepseek", split="train")
ds_code = ds.filter(lambda x: x['domain'] == 'code')
m = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
def verify_map(x):
    d = json.loads(x['difficulty_samples'])
    d = d[m][0]
    r = verify_code(
        a=d,
        s=json.loads(x['tests']),
        sep="</think>",
    )
    x['difficulty_ordered'] = [y[0] for y in r]
    d = json.loads(x['difficulty']) if x['difficulty'] else {}
    n_wrong = str(int(len(r) - sum([y[0] for y in r]))) + "/" + str(len(r))
    if m in d:
        # If the model is already in the difficulty, we update it
        a, b = d[m].split('/')
        n_wrong = str(int(a) + int(n_wrong.split('/')[0])) + "/" + str(int(b) + len(r))
    else:
        d[m] = n_wrong
    x['difficulty'] = json.dumps(d, ensure_ascii=False)
    return x

tmp = verify_map(ds_code[0])



tmp.push_to_hub("prefixsliding/tmpgeneric")



dsqwen32b = datasets.concatenate_datasets([ds_math, ds_cross, ds_generic_filter])
dsqwen.push_to_hub("prefixsliding/train_v6_qwen32b")




def merge_difficulty(x):
    if x['difficulty'] is None:
        print("No difficulty for", x['problem'])
        return x
    elif x['difficulty_qwen'] is None:
        return x
    dss = json.loads(x['difficulty'])
    dq = json.loads(x['difficulty_qwen'])
    dss['Qwen/Qwen3-32B'] = dq['Qwen/Qwen3-32B']
    x['difficulty'] = json.dumps(dss, ensure_ascii=False)
    return x


import pandas as pd

dstv6 = datasets.load_dataset("prefixsliding/train_v6")

# Convert to pandas DataFrames
df_qwen = ds.to_pandas()
df_tv6 = dstv6['train'].to_pandas()

# Rename Qwen's difficulty column to avoid conflict
df_qwen = df_qwen[['problem', 'difficulty']].rename(columns={'difficulty': 'difficulty_qwen'})

# Merge datasets
merged_df = df_tv6.merge(df_qwen, on='problem', how='left')



merged_ds = Dataset.from_pandas(merged_df, preserve_index=False)
merged_ds = merged_ds.map(merge_difficulty)
merged_ds.push_to_hub("prefixsliding/train_v6")

merged_ds[0]


# dsqwen = datasets.concatenate_datasets([ds_math, ds_code, ds_cross, ds_generic_filter])
dsqwenfix = datasets.concatenate_datasets([ds_mathx, ds_cross, tmp])
dsqwenfix.push_to_hub("prefixsliding/train_v6_qwen32b")


dsqwen.push_to_hub("prefixsliding/Qwen3-1.7B-0608")




### Reformat test cases from
# '...{"type": "stdin_stdout", "input": "4 2 4 2\\n", "output": "2\\n"}, {"type": "stdin_stdout", "input": "5 3 2 1\\n", "output": "0\\n"}, {"type": "stdin_stdout", "input": "155 2 10 5000\\n", "output": "343196694\\n"}, {"type": "stdin_stdout", "input": "10 2 3 4999\\n", "output": "0\\n"}, {"type": "stdin_stdout", "input": "50 5 4 81\\n", "output": "0\\n"}, {"type": "stdin_stdout", "input": "5000 574 4782 4798\\n", "output": "146295754\\n"}]'
# To '{"inputs": ["4 2 4 2\\n", "5 3 2 1\\n", "155 2 10 5000\\n", "10 2 3 4999\\n", "50 5 4 81\\n", "5000 574 4782 4798\\n"], "outputs": ["2\\n", "0\\n", "343196694\\n", "0\\n", "0\\n", "146295754\\n"]}'
def reformat_tests(x):
    if isinstance(x['tests'], str):
        tests = json.loads(x['tests'])
        if isinstance(tests, list):
            print("found")
            inputs = [test['input'] for test in tests]
            outputs = [test['output'] for test in tests]
            x['tests'] = json.dumps({"inputs": inputs, "outputs": outputs}, ensure_ascii=False)
    return x

# For quick testing
# verify_code(json.loads(tmp[8]['difficulty_samples'])['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'][0][0], json.loads(tmp[8]['tests']))

m = 'Qwen/Qwen3-32B'
def merge_difficulty(x):
    if x['difficulty_qwen'] is None:
        return x
    elif x['difficulty'] is None:
        x['difficulty'] = x['difficulty_qwen']
        # assert x['difficulty_ordered'] is None
        # x['difficulty_ordered'] = x['difficulty_ordered_code']
        return x
    # assert x['difficulty_ordered'] is None
    # x['difficulty_ordered'] = x['difficulty_ordered_code']
    dss = json.loads(x['difficulty'])
    dq = json.loads(x['difficulty_qwen'])
    assert dss[m] == dq[m], f"Got {dss} and {dq} for {x['problem']}"
    # dq = json.loads(x['difficulty_qwen'])
    # dss[m] = dq[m]
    # x['difficulty'] = json.dumps(dss, ensure_ascii=False)
    return x

ds = datasets.concatenate_datasets([ds, ds_code_new])
ds.push_to_hub("prefixsliding/train_v6_deepseek")


dstv6 = datasets.load_dataset("prefixsliding/train_v6", split="train")
