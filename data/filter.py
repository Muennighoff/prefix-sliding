import json
from datasets import load_dataset

ds = load_dataset("prefixsliding/train_v6", split="train")

mg = "Qwen/Qwen2.5-1.5B-Instruct"
mgalt = "Human"
md = "Qwen/Qwen3-1.7B"
mdalt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
mdalt2 = "Qwen/Qwen3-32B"

def filter_fn(x):
    if x['verifiability'] is False: return False
    if x["difficulty"] is None: return False
    if x["guessability"] is not None:
        g = json.loads(x["guessability"])
        if g.get(mg, '0/0')[0] != '0': return False # Can be guessed
        if g.get(mgalt, '0/0')[0] != '0': return False
    else:
        assert x['domain'] in ['code']
    d = json.loads(x["difficulty"])
    # Might be unsolvable
    if all(va == vb for v in d.values() for va, vb in [v.split("/")]): return False
    # Too easy
    if md in d:
        if d[md] in ['0/16', '1/16']: return False
    elif (mdalt in d) and (d[mdalt] in ['0/16', '1/16']): return False
    # For GR, the difficulty labels of their evals like 'deepseek-ai/DeepSeek-R1' are noisy
    # e.g. {'problem': 'Construct a triangle given the points where the median, angle bisector, and altitude drawn from one of its vertices intersect the circumcircle of the triangle.', 'solution': '\\text{Solution complete}', 'tests': None, 'domain': 'math', 'source': 'GeneralReasoning/GeneralThought-430K/Numina/NuminaMath', 'metadata': '{\'model_name\': [\'DeepSeek/DeepSeek-R1\'], \'verifier_score\': [1.0], \'task\': \'Math Olympiads\', \'question_url\': \'https://gr.inc/question/construct-a-triangle-given-the-points-where-the-me\', \'question_license\': \'Apache-2.0\', \'difficulty\': \'{"DeepSeek/DeepSeek-R1": "1/1"}\'}', 'guessability': '{"Qwen/Qwen2.5-1.5B-Instruct": "0/8"}', 'guessability_samples': '["Impossible", "The answer is not specified in the problem statement", "Impossible to construct such a triangle", "E", "The triangle is equilateral", "C", "Equilateral triangle", "E"]', 'verifiability': None, 'difficulty': '{"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "16/16", "Qwen/Qwen3-1.7B": "16/16", "deepseek-ai/DeepSeek-R1": "0/1"}'}
    # thus filter these out
    if x['source'].startswith("GeneralReasoning/GeneralThought-430K"):
        # subdict with trusted scores
        d = {k: v for k, v in d.items() if k in [md, mdalt, mdalt2]}
        if all(va == vb for v in d.values() for va, vb in [v.split("/")]):
            return False
    return True

dstv6f = dstv6.filter(filter_fn)
dstv6f.push_to_hub("prefixsliding/train_v6_filtered")

dstv6fm = dstv6f.filter(lambda x: x['domain'] in ['math'])
dstv6fm.push_to_hub("prefixsliding/train_v6_filtered_math")

guess_strings = [
    ["\nA:", "\nB:", "\nC:", "\nD:"],
    ["\nA.", "\nB.", "\nC.", "\nD."],
    ["\nA ", "\nB ", "\nC ", "\nD "],
    ["\nA)", "\nB)", "\nC)", "\nD)"],
    ["\n(A)", "\n(B)", "\n(C)", "\n(D)"],
    ["$\\text{(A)}", "$\\text{(B)}", "$\\text{(C)}", "$\\text{(D)}"],
    ["\n① ", "\n② ", "\n③ ", "\n④ "],
]

nonverif_strings = [
    ['Prove that'],
    ['prove that'],
    ['Show that'],
    # Could also split into separate problems but hard due to solution may only be for one
    ['\n(I) ', '\n(II) '],
    ['\n(Ⅰ) ', '\n(Ⅱ) '], 
    ['\n(1) ', '\n(2) '],
    ['\n1) ', '\n2) '],
    ['\n1. ', '\n2. '],
    ['\n$(1)$ ', '\n$(2)$ '],
    ['\n(a) ', '\n(b) '],
]

# Others that could be filtered out but not doing for now:
# "Does there exist "
# solution in "\text{No}", Yes...
# ['\na) ', '\nb) '],

def annotate(x):
    if any(all(s in x['problem'] for s in g) for g in guess_strings):
        if x['guessability'] is None:
            guessability = {}
        else:
            guessability = json.loads(x.get('guessability', '{}'))
        guessability['Human'] = '1/4'  # Assume human can guess
        x['guessability'] = json.dumps(guessability)
    if any(all(s in x['problem'] for s in n) for n in nonverif_strings):
        x['verifiability'] = False
    return x

dstv6 = dstv6.map(annotate, num_proc=8)
