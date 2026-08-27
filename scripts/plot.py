### Fig2 ###
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt

model_path = "Qwen/Qwen3-1.7B"
FONTSIZE = 16.5

# load model and tokenizer
config = AutoConfig.from_pretrained(model_path)
config._attn_implementation = "eager" # use vanilla attention to return attention weights
kwargs = {"torch_dtype": torch.float16, "device_map": "auto"} # use float16 to allow numpy conversion

model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path)

import json
with open('res/Qwen3-1.7B/samples_aime25_nofigures_tok131072_agg64_2025-10-05T02-20-11.454987.jsonl', 'r') as f:
    res = [json.loads(line) for line in f]

from scipy.ndimage import gaussian_filter1d

txt = res[2]['arguments']['gen_args_0']['arg_0'] + res[2]['filtered_resps'][0][40]
# Only focus on pure reasoning
txt = txt.split("</think>")[0]
print(f"Text len: {len(txt)}")
inputs = tokenizer(txt, return_tensors="pt")['input_ids'].to(model.device)
print("* Generating ...")
with torch.no_grad():
    attention_scores = model(inputs, output_attentions=True)['attentions'] # a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]

attention_scores = [attention_scores_layer.detach().cpu() for attention_scores_layer in attention_scores]
p = torch.stack([attention_scores[l][0].mean(dim=0)[-1] for l in range(len(attention_scores))], dim=0).mean(dim=0)

sigma = 50  # controls smoothing strength
p_smooth = gaussian_filter1d(p.float(), sigma=sigma)

prompt_len = len(tokenizer(res[2]['arguments']['gen_args_0']['arg_0']).input_ids)

fig, ax = plt.subplots(figsize=(12, 3), ncols=1)
ax.plot(p_smooth, lw=6, label=f"Gaussian smoothed (σ={sigma})", color="#E90909")
ax.plot(p_smooth[:prompt_len], lw=6, label=f"Gaussian smoothed (σ={sigma})", color="#0077B6")
ax.plot(list(range(len(p_smooth)-1024, len(p_smooth))), p_smooth[-1024:], lw=6, label=f"Gaussian smoothed (σ={sigma})", color="#0077B6")
ax.set_xlabel("Sequence length", fontsize=FONTSIZE)
ax.set_ylabel("Attention probability", fontsize=FONTSIZE)
plt.tick_params(axis="both", labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
plt.savefig("fig2.jpg", dpi=300, bbox_inches="tight")