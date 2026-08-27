#!/usr/bin/env python3
"""
Run a single config and write a per-job result JSON.

Usage:
  python run_single_config.py --model "Qwen/Qwen3-1.7B" --length 16384 --method SW --c 512 --outdir ./job_results
"""
import argparse
import json
import os
import time
import gc
import traceback
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--length", type=int, required=True, help="target length (min_tokens, max_tokens)")
    parser.add_argument("--method", required=True, choices=["Full", "SW", "SWR","Lastk", "Summary"], help="method")
    parser.add_argument("--c", type=int, default=0, help="sliding window / chunk size (0 if N/A)")
    parser.add_argument("--outdir", default="./job_results", help="where to write json result")
    parser.add_argument("--texts_per_batch", type=int, default=1, help="how many prompts to batch")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    job_name = f"{args.model.replace('/', '__')}_L{args.length}_{args.method}_{args.c}"
    out_file_tmp = outdir / f"{job_name}.json.tmp"
    out_file = outdir / f"{job_name}.json"

    result = {
        "model": args.model,
        "length": args.length,
        "method": args.method,
        "c": args.c,
        "start_ts": time.time(),
        "success": False,
        "payload": None,
    }

    try:
        # if SW -> set HF overrides and env
        hf_overrides = {"max_position_embeddings": args.length + 32}
        if args.method == "SW":
            hf_overrides = {"use_sliding_window": True, "sliding_window": args.c, "max_position_embeddings": args.length + 32}
            os.environ.update({"SWF": str(args.c)})
        elif args.method == "SWR":
            hf_overrides = {"use_sliding_window": True, "sliding_window": args.c, "max_position_embeddings": args.length + 32}

        # import inside run so slurm python environment can be separate
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        prompt = "Prime factorize 806917567."
        SYSTEM_PROMPT6 = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "pass", "description": "Passes the task to the next model to solve the problem from where you left off. Useful when thinking gets too long.", "parameters": {"type": "object", "properties": {"context": {"type": "string", "description": "Information to pass to the next model. E.g., ideas already tried, key results, next steps to perform..."}}, "required": ["context"]}}}
</tools>
"""
        if args.method == "Summary":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT6},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text += "<think>\n"

        s = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            min_tokens=args.length,
            max_tokens=args.length,
        )
        model = LLM(args.model, hf_overrides=hf_overrides)

        texts = [text] * args.texts_per_batch

        if args.method in ["Full", "SW", "SWR"]:
            start_time = time.time()
            output = model.generate(texts, sampling_params=s)
        elif args.method in ["Lastk", "Summary"]:
            # try to reuse your original tool-based generation if available
            try:
                from trl.trainer.grpo_config import GRPOConfig
                from trl.tools.parallel_tool_utils import generate_with_tool_batch
            except Exception as e:
                raise RuntimeError(f"Required trl.tools.parallel_tool_utils not available: {e}")

            tool_args = GRPOConfig(
                vllm_mode="colocate",
                temperature=0,
                top_p=1.0,
                max_completion_length=args.c,
                eos_token=tokenizer.eos_token,
                result_tokens=["<tool_response>", "</tool_response>"],
                saving_tokens=["<context>", "</context>"],
            )
            max_saving = (args.length // max(1, args.c)) - 1
            print(max_saving, args.c, args.length)
            start_time = time.time()
            _, x, s, y = generate_with_tool_batch(
                prompts=texts,
                args=tool_args,
                llm=model,
                tokenizer=tokenizer,
                budget_force_saving=256 if args.method == "Summary" else 0,
                max_saving=max_saving,
                saving_prompt=os.getenv("SAVING_PROMPT", "context_system") if args.method == 'Summary' else None,
                use_max=True,
                save_last_k=256 if args.method == "Lastk" else 0,
                min_max=True,
            )
            # generate_with_tool_batch in original code returned nothing (side-effect), so we fall back:
            output = []  # placeholder - but we'll record timing

        end_time = time.time()
        print(end_time - start_time)

        # compute tokens per second if possible
        toks = []
        if isinstance(output, list) and len(output) > 0:
            toks = [len(o.outputs[0].token_ids) for o in output]
            toks_per_s = sum(toks) / (end_time - start_time)
        else:
            toks_per_s = sum([x['generated_tokens'] for x in s]) / (end_time - start_time)

        # cleanup
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        result.update({
            "end_ts": time.time(),
            "duration_s": end_time - start_time,
            "success": True,
            "payload": {
                "toks_per_s": toks_per_s,
                "toks_reported": toks[:10] if toks else None
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        result.update({
            "end_ts": time.time(),
            "duration_s": time.time() - result["start_ts"],
            "success": False,
            "payload": {
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": tb
            }
        })

    # write atomically
    with open(out_file_tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(out_file_tmp, out_file)
    print(f"WROTE {out_file}")
    return 0

if __name__ == "__main__":
    main()
