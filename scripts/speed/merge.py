#!/usr/bin/env python3
"""
Merge individual per-job JSON files created by run_single_config.py into a single results.json
with structure:
{ model: { length: { method: value_or_dict } } }

Usage:
  python merge.py --in_dir ./job_results --out_file ./results.json
"""
import json
from pathlib import Path
import argparse

def load_job_file(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"failed to load {p}: {e}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="./job_results")
    parser.add_argument("--out_file", default="./results.json")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))

    results = {}
    missing = []
    for f in files:
        data = load_job_file(f)
        # If unexpected format, store under 'errors'
        if not isinstance(data, dict) or "model" not in data:
            missing.append(str(f))
            continue

        model = data.get("model")
        length = int(data.get("length"))
        method = data.get("method")
        c = data.get("c")

        if model not in results:
            results[model] = {}
        if str(length) not in results[model]:
            results[model][str(length)] = {}

        # unify representation:
        if method in ["SW", "SWR", "Lastk", "Summary"]:
            # ensure nested dict
            if method not in results[model][str(length)]:
                results[model][str(length)][method] = {}
            # only store toks_per_s if success else store error
            if data.get("success"):
                toks = data["payload"].get("toks_per_s")
                results[model][str(length)][method][str(c)] = toks
            else:
                results[model][str(length)][method][str(c)] = {"error": data["payload"]}
        else:
            # for Full/Lastk/Summary: store directly
            if data.get("success"):
                toks = data["payload"].get("toks_per_s")
                results[model][str(length)][method] = toks
            else:
                results[model][str(length)][method] = {"error": data["payload"]}

    out = {
        "generated_at": __import__("time").ctime(),
        "results": results,
        "skipped_files": missing,
        "source_folder": str(in_dir)
    }

    with open(args.out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote merged results to {args.out_file}. Found {len(files)} job files. Skipped {len(missing)} files.")

if __name__ == "__main__":
    main()
