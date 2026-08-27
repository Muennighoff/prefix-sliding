#!/usr/bin/env python3
"""
Create and submit one sbatch job per configuration.

Usage:
  python submit_jobs.py --outdir ./job_results --jobsdir ./jobs --dryrun

"""
import subprocess
import shlex
from pathlib import Path
import argparse
import textwrap
import os

MODELS = [
    "Qwen/Qwen3-1.7B",
    # add others as needed
]


LENS = [
    #1024,
    #2048,
    #4096,
    8192,
#    12288,
    16384,
    32768,
    65536,
#    131072,
#    262144,
#    524288,
    #1048576,
]

METHODS = [
    #"Full",
    "SW",
#"SWR",
#    "Lastk",
#    "Summary",
]

SW_VALUES = [
    # 512,
   #1024,
   #2048,
   4096,
    # 8192,
   #16384,
]



def sanitize_name(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_")

def make_sbatch_script(model, l, me, c, job_script_path: Path, results_dir: Path, python_path="python"):
    model_s = sanitize_name(model)
    job_name = f"{model_s}_L{l}_{me}_{c}"
    output_path = job_script_path.parent / f"{job_name}.out"
    results_dir.mkdir(parents=True, exist_ok=True)

    slurm_header = textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --job-name={job_name}
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:1
    #SBATCH --time=7-00:00:00
    #SBATCH --output={output_path}
    #SBATCH --exclusive
    #SBATCH --mem=0
    """)

    cmd = f"{python_path} run_single_config.py --model {shlex.quote(model)} --length {l} --method {me} --c {c} --outdir {shlex.quote(str(results_dir.resolve()))}"
    # Add --texts_per_batch 1024 for more precision
    body = f"conda activate ps\n\n{cmd}\n"
    return slurm_header + "\n" + body

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./job_results")
    parser.add_argument("--jobsdir", default="./jobs")
    parser.add_argument("--dryrun", action="store_true", help="do not actually sbatch, just write scripts")
    parser.add_argument("--python", default="python", help="python executable for sbatch script (on compute node)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    jobsdir = Path(args.jobsdir)
    jobsdir.mkdir(parents=True, exist_ok=True)

    submitted = []

    for m in MODELS:
        for l in LENS:
            for me in METHODS:
                cfg_loop = [''] if me == 'Full' else SW_VALUES
                for c in cfg_loop:
                    # skip invalid configs
                    if isinstance(c, int) and (c >= l):
                        print(f"Skipping c >= l: {m} l={l} me={me} c={c}")
                        continue
                    c_val = 0 if c == '' else c
                    job_script_path = jobsdir / f"{sanitize_name(m)}_L{l}_{me}_{c_val}.sh"
                    script_text = make_sbatch_script(m, l, me, c_val, job_script_path, outdir, python_path=args.python)
                    job_script_path.write_text(script_text)
                    job_script_path.chmod(0o755)
                    if args.dryrun:
                        print(f"[DRYRUN] wrote {job_script_path}")
                    else:
                        print(f"Submitting {job_script_path} ...")
                        res = subprocess.run(["sbatch", str(job_script_path)], capture_output=True, text=True)
                        if res.returncode != 0:
                            print(f"Failed to submit {job_script_path}: {res.stderr}")
                        else:
                            print(f"sbatch output: {res.stdout.strip()}")
                            submitted.append((job_script_path, res.stdout.strip()))

    print(f"Created {len(list(jobsdir.glob('*.sh')))} job scripts. Submitted {len(submitted)} (if not dryrun).")

if __name__ == "__main__":
    main()
