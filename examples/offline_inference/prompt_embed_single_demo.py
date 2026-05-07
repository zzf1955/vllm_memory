# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end single prompt-embeds hidden-states demo.

This script generates one Hugging Face prompt-embeds ground truth sample, then
runs vLLM token-prompt and prompt-embeds paths against that sample.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_RUN_ROOT = Path("/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_single")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--layer-ids", default="all")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def run_command(command: list[str], *, gpu: str) -> str:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return completed.stdout


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    gt_prefix = args.run_root / "single_gt"
    work_dir = args.run_root / "vllm-single"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("===== Single Demo Config =====")
    print(f"model: {args.model}")
    print(f"gpu: {args.gpu}")
    print(f"seq_len: {args.seq_len}")
    print(f"layer_ids: {args.layer_ids}")
    print(f"run_root: {args.run_root}")

    print("\n===== Generate Single Ground Truth =====")
    run_command(
        [
            sys.executable,
            "examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py",
            "--model",
            args.model,
            "--seq-len",
            str(args.seq_len),
            "--device",
            "cuda",
            "--dtype",
            args.dtype,
            "--output",
            str(gt_prefix.with_suffix(".pth")),
        ],
        gpu=args.gpu,
    )

    print("\n===== vLLM Single Prompt-Embeds Test =====")
    output = run_command(
        [
            sys.executable,
            "examples/offline_inference/prompt_embed_extract_hidden_states.py",
            "--model",
            args.model,
            "--input-pth",
            str(gt_prefix.with_suffix(".input.pth")),
            "--output-pth",
            str(gt_prefix.with_suffix(".output.pth")),
            "--work-dir",
            str(work_dir),
            "--layer-ids",
            args.layer_ids,
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--dtype",
            args.dtype,
        ],
        gpu=args.gpu,
    )

    generated_ok = "Generated ids match: True" in output
    hidden_ok = (
        "Prompt-embeds vs token hidden states: {'allclose': True" in output
    )
    status = "PASS" if generated_ok and hidden_ok else "FAIL"

    print("\n===== Single Demo Summary =====")
    print(f"single_generated_ids_match: {generated_ok}")
    print(f"single_prompt_embeds_vs_token_hidden_states: {hidden_ok}")
    print(f"single_prompt: {gt_prefix.with_suffix('.prompt.txt')}")
    print(f"single_input: {gt_prefix.with_suffix('.input.pth')}")
    print(f"single_output: {gt_prefix.with_suffix('.output.pth')}")
    print(f"single_work_dir: {work_dir}")
    print(f"SINGLE_DEMO_RESULT: {status}")

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
