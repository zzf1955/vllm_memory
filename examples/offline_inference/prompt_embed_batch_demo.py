# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end batch prompt-embeds hidden-states demo.

This script creates two batch references:
1. single GT: each prompt embedding is run alone with LLM.generate.
2. batch GT: all prompt embeddings are run together with LLM.generate.

Then it runs AsyncLLM batch prompt-embeds requests and compares every output
against both references. The pass/fail result is based on AsyncLLM vs batch GT.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_INPUT_PTH = Path("/disk_n/zzf/tmp/qwen3_gt_gpu.input.pth")
DEFAULT_RUN_ROOT = Path("/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-pth", type=Path, default=DEFAULT_INPUT_PTH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--layer-ids", default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--variant-scale-step", type=float, default=0.01)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def run_command(command: list[str], *, gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def sorted_gt_paths(path: Path) -> list[Path]:
    return sorted(
        path.glob("*.safetensors"),
        key=lambda file_path: int(file_path.name.split("-", 1)[0]),
    )


def sorted_async_paths(path: Path) -> dict[int, dict[int, Path]]:
    by_round: dict[int, dict[int, Path]] = {}
    for file_path in sorted(path.glob("*.safetensors")):
        parts = file_path.name.split("-")
        round_idx = int(parts[1][1:])
        batch_idx = int(parts[2][1:])
        by_round.setdefault(round_idx, {})[batch_idx] = file_path
    return by_round


def load_hidden_states(path: Path) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("hidden_states").float()


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    diff = (actual - expected).abs()
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "shape": tuple(actual.shape),
    }


def summarize(compares: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(compares),
        "failed": sum(0 if item["allclose"] else 1 for item in compares),
        "max_abs": max((item["max_abs"] for item in compares), default=0.0),
        "mean_abs_avg": (
            sum(item["mean_abs"] for item in compares) / len(compares)
            if compares
            else 0.0
        ),
    }


def print_compare(prefix: str, compare: dict[str, Any]) -> None:
    status = "PASS" if compare["allclose"] else "FAIL"
    print(
        f"{prefix} status={status} max_abs={compare['max_abs']} "
        f"mean_abs={compare['mean_abs']} shape={compare['shape']}"
    )


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if not args.input_pth.exists():
        raise FileNotFoundError(
            f"Missing input prompt embeddings: {args.input_pth}. "
            "Run prompt_embed_single_demo.py first or pass --input-pth."
        )

    print("===== Batch Demo Config =====")
    print(f"model: {args.model}")
    print(f"gpu: {args.gpu}")
    print(f"input_pth: {args.input_pth}")
    print(f"layer_ids: {args.layer_ids}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_rounds: {args.num_rounds}")
    print(f"run_root: {args.run_root}")

    single_gt_dir = args.run_root / "llm-single-ground-truth"
    batch_gt_dir = args.run_root / "llm-batch-ground-truth"
    async_dir = args.run_root / "async-prompt-embeds-batch-only"
    for path in (single_gt_dir, batch_gt_dir, async_dir):
        if path.exists():
            shutil.rmtree(path)

    print("\n===== Generate Single GT and Batch GT =====")
    run_command(
        [
            sys.executable,
            "examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py",
            "--model",
            args.model,
            "--input-pth",
            str(args.input_pth),
            "--work-dir",
            str(args.run_root),
            "--layer-ids",
            args.layer_ids,
            "--batch-size",
            str(args.batch_size),
            "--num-rounds",
            "1",
            "--variant-scale-step",
            str(args.variant_scale_step),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--dtype",
            args.dtype,
            "--mode",
            "gt-only",
        ],
        gpu=args.gpu,
    )

    print("\n===== Run AsyncLLM Batch Test =====")
    run_command(
        [
            sys.executable,
            "examples/offline_inference/prompt_embed_async_batch_only.py",
            "--model",
            args.model,
            "--input-pth",
            str(args.input_pth),
            "--work-dir",
            str(args.run_root),
            "--layer-ids",
            args.layer_ids,
            "--batch-size",
            str(args.batch_size),
            "--num-rounds",
            str(args.num_rounds),
            "--variant-scale-step",
            str(args.variant_scale_step),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--dtype",
            args.dtype,
        ],
        gpu=args.gpu,
    )

    single_paths = sorted_gt_paths(single_gt_dir)
    batch_paths = sorted_gt_paths(batch_gt_dir)
    async_paths = sorted_async_paths(async_dir)
    if len(single_paths) != args.batch_size or len(batch_paths) != args.batch_size:
        raise RuntimeError(
            f"Expected {args.batch_size} GT files, got "
            f"single={len(single_paths)} batch={len(batch_paths)}."
        )
    expected_async_count = args.batch_size * args.num_rounds
    actual_async_count = sum(len(items) for items in async_paths.values())
    if actual_async_count != expected_async_count:
        raise RuntimeError(
            f"Expected {expected_async_count} async files, got {actual_async_count}."
        )

    print("\n===== Compare Batch GT vs Single GT =====")
    single_vs_batch = []
    for batch_idx in range(args.batch_size):
        compare = compare_tensors(
            load_hidden_states(batch_paths[batch_idx]),
            load_hidden_states(single_paths[batch_idx]),
        )
        single_vs_batch.append(compare)
        print_compare(f"sample=b{batch_idx:03d} batch_gt_vs_single_gt", compare)

    print("\n===== Compare Async Batch vs GT =====")
    async_vs_batch = []
    async_vs_single = []
    for round_idx, items in sorted(async_paths.items()):
        for batch_idx in range(args.batch_size):
            hidden_states = load_hidden_states(items[batch_idx])
            batch_compare = compare_tensors(
                hidden_states, load_hidden_states(batch_paths[batch_idx])
            )
            single_compare = compare_tensors(
                hidden_states, load_hidden_states(single_paths[batch_idx])
            )
            async_vs_batch.append(batch_compare)
            async_vs_single.append(single_compare)
            print_compare(
                f"round={round_idx:03d} sample=b{batch_idx:03d} "
                "async_vs_batch_gt",
                batch_compare,
            )
            print_compare(
                f"round={round_idx:03d} sample=b{batch_idx:03d} "
                "async_vs_single_gt",
                single_compare,
            )

    single_vs_batch_summary = summarize(single_vs_batch)
    async_vs_batch_summary = summarize(async_vs_batch)
    async_vs_single_summary = summarize(async_vs_single)
    result = "PASS" if async_vs_batch_summary["failed"] == 0 else "FAIL"

    print("\n===== Batch Demo Summary =====")
    print(f"single_vs_batch_summary: {single_vs_batch_summary}")
    print(f"async_vs_batch_summary: {async_vs_batch_summary}")
    print(f"async_vs_single_summary: {async_vs_single_summary}")
    print(f"batch_work_dir: {args.run_root}")
    print(f"BATCH_DEMO_RESULT: {result}")

    if result != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
