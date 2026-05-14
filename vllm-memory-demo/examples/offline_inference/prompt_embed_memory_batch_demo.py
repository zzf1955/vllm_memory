# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare [random emb][prompt emb][random emb] single and batch outputs."""

import argparse
import asyncio
import gc
import tempfile
from pathlib import Path
from typing import Any

import torch

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

from prompt_embed_extract_hidden_states import (
    build_kv_transfer_config,
    build_speculative_config,
    get_sibling_prompt_path,
    load_input,
    load_prompt_text,
    parse_layer_ids,
    read_extracted_hidden_states,
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-pth", type=Path, required=True)
    parser.add_argument("--prompt-txt", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--layer-ids", default="all")
    parser.add_argument("--mem-len", type=int, default=8)
    parser.add_argument("--mem-seed", type=int, default=1234)
    parser.add_argument("--mem-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--variant-scale-step", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def make_mem_tensors(
    *,
    mem_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mem_len <= 0:
        raise ValueError("--mem-len must be positive.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    prefix = torch.randn(
        (mem_len, hidden_size), generator=generator, dtype=torch.float32
    )
    suffix = torch.randn(
        (mem_len, hidden_size), generator=generator, dtype=torch.float32
    )
    return (prefix * scale).to(dtype), (suffix * scale).to(dtype)


def make_full_prompt_embeds(
    *,
    mem_prefix: torch.Tensor,
    prompt_embeds: torch.Tensor,
    mem_suffix: torch.Tensor,
    batch_size: int,
    variant_scale_step: float,
) -> list[torch.Tensor]:
    full_prompts = []
    for batch_idx in range(batch_size):
        scale = 1.0 + batch_idx * variant_scale_step
        prompt_variant = prompt_embeds * scale
        full_prompts.append(
            torch.cat([mem_prefix, prompt_variant, mem_suffix], dim=0).contiguous()
        )
    return full_prompts


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "min_abs": float(diff.min().item()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "shape": tuple(actual.shape),
    }


def summarize_compares(compares: list[dict[str, Any]]) -> dict[str, Any]:
    failed = sum(
        0 if item["allclose"] and item["generated_ids_match"] else 1
        for item in compares
    )
    return {
        "status": "PASS" if failed == 0 else "FAIL",
        "count": len(compares),
        "failed": failed,
        "min_abs": min((item["min_abs"] for item in compares), default=0.0),
        "max_abs": max((item["max_abs"] for item in compares), default=0.0),
        "mean_abs_avg": (
            sum(item["mean_abs"] for item in compares) / len(compares)
            if compares
            else 0.0
        ),
        "shape": compares[0]["shape"] if compares else None,
    }


def print_compare(name: str, compare: dict[str, Any]) -> None:
    ok = compare["allclose"] and compare["generated_ids_match"]
    status = "PASS" if ok else "FAIL"
    print(
        f"{name} status={status} min_abs={compare['min_abs']} "
        f"max_abs={compare['max_abs']} mean_abs={compare['mean_abs']} "
        f"shape={compare['shape']} "
        f"generated_ids_match={compare['generated_ids_match']}"
    )


def collect_llm_outputs(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    full_prompt_embeds: list[torch.Tensor],
    storage_path: Path,
    layer_ids: list[int],
    batched: bool,
) -> list[dict[str, Any]]:
    llm = LLM(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_prompt_embeds=True,
        speculative_config=build_speculative_config(layer_ids),
        kv_transfer_config=build_kv_transfer_config(storage_path),
    )
    try:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            seed=0,
        )
        prompts = [
            {
                "prompt": prompt_text,
                "prompt_embeds": embeds,
            }
            for embeds in full_prompt_embeds
        ]
        if batched:
            outputs = llm.generate(prompts, sampling_params)
        else:
            outputs = []
            for prompt in prompts:
                one_output = llm.generate(prompt, sampling_params)
                if len(one_output) != 1:
                    raise RuntimeError(
                        f"Expected one single output, got {len(one_output)}."
                    )
                outputs.append(one_output[0])

        if len(outputs) != len(full_prompt_embeds):
            raise RuntimeError(
                f"Expected {len(full_prompt_embeds)} outputs, got {len(outputs)}."
            )
        return [output_to_record(output) for output in outputs]
    finally:
        if hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def output_to_record(output: RequestOutput) -> dict[str, Any]:
    tensors = read_extracted_hidden_states(output)
    return {
        "request_id": output.request_id,
        "generated_ids": output.outputs[0].token_ids,
        "hidden_states": tensors["hidden_states"],
        "token_ids": tensors["token_ids"],
        "hidden_states_path": output.kv_transfer_params["hidden_states_path"],
    }


async def collect_one_async(
    engine: AsyncLLM,
    *,
    request_id: str,
    prompt_text: str,
    prompt_embeds: torch.Tensor,
    sampling_params: SamplingParams,
) -> RequestOutput:
    final_output = None
    prompt: dict[str, Any] = {
        "prompt": prompt_text,
        "prompt_embeds": prompt_embeds,
    }
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        final_output = output
    if final_output is None:
        raise RuntimeError(f"Request {request_id} did not produce an output.")
    return final_output


async def collect_async_outputs(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    full_prompt_embeds: list[torch.Tensor],
    storage_path: Path,
    layer_ids: list[int],
    mode: str,
) -> list[dict[str, Any]]:
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_prompt_embeds=True,
        speculative_config=build_speculative_config(layer_ids),
        kv_transfer_config=build_kv_transfer_config(storage_path),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=0,
    )
    try:
        if mode == "single":
            outputs = []
            for batch_idx, embeds in enumerate(full_prompt_embeds):
                outputs.append(
                    await collect_one_async(
                        engine,
                        request_id=f"memory-single-b{batch_idx:03d}",
                        prompt_text=prompt_text,
                        prompt_embeds=embeds.clone(),
                        sampling_params=sampling_params,
                    )
                )
            return [output_to_record(output) for output in outputs]

        records = []
        for round_idx in range(args.num_rounds):
            tasks = []
            for batch_idx, embeds in enumerate(full_prompt_embeds):
                tasks.append(
                    asyncio.create_task(
                        collect_one_async(
                            engine,
                            request_id=(
                                f"memory-batch-r{round_idx:03d}-b{batch_idx:03d}"
                            ),
                            prompt_text=prompt_text,
                            prompt_embeds=embeds.clone(),
                            sampling_params=sampling_params,
                        )
                    )
                )
            outputs = await asyncio.gather(*tasks)
            records.extend(output_to_record(output) for output in outputs)
        return records
    finally:
        engine.shutdown()


def compare_records(
    *,
    actual: dict[str, Any],
    expected: dict[str, Any],
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    if tuple(actual["hidden_states"].shape) != expected_shape:
        raise AssertionError(
            f"Unexpected hidden_states shape for {actual['request_id']}: "
            f"{tuple(actual['hidden_states'].shape)} != {expected_shape}"
        )
    if actual["token_ids"].shape[0] != expected_shape[0]:
        raise AssertionError(
            f"Unexpected token_ids length for {actual['request_id']}: "
            f"{actual['token_ids'].shape[0]} != {expected_shape[0]}"
        )
    compare = compare_tensors(actual["hidden_states"], expected["hidden_states"])
    compare["generated_ids_match"] = (
        actual["generated_ids"] == expected["generated_ids"]
    )
    return compare


async def main_async(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_rounds <= 0:
        raise ValueError("--num-rounds must be positive.")

    input_data = load_input(args.input_pth)
    prompt_embeds = input_data["prompt_embeds"]
    metadata = input_data["metadata"]
    hidden_size = int(metadata["hidden_size"])
    prompt_len = len(prompt_embeds)
    layer_ids = parse_layer_ids(args.layer_ids, metadata)
    full_prompt_len = args.mem_len + prompt_len + args.mem_len

    if args.max_model_len < full_prompt_len + args.max_tokens:
        raise ValueError(
            "--max-model-len must be at least full prompt length + max tokens: "
            f"{full_prompt_len + args.max_tokens}"
        )

    prompt_path = args.prompt_txt or get_sibling_prompt_path(args.input_pth)
    prompt_text = load_prompt_text(prompt_path)
    mem_prefix, mem_suffix = make_mem_tensors(
        mem_len=args.mem_len,
        hidden_size=hidden_size,
        dtype=prompt_embeds.dtype,
        seed=args.mem_seed,
        scale=args.mem_scale,
    )
    full_prompt_embeds = make_full_prompt_embeds(
        mem_prefix=mem_prefix,
        prompt_embeds=prompt_embeds,
        mem_suffix=mem_suffix,
        batch_size=args.batch_size,
        variant_scale_step=args.variant_scale_step,
    )

    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vllm-prompt-embed-memory-"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    single_gt_dir = work_dir / "llm-single-memory-ground-truth"
    batch_gt_dir = work_dir / "llm-batch-memory-ground-truth"
    async_single_dir = work_dir / "async-single-memory"
    async_batch_dir = work_dir / "async-batch-memory"
    for path in (single_gt_dir, batch_gt_dir, async_single_dir, async_batch_dir):
        path.mkdir(parents=True, exist_ok=True)

    expected_shape = (full_prompt_len, len(layer_ids), hidden_size)

    print("===== Memory Prompt Demo Config =====")
    print("input_sequence: <random emb><prompt emb><random emb>")
    print(f"random_prefix_shape: {tuple(mem_prefix.shape)}")
    print(f"prompt_emb_shape: {tuple(prompt_embeds.shape)}")
    print(f"random_suffix_shape: {tuple(mem_suffix.shape)}")
    print(f"full_prompt_emb_shape: {tuple(full_prompt_embeds[0].shape)}")
    print(f"expected_output_shape: {expected_shape}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_rounds: {args.num_rounds}")
    print(f"layer_ids: {layer_ids}")
    print(f"work_dir: {work_dir}")

    print("\n===== Collect LLM Single GT =====")
    single_gt = collect_llm_outputs(
        args=args,
        prompt_text=prompt_text,
        full_prompt_embeds=full_prompt_embeds,
        storage_path=single_gt_dir,
        layer_ids=layer_ids,
        batched=False,
    )

    print("\n===== Run AsyncLLM Single Test =====")
    async_single = await collect_async_outputs(
        args=args,
        prompt_text=prompt_text,
        full_prompt_embeds=full_prompt_embeds,
        storage_path=async_single_dir,
        layer_ids=layer_ids,
        mode="single",
    )

    print("\n===== Collect LLM Batch GT =====")
    batch_gt = collect_llm_outputs(
        args=args,
        prompt_text=prompt_text,
        full_prompt_embeds=full_prompt_embeds,
        storage_path=batch_gt_dir,
        layer_ids=layer_ids,
        batched=True,
    )

    print("\n===== Run AsyncLLM Batch Test =====")
    async_batch = await collect_async_outputs(
        args=args,
        prompt_text=prompt_text,
        full_prompt_embeds=full_prompt_embeds,
        storage_path=async_batch_dir,
        layer_ids=layer_ids,
        mode="batch",
    )

    print("\n===== Compare Outputs =====")
    single_compares = []
    for batch_idx, (actual, expected) in enumerate(zip(async_single, single_gt)):
        compare = compare_records(
            actual=actual,
            expected=expected,
            expected_shape=expected_shape,
        )
        single_compares.append(compare)
        print_compare(f"single_vs_single sample=b{batch_idx:03d}", compare)

    batch_compares = []
    for idx, actual in enumerate(async_batch):
        batch_idx = idx % args.batch_size
        compare = compare_records(
            actual=actual,
            expected=batch_gt[batch_idx],
            expected_shape=expected_shape,
        )
        batch_compares.append(compare)
        round_idx = idx // args.batch_size
        print_compare(
            f"batch_vs_batch round={round_idx:03d} sample=b{batch_idx:03d}",
            compare,
        )

    single_summary = summarize_compares(single_compares)
    batch_summary = summarize_compares(batch_compares)
    result = (
        "PASS"
        if single_summary["status"] == "PASS" and batch_summary["status"] == "PASS"
        else "FAIL"
    )

    print("\n===== Memory Prompt Demo Summary =====")
    print("input_sequence: <random emb><prompt emb><random emb>")
    print(
        f"input_shapes: {{'random_emb': {tuple(mem_prefix.shape)}, "
        f"'prompt_emb': {tuple(prompt_embeds.shape)}, "
        f"'full_prompt_emb': {tuple(full_prompt_embeds[0].shape)}}}"
    )
    print(f"single_vs_single_summary: {single_summary}")
    print(f"batch_vs_batch_summary: {batch_summary}")
    print(f"MEMORY_DEMO_RESULT: {result}")

    if result != "PASS":
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
