# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stress test [MEM, prompt_embeds, MEM] inputs with extract_hidden_states.

The input prompt embeddings are constructed as:

    [random MEM prefix, prompt_embeds, random MEM suffix]

Both MEM tensors are fixed random vectors for the whole run. The script first
uses vLLM LLM.generate to collect baseline hidden states, then uses one AsyncLLM
engine to submit concurrent prompt-embeds requests. It compares two output
regions independently:

    prefix region: [MEM prefix, prompt hidden states]
    memory region: [MEM suffix]

Example:
    python examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
        --input-pth /tmp/qwen3_gt.input.pth \
        --work-dir /tmp/qwen3_memory_concurrent_demo \
        --batch-size 4 \
        --num-rounds 10
"""

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
    allclose_stats,
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
    parser = argparse.ArgumentParser(
        description=(
            "Stress test concurrent [MEM, prompt_embeds, MEM] inputs with "
            "extract_hidden_states."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-pth", type=Path, required=True)
    parser.add_argument("--prompt-txt", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--layer-ids", default="1")
    parser.add_argument("--mem-len", type=int, default=8)
    parser.add_argument("--mem-seed", type=int, default=1234)
    parser.add_argument("--mem-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--variant-scale-step",
        type=float,
        default=0.01,
        help=(
            "Scale delta used to create distinct prompt-embed variants within "
            "one concurrent batch. MEM vectors stay fixed."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=True,
        help="Run vLLM in eager mode. Enabled by default for this stress demo.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow vLLM compilation/cudagraph capture.",
    )
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
    mem_prefix = torch.randn(
        (mem_len, hidden_size), generator=generator, dtype=torch.float32
    )
    mem_suffix = torch.randn(
        (mem_len, hidden_size), generator=generator, dtype=torch.float32
    )
    return (mem_prefix * scale).to(dtype), (mem_suffix * scale).to(dtype)


def make_full_prompt_embeds(
    *,
    mem_prefix: torch.Tensor,
    prompt_embeds: torch.Tensor,
    mem_suffix: torch.Tensor,
    batch_size: int,
    variant_scale_step: float,
) -> list[torch.Tensor]:
    variants = []
    for batch_idx in range(batch_size):
        scale = 1.0 + batch_idx * variant_scale_step
        prompt_variant = prompt_embeds * scale
        variants.append(
            torch.cat([mem_prefix, prompt_variant, mem_suffix], dim=0).contiguous()
        )
    return variants


def split_compare_stats(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    prefix_len: int,
) -> dict[str, dict[str, Any]]:
    return {
        "prefix": allclose_stats(actual[:prefix_len], expected[:prefix_len]),
        "memory": allclose_stats(actual[prefix_len:], expected[prefix_len:]),
    }


def stats_allclose(stats: dict[str, dict[str, Any]]) -> bool:
    return stats["prefix"]["allclose"] and stats["memory"]["allclose"]


async def collect_one_async(
    engine: AsyncLLM,
    *,
    request_id: str,
    prompt_text: str,
    prompt_embeds: torch.Tensor,
    sampling_params: SamplingParams,
) -> RequestOutput:
    final_output = None
    prompt = {
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


def collect_llm_baselines(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    full_prompt_embeds: list[torch.Tensor],
    storage_path: Path,
    layer_ids: list[int],
) -> list[dict[str, Any]]:
    llm = LLM(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
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
        outputs = llm.generate(prompts, sampling_params)
        if len(outputs) != len(full_prompt_embeds):
            raise RuntimeError(
                f"Expected {len(full_prompt_embeds)} outputs, got {len(outputs)}."
            )

        baselines = []
        for output in outputs:
            tensors = read_extracted_hidden_states(output)
            baselines.append(
                {
                    "request_id": output.request_id,
                    "generated_ids": output.outputs[0].token_ids,
                    "hidden_states": tensors["hidden_states"],
                    "token_ids": tensors["token_ids"],
                    "hidden_states_path": output.kv_transfer_params[
                        "hidden_states_path"
                    ],
                }
            )
        return baselines
    finally:
        if hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_async_output(
    output: RequestOutput,
    *,
    baseline: dict[str, Any],
    expected_shape: tuple[int, int, int],
    prefix_len: int,
) -> dict[str, Any]:
    tensors = read_extracted_hidden_states(output)
    hidden_states = tensors["hidden_states"]
    token_ids = tensors["token_ids"]

    if tuple(hidden_states.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected hidden_states shape for {output.request_id}: "
            f"{tuple(hidden_states.shape)} != {expected_shape}"
        )
    if token_ids.shape[0] != expected_shape[0]:
        raise AssertionError(
            f"Unexpected token_ids length for {output.request_id}: "
            f"{token_ids.shape[0]} != {expected_shape[0]}"
        )

    generated_ids = output.outputs[0].token_ids
    if generated_ids != baseline["generated_ids"]:
        raise AssertionError(
            f"Generated ids changed for {output.request_id}: "
            f"{generated_ids} != {baseline['generated_ids']}"
        )

    compare = split_compare_stats(
        actual=hidden_states,
        expected=baseline["hidden_states"],
        prefix_len=prefix_len,
    )
    return {
        "request_id": output.request_id,
        "generated_ids": generated_ids,
        "compare": compare,
        "hidden_states_path": output.kv_transfer_params["hidden_states_path"],
    }


async def run_stress(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_rounds <= 0:
        raise ValueError("--num-rounds must be positive.")

    input_data = load_input(args.input_pth)
    prompt_embeds = input_data["prompt_embeds"]
    metadata = input_data["metadata"]
    layer_ids = parse_layer_ids(args.layer_ids, metadata)
    hidden_size = int(metadata["hidden_size"])
    prompt_len = len(prompt_embeds)
    full_prompt_len = args.mem_len + prompt_len + args.mem_len
    prefix_len = args.mem_len + prompt_len
    memory_len = args.mem_len

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

    baseline_storage_path = work_dir / "llm-generate-baseline"
    async_storage_path = work_dir / "async-prompt-embeds-memory"
    baseline_storage_path.mkdir(parents=True, exist_ok=True)
    async_storage_path.mkdir(parents=True, exist_ok=True)

    expected_shape = (full_prompt_len, len(layer_ids), hidden_size)

    print("Prompt text:", prompt_text)
    print("Prompt len:", prompt_len)
    print("MEM len:", args.mem_len)
    print("Full prompt len:", full_prompt_len)
    print("Prefix compare len:", prefix_len)
    print("Memory compare len:", memory_len)
    print("Layer ids:", layer_ids)
    print("Batch size:", args.batch_size)
    print("Num rounds:", args.num_rounds)
    print("Variant scale step:", args.variant_scale_step)
    print("MEM seed:", args.mem_seed)
    print("MEM scale:", args.mem_scale)
    print("Baseline storage path:", baseline_storage_path)
    print("Async storage path:", async_storage_path)

    baselines = collect_llm_baselines(
        args=args,
        prompt_text=prompt_text,
        full_prompt_embeds=full_prompt_embeds,
        storage_path=baseline_storage_path,
        layer_ids=layer_ids,
    )
    for batch_idx, baseline in enumerate(baselines):
        if tuple(baseline["hidden_states"].shape) != expected_shape:
            raise AssertionError(
                f"Unexpected baseline hidden_states shape for batch {batch_idx}: "
                f"{tuple(baseline['hidden_states'].shape)} != {expected_shape}"
            )
        if baseline["token_ids"].shape[0] != full_prompt_len:
            raise AssertionError(
                f"Unexpected baseline token_ids length for batch {batch_idx}: "
                f"{baseline['token_ids'].shape[0]} != {full_prompt_len}"
            )

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        enable_prompt_embeds=True,
        speculative_config=build_speculative_config(layer_ids),
        kv_transfer_config=build_kv_transfer_config(async_storage_path),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=0,
    )

    total_requests = 0
    failed_compares = 0
    prefix_max_abs = 0.0
    memory_max_abs = 0.0
    prefix_mean_abs_total = 0.0
    memory_mean_abs_total = 0.0

    try:
        for round_idx in range(args.num_rounds):
            tasks = []
            for batch_idx, embeds in enumerate(full_prompt_embeds):
                request_id = f"memory-r{round_idx:03d}-b{batch_idx:03d}"
                tasks.append(
                    asyncio.create_task(
                        collect_one_async(
                            engine,
                            request_id=request_id,
                            prompt_text=prompt_text,
                            prompt_embeds=embeds.clone(),
                            sampling_params=sampling_params,
                        )
                    )
                )

            outputs = await asyncio.gather(*tasks)
            round_prefix_max_abs = 0.0
            round_memory_max_abs = 0.0
            round_failed_compares = 0
            round_generated_ids = []

            for batch_idx, output in enumerate(outputs):
                result = validate_async_output(
                    output,
                    baseline=baselines[batch_idx],
                    expected_shape=expected_shape,
                    prefix_len=prefix_len,
                )
                compare = result["compare"]
                prefix_stats = compare["prefix"]
                memory_stats = compare["memory"]

                prefix_max_abs = max(prefix_max_abs, prefix_stats["max_abs"])
                memory_max_abs = max(memory_max_abs, memory_stats["max_abs"])
                round_prefix_max_abs = max(
                    round_prefix_max_abs, prefix_stats["max_abs"]
                )
                round_memory_max_abs = max(
                    round_memory_max_abs, memory_stats["max_abs"]
                )
                prefix_mean_abs_total += prefix_stats["mean_abs"]
                memory_mean_abs_total += memory_stats["mean_abs"]
                if not stats_allclose(compare):
                    failed_compares += 1
                    round_failed_compares += 1

                round_generated_ids.append(result["generated_ids"])
                total_requests += 1

            print(
                f"round={round_idx} ok requests={len(outputs)} "
                f"generated_ids={round_generated_ids} "
                f"prefix_max_abs={round_prefix_max_abs} "
                f"memory_max_abs={round_memory_max_abs} "
                f"round_failed_compares={round_failed_compares}"
            )
    finally:
        engine.shutdown()

    prefix_mean_abs_avg = (
        prefix_mean_abs_total / total_requests if total_requests else 0.0
    )
    memory_mean_abs_avg = (
        memory_mean_abs_total / total_requests if total_requests else 0.0
    )
    print("Memory stress test summary:")
    print("  total_requests:", total_requests)
    print("  failed_compares:", failed_compares)
    print("  prefix_max_abs:", prefix_max_abs)
    print("  prefix_mean_abs_avg:", prefix_mean_abs_avg)
    print("  memory_max_abs:", memory_max_abs)
    print("  memory_mean_abs_avg:", memory_mean_abs_avg)
    print("  expected_shape:", expected_shape)
    print("  baseline_generated_ids:", [b["generated_ids"] for b in baselines])

    if failed_compares:
        raise AssertionError(f"{failed_compares} hidden-state comparisons failed.")


def main() -> None:
    args = parse_args()
    asyncio.run(run_stress(args))


if __name__ == "__main__":
    main()
