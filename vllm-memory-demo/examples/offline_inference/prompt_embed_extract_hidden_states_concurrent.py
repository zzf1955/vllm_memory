# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stress test concurrent prompt embeddings with extract_hidden_states.

This script starts one AsyncLLM engine, builds a batch of distinct prompt-embed
variants, collects a sequential reference output for each variant, then submits
the variants concurrently for several rounds. Each concurrent request is checked
against its own sequential reference. It is intended to catch request-mixing or
zero-length hidden state bugs in the prompt_embeds + ExampleHiddenStatesConnector
path.

Example:
    python examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
        --input-pth /tmp/qwen3_gt.input.pth \
        --output-pth /tmp/qwen3_gt.output.pth \
        --work-dir /tmp/qwen3_concurrent_demo \
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
from safetensors import safe_open

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

from prompt_embed_extract_hidden_states import (
    allclose_stats,
    build_kv_transfer_config,
    build_speculative_config,
    get_sibling_prompt_path,
    load_hf_selected_hidden_states,
    load_input,
    load_prompt_text,
    parse_layer_ids,
    read_extracted_hidden_states,
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress test concurrent prompt_embeds + extract_hidden_states."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-pth", type=Path, required=True)
    parser.add_argument("--prompt-txt", type=Path)
    parser.add_argument("--output-pth", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--layer-ids", default="1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("full", "gt-only", "async-only"),
        default="full",
        help=(
            "full: generate single/batch GT then run AsyncLLM; gt-only: only "
            "write GT files; async-only: load existing GT files and run AsyncLLM."
        ),
    )
    parser.add_argument(
        "--variant-scale-step",
        type=float,
        default=0.01,
        help=(
            "Scale delta used to create distinct prompt-embed variants within "
            "one concurrent batch. Variant i uses scale 1 + i * step."
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


async def collect_one(
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


def validate_output(
    output: RequestOutput,
    *,
    expected_hidden_states: torch.Tensor | None,
    expected_shape: tuple[int, int, int],
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
    result: dict[str, Any] = {
        "request_id": output.request_id,
        "generated_ids": generated_ids,
        "hidden_shape": tuple(hidden_states.shape),
        "token_ids": token_ids.tolist(),
        "hidden_states_path": output.kv_transfer_params["hidden_states_path"],
    }

    if expected_hidden_states is not None:
        result["compare"] = allclose_stats(hidden_states, expected_hidden_states)

    return result


def make_prompt_embed_variants(
    prompt_embeds: torch.Tensor,
    batch_size: int,
    scale_step: float,
) -> list[torch.Tensor]:
    variants = []
    for idx in range(batch_size):
        scale = 1.0 + idx * scale_step
        variants.append((prompt_embeds * scale).clone())
    return variants


def collect_llm_ground_truth(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    prompt_embeds: list[torch.Tensor],
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
            for embeds in prompt_embeds
        ]
        if batched:
            outputs = llm.generate(prompts, sampling_params)
        else:
            outputs = []
            for prompt in prompts:
                one_output = llm.generate(prompt, sampling_params)
                if len(one_output) != 1:
                    raise RuntimeError(
                        f"Expected one single GT output, got {len(one_output)}."
                    )
                outputs.append(one_output[0])

        if len(outputs) != len(prompt_embeds):
            raise RuntimeError(
                f"Expected {len(prompt_embeds)} GT outputs, got {len(outputs)}."
            )

        ground_truth = []
        for output in outputs:
            tensors = read_extracted_hidden_states(output)
            ground_truth.append(
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
        return ground_truth
    finally:
        if hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_compares(compares: list[dict[str, Any]]) -> dict[str, Any]:
    failed = sum(0 if compare["allclose"] else 1 for compare in compares)
    return {
        "failed_compares": failed,
        "max_abs": max((compare["max_abs"] for compare in compares), default=0.0),
        "mean_abs_avg": (
            sum(compare["mean_abs"] for compare in compares) / len(compares)
            if compares
            else 0.0
        ),
    }


def load_ground_truth_from_storage(storage_path: Path) -> list[dict[str, Any]]:
    paths = sorted(
        storage_path.glob("*.safetensors"),
        key=lambda path: int(path.name.split("-", 1)[0]),
    )
    if not paths:
        raise FileNotFoundError(f"No safetensors files found in {storage_path}.")

    ground_truth = []
    for path in paths:
        with safe_open(path, framework="pt") as f:
            ground_truth.append(
                {
                    "request_id": path.stem,
                    "generated_ids": None,
                    "hidden_states": f.get_tensor("hidden_states"),
                    "token_ids": f.get_tensor("token_ids"),
                    "hidden_states_path": str(path),
                }
            )
    return ground_truth


async def run_stress(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_rounds <= 0:
        raise ValueError("--num-rounds must be positive.")

    input_data = load_input(args.input_pth)
    input_ids = input_data["input_ids"].to(torch.long)
    prompt_embeds = input_data["prompt_embeds"]
    prompt_path = args.prompt_txt or get_sibling_prompt_path(args.input_pth)
    prompt_text = load_prompt_text(prompt_path)
    metadata = input_data["metadata"]
    layer_ids = parse_layer_ids(args.layer_ids, metadata)

    if args.max_model_len < len(input_ids) + args.max_tokens:
        raise ValueError(
            "--max-model-len must be at least prompt length + max tokens: "
            f"{len(input_ids) + args.max_tokens}"
        )

    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vllm-prompt-embed-concurrent-"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    single_gt_storage_path = work_dir / "llm-single-ground-truth"
    batch_gt_storage_path = work_dir / "llm-batch-ground-truth"
    async_storage_path = work_dir / "async-prompt-embeds-concurrent"
    single_gt_storage_path.mkdir(parents=True, exist_ok=True)
    batch_gt_storage_path.mkdir(parents=True, exist_ok=True)
    async_storage_path.mkdir(parents=True, exist_ok=True)

    expected_hidden_states = None
    if args.output_pth is not None:
        expected_hidden_states = load_hf_selected_hidden_states(
            args.output_pth, layer_ids
        )

    hidden_size = int(metadata["hidden_size"])
    expected_shape = (len(input_ids), len(layer_ids), hidden_size)
    prompt_embed_variants = make_prompt_embed_variants(
        prompt_embeds, args.batch_size, args.variant_scale_step
    )

    total_requests = 0
    batch_gt_failed_compares = 0
    single_gt_failed_compares = 0
    batch_gt_max_abs = 0.0
    single_gt_max_abs = 0.0
    batch_gt_mean_abs_total = 0.0
    single_gt_mean_abs_total = 0.0
    engine: AsyncLLM | None = None

    try:
        print("Prompt text:", prompt_text)
        print("Prompt length:", len(input_ids))
        print("Layer ids:", layer_ids)
        print("Batch size:", args.batch_size)
        print("Num rounds:", args.num_rounds)
        print("Variant scale step:", args.variant_scale_step)
        print("Single GT storage path:", single_gt_storage_path)
        print("Batch GT storage path:", batch_gt_storage_path)
        print("Async storage path:", async_storage_path)

        if args.mode == "async-only":
            single_ground_truth = load_ground_truth_from_storage(
                single_gt_storage_path
            )
            batch_ground_truth = load_ground_truth_from_storage(batch_gt_storage_path)
        else:
            single_ground_truth = collect_llm_ground_truth(
                args=args,
                prompt_text=prompt_text,
                prompt_embeds=prompt_embed_variants,
                storage_path=single_gt_storage_path,
                layer_ids=layer_ids,
                batched=False,
            )
            batch_ground_truth = collect_llm_ground_truth(
                args=args,
                prompt_text=prompt_text,
                prompt_embeds=prompt_embed_variants,
                storage_path=batch_gt_storage_path,
                layer_ids=layer_ids,
                batched=True,
            )
        reference_generated_ids = [
            baseline["generated_ids"] for baseline in batch_ground_truth
        ]

        single_vs_batch_compares = []
        for batch_idx, batch_baseline in enumerate(batch_ground_truth):
            if tuple(batch_baseline["hidden_states"].shape) != expected_shape:
                raise AssertionError(
                    f"Unexpected batch GT hidden_states shape for batch "
                    f"{batch_idx}: "
                    f"{tuple(batch_baseline['hidden_states'].shape)} "
                    f"!= {expected_shape}"
                )
            single_baseline = single_ground_truth[batch_idx]
            single_vs_batch_compares.append(
                allclose_stats(
                    batch_baseline["hidden_states"],
                    single_baseline["hidden_states"],
                )
            )
            if (
                batch_baseline["generated_ids"] is not None
                and single_baseline["generated_ids"] is not None
                and batch_baseline["generated_ids"]
                != single_baseline["generated_ids"]
            ):
                print(
                    "Single GT generated ids differ from batch GT for "
                    f"batch {batch_idx}: {single_baseline['generated_ids']} "
                    f"!= {batch_baseline['generated_ids']}"
                )

        print("Batch GT vs single GT:", summarize_compares(single_vs_batch_compares))
        if expected_hidden_states is not None:
            print(
                "Batch GT variant 0 vs HF:",
                allclose_stats(
                    batch_ground_truth[0]["hidden_states"], expected_hidden_states
                ),
            )

        if args.mode == "gt-only":
            return

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

        for round_idx in range(args.num_rounds):
            tasks = []
            for batch_idx in range(args.batch_size):
                request_id = f"concurrent-r{round_idx:03d}-b{batch_idx:03d}"
                tasks.append(
                    asyncio.create_task(
                        collect_one(
                            engine,
                            request_id=request_id,
                            prompt_text=prompt_text,
                            prompt_embeds=prompt_embed_variants[batch_idx].clone(),
                            sampling_params=sampling_params,
                        )
                    )
                )

            outputs = await asyncio.gather(*tasks)
            round_generated_ids = []
            round_batch_gt_max_abs = 0.0
            round_single_gt_max_abs = 0.0
            round_batch_gt_failed_compares = 0
            round_single_gt_failed_compares = 0

            for batch_idx, output in enumerate(outputs):
                result = validate_output(
                    output,
                    expected_hidden_states=None,
                    expected_shape=expected_shape,
                )
                generated_ids = result["generated_ids"]
                round_generated_ids.append(generated_ids)
                if (
                    reference_generated_ids[batch_idx] is not None
                    and generated_ids != reference_generated_ids[batch_idx]
                ):
                    raise AssertionError(
                        f"Generated ids changed for {result['request_id']}: "
                        f"{generated_ids} != {reference_generated_ids[batch_idx]}"
                    )

                tensors = read_extracted_hidden_states(output)
                batch_gt_compare = allclose_stats(
                    tensors["hidden_states"],
                    batch_ground_truth[batch_idx]["hidden_states"],
                )
                single_gt_compare = allclose_stats(
                    tensors["hidden_states"],
                    single_ground_truth[batch_idx]["hidden_states"],
                )
                batch_gt_max_abs = max(batch_gt_max_abs,
                                       batch_gt_compare["max_abs"])
                single_gt_max_abs = max(single_gt_max_abs,
                                        single_gt_compare["max_abs"])
                round_batch_gt_max_abs = max(
                    round_batch_gt_max_abs, batch_gt_compare["max_abs"]
                )
                round_single_gt_max_abs = max(
                    round_single_gt_max_abs, single_gt_compare["max_abs"]
                )
                batch_gt_mean_abs_total += batch_gt_compare["mean_abs"]
                single_gt_mean_abs_total += single_gt_compare["mean_abs"]
                if not batch_gt_compare["allclose"]:
                    batch_gt_failed_compares += 1
                    round_batch_gt_failed_compares += 1
                if not single_gt_compare["allclose"]:
                    single_gt_failed_compares += 1
                    round_single_gt_failed_compares += 1

                total_requests += 1

            print(
                f"round={round_idx} ok requests={len(outputs)} "
                f"generated_ids={round_generated_ids} "
                f"batch_gt_max_abs={round_batch_gt_max_abs} "
                f"batch_gt_failed_compares={round_batch_gt_failed_compares} "
                f"single_gt_max_abs={round_single_gt_max_abs} "
                f"single_gt_failed_compares={round_single_gt_failed_compares}"
            )

    finally:
        if engine is not None:
            engine.shutdown()

    batch_gt_mean_abs_avg = (
        batch_gt_mean_abs_total / total_requests if total_requests else 0.0
    )
    single_gt_mean_abs_avg = (
        single_gt_mean_abs_total / total_requests if total_requests else 0.0
    )
    print("Stress test summary:")
    print("  total_requests:", total_requests)
    print("  batch_gt_failed_compares:", batch_gt_failed_compares)
    print("  batch_gt_max_abs:", batch_gt_max_abs)
    print("  batch_gt_mean_abs_avg:", batch_gt_mean_abs_avg)
    print("  single_gt_failed_compares:", single_gt_failed_compares)
    print("  single_gt_max_abs:", single_gt_max_abs)
    print("  single_gt_mean_abs_avg:", single_gt_mean_abs_avg)
    print("  expected_shape:", expected_shape)
    print("  generated_ids:", reference_generated_ids)

    if batch_gt_failed_compares:
        raise AssertionError(
            f"{batch_gt_failed_compares} batch-GT hidden-state comparisons "
            "failed."
        )


def main() -> None:
    args = parse_args()
    asyncio.run(run_stress(args))


if __name__ == "__main__":
    main()
