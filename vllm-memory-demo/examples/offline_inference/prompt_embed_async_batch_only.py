# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run only AsyncLLM prompt-embeds batch requests and save hidden states."""

import argparse
import asyncio
import tempfile
from pathlib import Path
from typing import Any

import torch

from vllm import SamplingParams
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
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-pth", type=Path, required=True)
    parser.add_argument("--prompt-txt", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--layer-ids", default="1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--variant-scale-step", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


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


async def collect_one(
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


async def main_async(args: argparse.Namespace) -> None:
    input_data = load_input(args.input_pth)
    input_ids = input_data["input_ids"].to(torch.long)
    prompt_embeds = input_data["prompt_embeds"]
    prompt_path = args.prompt_txt or get_sibling_prompt_path(args.input_pth)
    prompt_text = load_prompt_text(prompt_path)
    layer_ids = parse_layer_ids(args.layer_ids, input_data["metadata"])

    if args.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="vllm-prompt-embed-async-batch-"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    storage_path = work_dir / "async-prompt-embeds-batch-only"
    storage_path.mkdir(parents=True, exist_ok=True)

    variants = make_prompt_embed_variants(
        prompt_embeds, args.batch_size, args.variant_scale_step
    )

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
        print("Prompt length:", len(input_ids))
        print("Layer ids:", layer_ids)
        print("Batch size:", args.batch_size)
        print("Num rounds:", args.num_rounds)
        print("Storage path:", storage_path)
        for round_idx in range(args.num_rounds):
            tasks = []
            for batch_idx, variant in enumerate(variants):
                tasks.append(
                    asyncio.create_task(
                        collect_one(
                            engine,
                            request_id=(
                                f"async-r{round_idx:03d}-b{batch_idx:03d}"
                            ),
                            prompt_text=prompt_text,
                            prompt_embeds=variant.clone(),
                            sampling_params=sampling_params,
                        )
                    )
                )
            outputs = await asyncio.gather(*tasks)
            print(
                f"round={round_idx} generated_ids="
                f"{[output.outputs[0].token_ids for output in outputs]}"
            )
    finally:
        engine.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
