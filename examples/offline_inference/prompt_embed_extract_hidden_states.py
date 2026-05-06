# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run prompt embeddings with extract_hidden_states and compare against LLM.generate.

This example uses the same prompt embeddings as input to AsyncLLM and compares
the extracted hidden states against a normal token prompt run with LLM.generate.
Optionally, it can also compare against Hugging Face tensors saved by
save_prompt_embed_hidden_states_ground_truth.py.

Example:
    python examples/offline_inference/prompt_embed_extract_hidden_states.py \
        --input-pth /tmp/qwen3_gt.input.pth \
        --output-pth /tmp/qwen3_gt.output.pth \
        --work-dir /tmp/qwen3_extract_demo
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


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vLLM prompt_embeds + extract_hidden_states against "
            "normal token-prompt LLM.generate."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model name or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--input-pth",
        type=Path,
        required=True,
        help="Input .pth file saved by save_prompt_embed_hidden_states_ground_truth.py.",
    )
    parser.add_argument(
        "--prompt-txt",
        type=Path,
        help=(
            "Optional prompt .txt file saved by "
            "save_prompt_embed_hidden_states_ground_truth.py. If omitted, the "
            "script tries the sibling .prompt.txt path."
        ),
    )
    parser.add_argument(
        "--output-pth",
        type=Path,
        help="Optional HF output .pth file for hidden-state comparison.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for connector safetensors outputs.",
    )
    parser.add_argument(
        "--layer-ids",
        default="1",
        help=(
            "Comma-separated target layer ids for extraction, or 'all'. "
            "'all' uses 1..num_hidden_layers from the saved input metadata."
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
        help="Run vLLM in eager mode. Enabled by default for this small demo.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow vLLM compilation/cudagraph capture.",
    )
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def load_input(input_path: Path) -> dict[str, Any]:
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    required = {"input_ids", "prompt_embeds", "metadata"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{input_path} is missing keys: {sorted(missing)}")
    return data


def get_sibling_prompt_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".input.pth"):
        return input_path.with_name(f"{name.removesuffix('.input.pth')}.prompt.txt")
    return input_path.with_suffix(".prompt.txt")


def load_prompt_text(prompt_path: Path | None) -> str:
    if prompt_path is None or not prompt_path.exists():
        return ""

    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    for marker in ("effective_text:", "text:"):
        if marker not in lines:
            continue
        start = lines.index(marker) + 1
        collected = []
        for line in lines[start:]:
            if line == "":
                break
            collected.append(line)
        if collected:
            return "\n".join(collected)
    return ""


def parse_layer_ids(value: str, metadata: dict[str, Any]) -> list[int]:
    if value == "all":
        num_hidden_layers = metadata.get("num_hidden_layers")
        if num_hidden_layers is None:
            raise ValueError("Cannot use --layer-ids all without num_hidden_layers.")
        return list(range(1, int(num_hidden_layers) + 1))

    layer_ids = [int(item) for item in value.split(",") if item.strip()]
    if not layer_ids:
        raise ValueError("--layer-ids must not be empty.")
    return layer_ids


def build_speculative_config(layer_ids: list[int]) -> dict[str, Any]:
    return {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": layer_ids,
            }
        },
    }


def build_kv_transfer_config(storage_path: Path) -> dict[str, Any]:
    return {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": str(storage_path),
        },
    }


def read_extracted_hidden_states(output: RequestOutput) -> dict[str, torch.Tensor]:
    if output.kv_transfer_params is None:
        raise RuntimeError("Request output did not include kv_transfer_params.")
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    if hidden_states_path is None:
        raise RuntimeError("kv_transfer_params did not include hidden_states_path.")

    with safe_open(hidden_states_path, framework="pt") as f:
        return {
            "hidden_states": f.get_tensor("hidden_states"),
            "token_ids": f.get_tensor("token_ids"),
        }


def allclose_stats(
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
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "shape": tuple(actual.shape),
    }


async def run_async_prompt_embeds(
    *,
    args: argparse.Namespace,
    prompt_embeds: torch.Tensor,
    prompt_text: str,
    storage_path: Path,
    layer_ids: list[int],
) -> RequestOutput:
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prompt_embeds=True,
        speculative_config=build_speculative_config(layer_ids),
        kv_transfer_config=build_kv_transfer_config(storage_path),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            seed=0,
        )
        final_output = None
        prompt = {
            "prompt": prompt_text,
            "prompt_embeds": prompt_embeds,
        }
        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id="prompt-embeds-extract-hidden-states",
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError("AsyncLLM did not produce an output.")
        return final_output
    finally:
        engine.shutdown()


def run_llm_token_prompt(
    *,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    prompt_text: str,
    storage_path: Path,
    layer_ids: list[int],
) -> RequestOutput:
    llm = LLM(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        speculative_config=build_speculative_config(layer_ids),
        kv_transfer_config=build_kv_transfer_config(storage_path),
    )
    try:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            seed=0,
        )
        outputs = llm.generate(
            {
                "prompt": prompt_text,
                "prompt_token_ids": input_ids.tolist(),
            },
            sampling_params,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one output, got {len(outputs)}.")
        return outputs[0]
    finally:
        if hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_hf_selected_hidden_states(
    output_path: Path,
    layer_ids: list[int],
) -> torch.Tensor:
    data = torch.load(output_path, map_location="cpu", weights_only=False)
    if "hidden_states" not in data:
        raise ValueError(f"{output_path} is missing key: hidden_states")

    hidden_states = data["hidden_states"]
    selected = []
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= hidden_states.shape[0]:
            raise ValueError(
                f"Layer id {layer_id} is outside HF hidden_states range "
                f"0..{hidden_states.shape[0] - 1}."
            )
        selected.append(hidden_states[layer_id])
    return torch.stack(selected, dim=1)


def main() -> None:
    args = parse_args()
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
        work_dir = Path(tempfile.mkdtemp(prefix="vllm-prompt-embed-hidden-states-"))
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    token_storage_path = work_dir / "llm-token-prompt"
    embed_storage_path = work_dir / "async-prompt-embeds"
    token_storage_path.mkdir(parents=True, exist_ok=True)
    embed_storage_path.mkdir(parents=True, exist_ok=True)

    token_output = run_llm_token_prompt(
        args=args,
        input_ids=input_ids,
        prompt_text=prompt_text,
        storage_path=token_storage_path,
        layer_ids=layer_ids,
    )
    token_tensors = read_extracted_hidden_states(token_output)

    embed_output = asyncio.run(
        run_async_prompt_embeds(
            args=args,
            prompt_embeds=prompt_embeds,
            prompt_text=prompt_text,
            storage_path=embed_storage_path,
            layer_ids=layer_ids,
        )
    )
    embed_tensors = read_extracted_hidden_states(embed_output)

    token_generated = token_output.outputs[0].token_ids
    embed_generated = embed_output.outputs[0].token_ids
    hidden_compare = allclose_stats(
        embed_tensors["hidden_states"],
        token_tensors["hidden_states"],
    )

    print("Prompt text:", prompt_text)
    print("Prompt length:", len(input_ids))
    print("Layer ids:", layer_ids)
    print("LLM token generated ids:", token_generated)
    print("Async prompt-embeds generated ids:", embed_generated)
    print("Generated ids match:", token_generated == embed_generated)
    print("LLM token hidden shape:", tuple(token_tensors["hidden_states"].shape))
    print("Prompt-embeds hidden shape:", tuple(embed_tensors["hidden_states"].shape))
    print("Prompt-embeds token_ids:", embed_tensors["token_ids"].tolist())
    print("Prompt-embeds vs token hidden states:", hidden_compare)
    print(
        "LLM token safetensors:",
        token_output.kv_transfer_params["hidden_states_path"],
    )
    print(
        "Async prompt-embeds safetensors:",
        embed_output.kv_transfer_params["hidden_states_path"],
    )

    if args.output_pth is not None:
        hf_hidden_states = load_hf_selected_hidden_states(args.output_pth, layer_ids)
        hf_compare = allclose_stats(embed_tensors["hidden_states"], hf_hidden_states)
        print("Prompt-embeds vs HF hidden states:", hf_compare)


if __name__ == "__main__":
    main()
