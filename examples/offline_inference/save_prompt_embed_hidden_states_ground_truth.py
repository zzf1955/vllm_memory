# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Save Hugging Face ground truth prompt embeddings and hidden states.

This script runs one forward pass with ``inputs_embeds`` and stores the
per-token input embeddings plus all prompt hidden states in a ``.pth`` file.
It is intended as a reference for validating vLLM prompt-embeds +
extract-hidden-states experiments with ``max_tokens=1``.

Example:
    python examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py \
        --seq-len 16 \
        --output /tmp/qwen3_gt.pth

This writes:
    /tmp/qwen3_gt.prompt.txt
    /tmp/qwen3_gt.input.pth
    /tmp/qwen3_gt.output.pth
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TEXT = (
    "vLLM can accept prompt embeddings directly and extract hidden states "
    "from selected layers for fast inference experiments."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save prompt embeddings and hidden states ground truth."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model name or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Input text to tokenize before collecting ground truth tensors.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Maximum prompt length in tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompt_embed_hidden_states_gt.pth"),
        help=(
            "Output prefix or .pth path. The script writes .prompt.txt, "
            ".input.pth, and .output.pth files using this stem."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model and forward dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the forward pass. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading model/tokenizer.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype]


def build_metadata(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
) -> dict[str, Any]:
    config = model.config
    return {
        "dtype": str(prompt_embeds.dtype).removeprefix("torch."),
        "requested_dtype": args.dtype,
        "device": str(args.device),
        "requested_seq_len": args.seq_len,
        "hidden_size": int(prompt_embeds.shape[-1]),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
        "num_hidden_state_tensors": int(hidden_states.shape[0]),
        "vocab_size": int(getattr(config, "vocab_size")),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "input_ids_dtype": str(input_ids.dtype).removeprefix("torch."),
    }


def get_output_paths(output: Path) -> tuple[Path, Path, Path]:
    output_prefix = output.with_suffix("") if output.suffix == ".pth" else output
    return (
        output_prefix.with_suffix(".prompt.txt"),
        output_prefix.with_suffix(".input.pth"),
        output_prefix.with_suffix(".output.pth"),
    )


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0:
        raise ValueError(f"--seq-len must be positive, got {args.seq_len}")

    trust_remote_code = not args.no_trust_remote_code
    torch_dtype = get_torch_dtype(args.dtype)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, : args.seq_len].to(device)
    attention_mask = encoded.attention_mask[:, : args.seq_len].to(device)

    if input_ids.shape[1] == 0:
        raise ValueError("Tokenized input is empty.")

    with torch.inference_mode():
        prompt_embeds = model.get_input_embeddings()(input_ids)
        outputs = model(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states.")

    prompt_embeds_cpu = prompt_embeds.squeeze(0).detach().cpu()
    hidden_states_cpu = torch.stack([
        hidden_state.squeeze(0).detach().cpu()
        for hidden_state in outputs.hidden_states
    ])
    input_ids_cpu = input_ids.squeeze(0).detach().cpu()
    attention_mask_cpu = attention_mask.squeeze(0).detach().cpu()
    layer_hidden_states_cpu = hidden_states_cpu[1:]

    effective_text = tokenizer.decode(
        input_ids_cpu.tolist(), skip_special_tokens=False
    )
    metadata = build_metadata(
        args=args,
        model=model,
        input_ids=input_ids_cpu,
        prompt_embeds=prompt_embeds_cpu,
        hidden_states=hidden_states_cpu,
    )
    prompt_data = {
        "model": args.model,
        "text": args.text,
        "effective_text": effective_text,
        "seq_len": int(input_ids_cpu.shape[0]),
    }
    input_data = {
        "model": args.model,
        "seq_len": int(input_ids_cpu.shape[0]),
        "input_ids": input_ids_cpu,
        "attention_mask": attention_mask_cpu,
        "prompt_embeds": prompt_embeds_cpu,
        "metadata": metadata,
    }
    output_data = {
        "model": args.model,
        "seq_len": int(input_ids_cpu.shape[0]),
        "hidden_states": hidden_states_cpu,
        "layer_hidden_states": layer_hidden_states_cpu,
        "generated_max_new_tokens": 1,
        "metadata": metadata,
    }

    prompt_path, input_path, output_path = get_output_paths(args.output)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path.write_text(
        "\n".join([
            f"model: {prompt_data['model']}",
            f"seq_len: {prompt_data['seq_len']}",
            "text:",
            prompt_data["text"],
            "effective_text:",
            prompt_data["effective_text"],
            "",
        ]),
        encoding="utf-8",
    )
    torch.save(input_data, input_path)
    torch.save(output_data, output_path)

    print("Saved ground truth:")
    print(f"  prompt: {prompt_path}")
    print(f"  input:  {input_path}")
    print(f"  output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Prompt tokens: {input_data['seq_len']}")
    print(f"prompt_embeds shape: {tuple(prompt_embeds_cpu.shape)}")
    print(f"hidden_states shape: {tuple(hidden_states_cpu.shape)}")
    print(f"layer_hidden_states shape: {tuple(layer_hidden_states_cpu.shape)}")


if __name__ == "__main__":
    main()
