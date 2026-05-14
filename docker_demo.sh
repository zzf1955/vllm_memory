#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SOURCE_ROOT:=/home/dpsk_a2a/vllm-memory}"

if [[ "${DEMO_ROOT}" == "${SOURCE_ROOT}" ]]; then
  echo "ERROR: do not run this script from the vLLM source checkout." >&2
  echo "Run: bash /home/dpsk_a2a/vllm-memory-demo/docker_demo.sh" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_ROOT}/examples/offline_inference" ]]; then
  echo "ERROR: missing source examples: ${SOURCE_ROOT}/examples/offline_inference" >&2
  exit 1
fi

mkdir -p "${DEMO_ROOT}/examples"
rm -rf "${DEMO_ROOT}/examples/offline_inference"
cp -a "${SOURCE_ROOT}/examples/offline_inference" \
  "${DEMO_ROOT}/examples/offline_inference"

# This docker demo assumes the patched Python files have already been copied
# into the image's installed vLLM package:
#   /usr/local/lib/python3.12/dist-packages/vllm
#
# Do not set PYTHONPATH here. For this reproduction, vLLM should import from
# dist-packages, not from the checked-out source tree.
unset PYTHONPATH

: "${PYTHON:=python}"
: "${GPU:=0}"
: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${SEQ_LEN:=16}"
: "${BATCH_SIZE:=4}"
: "${NUM_ROUNDS:=10}"
: "${GPU_MEMORY_UTILIZATION:=0.1}"
: "${MAX_MODEL_LEN:=512}"
: "${LAYER_IDS:=all}"
: "${DTYPE:=bfloat16}"

: "${RUN_ROOT:=/home/dpsk_a2a/tmp/vllm_prompt_embed_hidden_states_demo}"
: "${HF_HOME:=/home/dpsk_a2a/.cache/huggingface}"
: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
: "${UV_CACHE_DIR:=/home/dpsk_a2a/.cache/uv}"
: "${PIP_CACHE_DIR:=/home/dpsk_a2a/.cache/pip}"
: "${TMPDIR:=/home/dpsk_a2a/tmp}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_HUB_DISABLE_XET:=1}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=600}"
: "${HF_HUB_ETAG_TIMEOUT:=60}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE
export UV_CACHE_DIR
export PIP_CACHE_DIR
export TMPDIR
export HF_ENDPOINT
export HF_HUB_DISABLE_XET
export HF_HUB_DOWNLOAD_TIMEOUT
export HF_HUB_ETAG_TIMEOUT

# The original local script used a host-local proxy. In the container this
# usually points to nothing and causes "Connection refused".
if [[ "${CLEAR_PROXY:-1}" == "1" ]]; then
  unset http_proxy
  unset https_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
fi
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

SINGLE_ROOT="${RUN_ROOT}/single"
BATCH_ROOT="${RUN_ROOT}/batch_b${BATCH_SIZE}_r${NUM_ROUNDS}"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/docker_demo_$(date +%Y%m%d_%H%M%S).log"

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${SINGLE_ROOT}" \
  "${BATCH_ROOT}" \
  "${LOG_DIR}"

run_step() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "+ $*"
  "$@"
}

print_final_summary() {
  echo
  echo "===== FINAL SUMMARY ====="
  "${PYTHON}" - \
    "${SINGLE_ROOT}/single_gt.prompt.txt" \
    "${SINGLE_ROOT}/single_gt.input.pth" \
    "${LOG_FILE}" \
    "${BATCH_SIZE}" \
    "${NUM_ROUNDS}" <<'PY'
import ast
import sys
from pathlib import Path

import torch

prompt_path = Path(sys.argv[1])
input_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
batch_size = int(sys.argv[4])
num_rounds = int(sys.argv[5])


def read_prompt_sections(path: Path) -> dict[str, str]:
    sections: dict[str, str] = {}
    if not path.exists():
        return sections
    current: str | None = None
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line in {"text:", "effective_text:"}:
            if current is not None:
                sections[current] = "\n".join(values)
            current = line[:-1]
            values = []
        elif current is not None:
            values.append(line)
    if current is not None:
        sections[current] = "\n".join(values).rstrip()
    return sections


def last_value(lines: list[str], prefix: str) -> str | None:
    for line in reversed(lines):
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def parse_dict(lines: list[str], prefix: str) -> dict | None:
    value = last_value(lines, prefix)
    if not value:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def pass_fail(ok: bool | None) -> str:
    if ok is None:
        return "MISSING"
    return "PASS" if ok else "FAIL"


def print_compare(name: str, data: dict | None, *, failed_key: bool = False) -> None:
    if data is None:
        print(f"{name}: MISSING")
        return
    if failed_key:
        ok = data.get("failed") == 0
        print(
            f"{name}: {pass_fail(ok)} "
            f"count={data.get('count')} failed={data.get('failed')} "
            f"max_abs={data.get('max_abs')} mean_abs_avg={data.get('mean_abs_avg')}"
        )
        return
    ok = bool(data.get("allclose"))
    print(
        f"{name}: {pass_fail(ok)} "
        f"max_abs={data.get('max_abs')} mean_abs={data.get('mean_abs')} "
        f"shape={data.get('shape')}"
    )


prompt_sections = read_prompt_sections(prompt_path)
input_data = (
    torch.load(input_path, map_location="cpu", weights_only=False)
    if input_path.exists()
    else {}
)
input_ids = input_data.get("input_ids")
prompt_embeds = input_data.get("prompt_embeds")
attention_mask = input_data.get("attention_mask")

print("[Input prompt format]")
print("token baseline input:")
print("  {")
print("    'prompt': <effective_text>,")
print("    'prompt_token_ids': input_ids,")
print("  }")
print("prompt-embeds input:")
print("  {")
print("    'prompt': <effective_text>,")
print("    'prompt_embeds': prompt_embeds,")
print("  }")
print("batch prompt-embeds input:")
print(
    "  same prompt text; prompt_embeds variants are "
    f"base_prompt_embeds * (1 + batch_idx * variant_scale_step), "
    f"batch_size={batch_size}, num_rounds={num_rounds}"
)
print(f"prompt file: {prompt_path}")
print(f"original text: {prompt_sections.get('text', '<missing>')!r}")
print(f"effective prompt: {prompt_sections.get('effective_text', '<missing>')!r}")
if input_ids is not None:
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"input_ids: {input_ids.tolist()}")
if attention_mask is not None:
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
if prompt_embeds is not None:
    print(f"prompt_embeds shape: {tuple(prompt_embeds.shape)}")

lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

print()
print("[Single test]")
generated = last_value(lines, "Generated ids match:")
generated_ok = None if generated is None else generated == "True"
print(f"generated_ids_match: {pass_fail(generated_ok)} value={generated}")
print_compare(
    "prompt_embeds_vs_token_hidden_states",
    parse_dict(lines, "Prompt-embeds vs token hidden states:"),
)
print_compare(
    "prompt_embeds_vs_hf_hidden_states",
    parse_dict(lines, "Prompt-embeds vs HF hidden states:"),
)
single_result = last_value(lines, "SINGLE_DEMO_RESULT:")
print(f"single_demo_result: {single_result or 'MISSING'}")

print()
print("[Batch test]")
print_compare(
    "batch_gt_vs_single_gt",
    parse_dict(lines, "single_vs_batch_summary:"),
    failed_key=True,
)
print_compare(
    "async_batch_vs_batch_gt",
    parse_dict(lines, "async_vs_batch_summary:"),
    failed_key=True,
)
print_compare(
    "async_batch_vs_single_gt",
    parse_dict(lines, "async_vs_single_summary:"),
    failed_key=True,
)
batch_result = last_value(lines, "BATCH_DEMO_RESULT:")
print(f"batch_demo_result: {batch_result or 'MISSING'}")

overall_ok = single_result == "PASS" and batch_result == "PASS"
print()
print(f"OVERALL_RESULT: {pass_fail(overall_ok)}")
PY
}

main() {
  echo "Demo root: ${RUN_ROOT}"
  echo "Log file: ${LOG_FILE}"
  echo "Demo code root: ${DEMO_ROOT}"
  echo "PWD: $(pwd)"
  echo "Python: ${PYTHON}"
  echo "GPU: ${GPU}"
  echo "Model: ${MODEL}"
  echo "SEQ_LEN: ${SEQ_LEN}"
  echo "LAYER_IDS: ${LAYER_IDS}"
  echo "BATCH_SIZE: ${BATCH_SIZE}"
  echo "NUM_ROUNDS: ${NUM_ROUNDS}"
  echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
  echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
  echo "HF_HOME: ${HF_HOME}"
  echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
  echo "HF_ENDPOINT: ${HF_ENDPOINT}"
  echo "TMPDIR: ${TMPDIR}"

  echo
  echo "===== vLLM import check ====="
  "${PYTHON}" - <<'PY'
import inspect
import sys
import torch
import vllm
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as connector,
)

print("cwd sys.path[0]:", sys.path[0])
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vllm version:", getattr(vllm, "__version__", None))
print("vllm file:", inspect.getfile(vllm))
print("connector file:", inspect.getfile(connector))
print("has token_ids_from_request_data:", hasattr(connector, "token_ids_from_request_data"))
PY

  echo
  echo "Expected vLLM file for this docker demo:"
  echo "  /usr/local/lib/python3.12/dist-packages/vllm/__init__.py"

  run_step "GPU status before run" \
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

  cd "${DEMO_ROOT}"

  run_step "Single sample: generate GT and run vLLM single test" \
    "${PYTHON}" examples/offline_inference/prompt_embed_single_demo.py \
      --model "${MODEL}" \
      --run-root "${SINGLE_ROOT}" \
      --gpu "${GPU}" \
      --seq-len "${SEQ_LEN}" \
      --layer-ids "${LAYER_IDS}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

  run_step "Batch sample: generate single/batch GT and run vLLM batch test" \
    "${PYTHON}" examples/offline_inference/prompt_embed_batch_demo.py \
      --model "${MODEL}" \
      --input-pth "${SINGLE_ROOT}/single_gt.input.pth" \
      --run-root "${BATCH_ROOT}" \
      --gpu "${GPU}" \
      --layer-ids "${LAYER_IDS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-rounds "${NUM_ROUNDS}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

  echo
  echo "===== Output files ====="
  echo "Single root: ${SINGLE_ROOT}"
  echo "Batch root:  ${BATCH_ROOT}"
  echo "Log file:    ${LOG_FILE}"

  print_final_summary
}

main 2>&1 | tee "${LOG_FILE}"
