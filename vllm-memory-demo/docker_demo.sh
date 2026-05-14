#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${DEMO_ROOT}/examples/offline_inference" ]]; then
  echo "ERROR: missing bundled examples: ${DEMO_ROOT}/examples/offline_inference" >&2
  echo "Run setup.sh first, or place examples under the demo root." >&2
  exit 1
fi

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
: "${MEM_LEN:=8}"
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
: "${HF_HUB_DISABLE_PROGRESS_BARS:=0}"
: "${HF_HUB_VERBOSITY:=info}"
: "${TRANSFORMERS_VERBOSITY:=info}"
: "${PYTHONUNBUFFERED:=1}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE
export UV_CACHE_DIR
export PIP_CACHE_DIR
export TMPDIR
export HF_ENDPOINT
export HF_HUB_DISABLE_XET
export HF_HUB_DOWNLOAD_TIMEOUT
export HF_HUB_ETAG_TIMEOUT
export HF_HUB_DISABLE_PROGRESS_BARS
export HF_HUB_VERBOSITY
export TRANSFORMERS_VERBOSITY
export PYTHONUNBUFFERED
export CUDA_VISIBLE_DEVICES="${GPU}"

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

download_model_snapshot() {
  "${PYTHON}" - "${MODEL}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model = sys.argv[1]
print(f"Downloading/checking model snapshot: {model}", flush=True)
print(f"HF_HOME={os.environ.get('HF_HOME')}", flush=True)
print(
    "HUGGINGFACE_HUB_CACHE="
    f"{os.environ.get('HUGGINGFACE_HUB_CACHE')}",
    flush=True,
)
path = snapshot_download(
    repo_id=model,
    repo_type="model",
    resume_download=True,
)
print(f"Model snapshot is ready: {path}", flush=True)
PY
}

print_final_summary() {
  echo
  echo "===== FINAL SUMMARY ====="
  "${PYTHON}" - \
    "${SINGLE_ROOT}/single_gt.input.pth" \
    "${LOG_FILE}" \
    "${BATCH_SIZE}" \
    "${NUM_ROUNDS}" \
    "${MEM_LEN}" <<'PY'
import ast
import sys
from pathlib import Path

import torch

input_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
batch_size = int(sys.argv[3])
num_rounds = int(sys.argv[4])
mem_len = int(sys.argv[5])


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


def print_compare(name: str, data: dict | None) -> None:
    if data is None:
        print(f"{name}: MISSING")
        return
    ok = data.get("status") == "PASS"
    print(
        f"{name}: {pass_fail(ok)} "
        f"count={data.get('count')} failed={data.get('failed')} "
        f"min_abs={data.get('min_abs')} max_abs={data.get('max_abs')} "
        f"mean_abs_avg={data.get('mean_abs_avg')} "
        f"shape={data.get('shape')}"
    )


input_data = (
    torch.load(input_path, map_location="cpu", weights_only=False)
    if input_path.exists()
    else {}
)
input_ids = input_data.get("input_ids")
prompt_embeds = input_data.get("prompt_embeds")
attention_mask = input_data.get("attention_mask")

print("[Input]")
print("input sequence: <random emb><prompt emb><random emb>")
if input_ids is not None:
    print(f"input_ids shape: {tuple(input_ids.shape)}")
if attention_mask is not None:
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
if prompt_embeds is not None:
    print(f"prompt_embeds shape: {tuple(prompt_embeds.shape)}")
    hidden_size = int(prompt_embeds.shape[-1])
    prompt_len = int(prompt_embeds.shape[0])
    print(f"random_emb prefix shape: ({mem_len}, {hidden_size})")
    print(f"random_emb suffix shape: ({mem_len}, {hidden_size})")
    print(f"full_prompt_emb shape: ({mem_len + prompt_len + mem_len}, {hidden_size})")
    print(
        "batch variants: "
        f"batch_size={batch_size}, num_rounds={num_rounds}, "
        "prompt_emb is scaled per batch index"
    )

lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

print()
print("[Output]")
print_compare(
    "single_vs_single",
    parse_dict(lines, "single_vs_single_summary:"),
)
print_compare(
    "batch_vs_batch",
    parse_dict(lines, "batch_vs_batch_summary:"),
)
memory_result = last_value(lines, "MEMORY_DEMO_RESULT:")
print(f"memory_demo_result: {memory_result or 'MISSING'}")

overall_ok = memory_result == "PASS"
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
  echo "MEM_LEN: ${MEM_LEN}"
  echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
  echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
  echo "HF_HOME: ${HF_HOME}"
  echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
  echo "HF_ENDPOINT: ${HF_ENDPOINT}"
  echo "TMPDIR: ${TMPDIR}"
  echo "HF_HUB_VERBOSITY: ${HF_HUB_VERBOSITY}"
  echo "TRANSFORMERS_VERBOSITY: ${TRANSFORMERS_VERBOSITY}"

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

  run_step "Download/check Hugging Face model snapshot" \
    download_model_snapshot

  cd "${DEMO_ROOT}"

  run_step "Generate base prompt embeddings" \
    "${PYTHON}" examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py \
      --model "${MODEL}" \
      --seq-len "${SEQ_LEN}" \
      --device cuda \
      --dtype "${DTYPE}" \
      --output "${SINGLE_ROOT}/single_gt.pth"

  run_step "Memory prompt test: [random emb][prompt emb][random emb]" \
    "${PYTHON}" examples/offline_inference/prompt_embed_memory_batch_demo.py \
      --model "${MODEL}" \
      --input-pth "${SINGLE_ROOT}/single_gt.input.pth" \
      --work-dir "${BATCH_ROOT}" \
      --layer-ids "${LAYER_IDS}" \
      --mem-len "${MEM_LEN}" \
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
