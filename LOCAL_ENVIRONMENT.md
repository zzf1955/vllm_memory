# vLLM 本机环境说明

本文记录 `/disk_n/zzf` 当前机器环境和在 `/disk_n/zzf/vllm_memory` 跑 vLLM 的推荐方式。
快照时间：2026-05-06 10:23 CST。

## 机器概况

| 项 | 当前值 |
| --- | --- |
| 主机 | `leadtek-4` |
| OS | Ubuntu 24.04.2 LTS, Linux 6.11.0-21-generic |
| 用户 | `leadtek` |
| GPU | 4x NVIDIA GeForce RTX 4090 D, 24 GB each, compute capability 8.9 |
| Driver | 570.133.07 |
| CUDA runtime reported by driver | 12.8 |
| CUDA toolkit | `/usr/local/cuda-12.8`, `nvcc` 12.8.61 |
| GCC/G++ | Ubuntu 12.3.0 |
| Python in current shell | base conda Python 3.13.9 |
| `uv` | 0.11.3 at `/home/leadtek/.local/bin/uv` |
| `conda` | 26.1.0 at `/home/leadtek/miniconda3/bin/conda` |

磁盘状态：

| Mount | Size | Used | Avail | 说明 |
| --- | ---: | ---: | ---: | --- |
| `/` | 937G | 866G | 24G | 根分区接近满，不要把模型、pip、uv、torch 缓存写到默认 home cache |
| `/disk_n` | 19T | 4.4T | 13T | 大盘，vLLM 环境和模型缓存应放这里 |

## GPU 使用状态

快照时 `nvidia-smi` 显示：

| GPU | 显存占用 | 主要进程 |
| --- | ---: | --- |
| 0 | 23823 / 24564 MiB | 已有 `VLLM::EngineCore`，另有 FLIP 评估 |
| 1 | 17 / 24564 MiB | 基本空闲 |
| 2 | 17722 / 24564 MiB | FLIP 训练，占用高 |
| 3 | 11892 / 24564 MiB | 其他用户评估，占用高 |

启动 vLLM 前先检查：

```bash
nvidia-smi
```

当前最适合 smoke test 的是 GPU 1，但这是快照状态，运行前以实时 `nvidia-smi` 为准。
文档里的通用命令不默认写死 `CUDA_VISIBLE_DEVICES`，需要单卡隔离时再按实时占用手动指定。

## `/disk_n/zzf` 目录盘点

`/disk_n/zzf` 总占用约 2.0T。主要目录：

| 路径 | 占用 | 用途判断 |
| --- | ---: | --- |
| `/disk_n/zzf/flip` | 1.1T | FLIP 主项目，含数据、训练输出、权重和文档，可参考其文档组织方式 |
| `/disk_n/zzf/diffusion_policy` | 450G | diffusion policy 项目和数据 |
| `/disk_n/zzf/.cache` | 277G | Hugging Face、uv、torch、ModelScope 等缓存 |
| `/disk_n/zzf/video-gen` | 96G | 视频生成相关环境和测试数据 |
| `/disk_n/zzf/3D-Diffusion-Policy` | 40G | 3D diffusion policy 项目 |
| `/disk_n/zzf/ComfyUI` | 30G | ComfyUI |
| `/disk_n/zzf/conda_envs` | 15G | 放在大盘上的 conda 环境 |
| `/disk_n/zzf/vllm_memory` | 105M | 当前 vLLM 仓库 checkout |

缓存细分：

| 路径 | 占用 | 说明 |
| --- | ---: | --- |
| `/disk_n/zzf/.cache/huggingface` | 256G | Hugging Face hub/datasets/modules |
| `/disk_n/zzf/.cache/uv` | 19G | uv cache |
| `/disk_n/zzf/.cache/torch` | 1.8G | torch cache |
| `/disk_n/zzf/.cache/modelscope` | 526M | ModelScope cache |
| `/disk_n/zzf/.pip_cache` | 334M | pip cache |

Hugging Face cache 里已有若干模型/数据集，和 vLLM 相关的 LLM 包括：

- `models--GSAI-ML--LLaDA-8B-Instruct` (~15G)
- `models--Dream-org--Dream-v0-Instruct-7B` (~15G)
- `models--meta-llama--Llama-3.1-8B-Instruct` / `Meta-Llama-3.1-8B-Instruct` (~15G each)
- `models--meta-llama--Llama-3.2-1B` (~2.4G)

这些目录只能说明本机 cache 中有对应条目；是否可直接加载还取决于模型文件是否完整、token 权限和 vLLM 对该模型架构的支持。

## 当前 vLLM 仓库状态

仓库路径：

```bash
cd /disk_n/zzf/vllm_memory
```

当前 checkout：

| 项 | 当前值 |
| --- | --- |
| remote | `https://github.com/vllm-project/vllm.git` |
| HEAD | `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |
| tag | `v0.18.1` |
| branch | detached HEAD |
| working tree | 当前无已跟踪文件改动；新增本文档后会出现未跟踪/新增文件 |

仓库要求：

- `pyproject.toml` 要求 Python `>=3.10,<3.14`。
- 当前 build dependency pin 到 `torch==2.10.0`。
- CUDA 机器建议用全新的环境，不要混用 FLIP 或其他项目环境。
- 只改 Python 代码时优先用 `VLLM_USE_PRECOMPILED=1 uv pip install -e .`，可避免本地重新编译 CUDA/C++。
- 改 `csrc/`、kernel 或 C++ 时必须完整源码编译：`uv pip install -e .`。

当前没有 `.venv`。已有的 `ppvllm` conda 环境可作为历史参考，但不建议拿来开发当前 checkout：

```text
conda env: ppvllm
Python: 3.10.18
Torch: 2.7.1+cu128
vLLM: 0.10.1.dev98+g1cbf951ba.d20250727
```

`flip` 环境里能从当前目录 import 本仓库源码，但它的 torch 是 `2.11.0+cu128`，也不适合作为当前 vLLM 仓库的标准开发环境。

## 推荐环境变量

根分区剩余空间很少，跑 vLLM 前建议固定这些路径：

```bash
export CONDA_PREFIX=/disk_n/zzf/conda_envs/vllm_memory
export PATH="$CONDA_PREFIX/bin:$PATH"

export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/disk_n/zzf/.cache/huggingface/hub
export TORCH_HOME=/disk_n/zzf/.cache/torch
export UV_CACHE_DIR=/disk_n/zzf/.cache/uv
export PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
export VLLM_CACHE_ROOT=/disk_n/zzf/.cache/vllm
export TMPDIR=/disk_n/zzf/tmp
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"
```

本机有 `mihomo` 代理监听 `20171`。需要下载模型或 wheel 时可临时开启：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://127.0.0.1:20171
export https_proxy=http://127.0.0.1:20171
export no_proxy=localhost,127.0.0.1
```

如果不需要外网，保持代理变量为空即可。

当前推荐使用已经创建好的 conda prefix 环境：

```bash
source /home/leadtek/miniconda3/etc/profile.d/conda.sh
conda activate /disk_n/zzf/conda_envs/vllm_memory

export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export UV_CACHE_DIR=/disk_n/zzf/.cache/uv
export PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
export TMPDIR=/disk_n/zzf/tmp
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://127.0.0.1:20171
export https_proxy=http://127.0.0.1:20171
export no_proxy=localhost,127.0.0.1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" "$TMPDIR"
```

该环境当前检查结果：

```text
python: /disk_n/zzf/conda_envs/vllm_memory/bin/python
torch: 2.10.0+cu128, cuda_available=True
transformers: 4.57.6
```

## 运行前资源检查

每次启动 vLLM、编译 CUDA 扩展、跑测试或 benchmark 前，先看系统内存和显卡占用，避免把共享机器打爆：

```bash
free -h
nvidia-smi
ps -eo pid,user,comm,args --sort=-%mem | head -40
```

如果要持续观察：

```bash
watch -n 1 'free -h; echo; nvidia-smi'
```

判断原则：

- 大模型服务启动前，目标 GPU 至少应有足够连续空闲显存；不要只看 GPU util，也要看显存占用和已有进程用户。
- 编译 CUDA/FlashAttention 前，先确认 CPU 内存和 swap 余量；高并行编译比普通 Python 测试更容易触发 OOM。
- 端口、GPU、长时间训练进程都是共享资源；不确定时先查 `ss -ltnp`、`nvidia-smi` 和 `ps`。

## 安装方式

### Python-only 开发安装

适合只改 Python 代码或先把服务跑起来：

```bash
cd /disk_n/zzf/vllm_memory

export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export TORCH_HOME=/disk_n/zzf/.cache/torch
export UV_CACHE_DIR=/disk_n/zzf/.cache/uv
export PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
export VLLM_CACHE_ROOT=/disk_n/zzf/.cache/vllm
export TMPDIR=/disk_n/zzf/tmp

# 不在这里限制 CUDA wheel 版本；让 uv / vLLM 根据当前环境自动选择。

uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install -r requirements/lint.txt
pre-commit install

VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

安装后做最小检查：

```bash
python - <<'PY'
import torch
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("vllm", getattr(vllm, "__version__", "unknown"))
PY
```

### 完整源码编译

只有改 C++/CUDA/kernel 时使用。编译会占用更多 CPU、内存和临时目录空间：

```bash
cd /disk_n/zzf/vllm_memory
source .venv/bin/activate

export MAX_JOBS=64
export NVCC_THREADS=4

uv pip install -e .
```

如果已有 torch 且需要复用它，可参考仓库自带流程：

```bash
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

### FlashAttention 编译

需要本地编译 FlashAttention 时，worker 并行数按 64 写：

```bash
cd /disk_n/zzf/vllm_memory
source .venv/bin/activate

free -h
nvidia-smi

export MAX_JOBS=64
uv pip install flash-attn --no-build-isolation
```

如果 `free -h` 显示内存余量不足，先不要启动编译；等其他训练/编译结束或释放资源后再跑。

## 启动 vLLM 服务

端口状态快照：

- `0.0.0.0:8000` 已被其他用户的 vLLM OpenAI API server 占用，命令是 `Qwen/Qwen2.5-7B-Instruct`。
- `*:20171` 是本机 `mihomo` 代理端口。

因此本用户测试建议避开 8000，用 8001 或其他空闲端口。

单卡 smoke test 示例：

```bash
cd /disk_n/zzf/vllm_memory
source .venv/bin/activate

export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export VLLM_CACHE_ROOT=/disk_n/zzf/.cache/vllm
export TMPDIR=/disk_n/zzf/tmp

free -h
nvidia-smi

vllm serve facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 2048
```

另一个 shell 测试：

```bash
curl http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, my name is",
    "max_tokens": 16
  }'
```

如果使用 chat model：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-name-or-local-path>",
    "messages": [{"role": "user", "content": "用一句话介绍 vLLM"}],
    "max_tokens": 64
  }'
```

## 离线推理 smoke test

安装完成后也可以先跑离线脚本，避免端口冲突：

```bash
cd /disk_n/zzf/vllm_memory
source .venv/bin/activate

free -h
nvidia-smi

python examples/basic/offline_inference/basic.py
```

如需用更小的自定义脚本：

```bash
free -h
nvidia-smi

python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.30,
    max_model_len=2048,
)
outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(max_tokens=16, temperature=0.0),
)
print(outputs[0].outputs[0].text)
PY
```

## 2026-05-06 实际跑通记录

本次已按用户要求创建 conda prefix 环境：

```bash
conda activate /disk_n/zzf/conda_envs/vllm_memory
```

实际版本：

```text
Python: 3.12.13
torch: 2.10.0+cu128
torch CUDA: 12.8
vLLM: 0.18.1
vLLM extension: /disk_n/zzf/vllm_memory/vllm/_C.abi3.so
```

安装过程说明：

- 先试过 `VLLM_USE_PRECOMPILED=1 uv pip install -e .`，但当前 detached checkout 找不到 main branch merge-base，安装器 fallback 到 nightly wheel；该 wheel 与当前源码/torch ABI 不匹配，报错为 `undefined symbol: _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib`。
- 已删除 nightly wheel 留下的 ignored `.so` 产物，改用本机源码编译。
- 本机源码编译命令使用了 `MAX_JOBS=64` 和 `NVCC_THREADS=4`；日志显示 `Using MAX_JOBS=64 as the number of jobs.` 和 `Using NVCC_THREADS=4 as the number of nvcc threads.`。
- 未设置 `VLLM_MAIN_CUDA_VERSION`、`VLLM_PRECOMPILED_WHEEL_VARIANT`、`UV_TORCH_BACKEND` 等 CUDA wheel 限制变量。
- 编译生成的 `.deps/`、`vllm/*.so`、`vllm/vllm_flash_attn/*.so`、`vllm.egg-info/`、`__pycache__/` 等是 ignored 构建产物；`git status --short` 只显示本文档为新增文件。

模型下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export HF_HUB_DISABLE_XET=1
export http_proxy=http://127.0.0.1:20171
export https_proxy=http://127.0.0.1:20171
export HTTP_PROXY=http://127.0.0.1:20171
export HTTPS_PROXY=http://127.0.0.1:20171
```

`Qwen/Qwen3-0.8B` 在 hf-mirror API 上返回 404；已用官方存在且更小的 `Qwen/Qwen3-0.6B` 跑通。模型 cache 路径：

```text
/disk_n/zzf/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

验证命令使用 GPU 3 做低显存 smoke test，因为运行前 `nvidia-smi` 显示 GPU 3 剩余显存最多且 util 低：

```bash
export CUDA_VISIBLE_DEVICES=3
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_LOGGING_LEVEL=INFO

python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=256,
    max_num_seqs=1,
    gpu_memory_utilization=0.15,
    enforce_eager=True,
    trust_remote_code=True,
    disable_log_stats=True,
)
outputs = llm.generate(
    ["用一句话介绍 vLLM。"],
    SamplingParams(temperature=0.0, max_tokens=24),
)
print(outputs[0].outputs[0].text.strip())
PY
```

验证结果：

```text
Resolved architecture: Qwen3ForCausalLM
Using FLASH_ATTN attention backend
Using FlashAttention version 2
Model loading took 1.12 GiB memory
Available KV cache memory: 2.28 GiB
GPU KV cache size: 21,344 tokens
OUTPUT: vLLM 是一个高性能的模型推理框架，支持多种模型类型，包括但不限于 transformer、llama、ll
```

验证结束后已检查 `nvidia-smi` 和进程列表，没有本次 smoke test 的残留 vLLM 进程。

## 测试和 lint

按照 `AGENTS.md`，测试依赖从 `requirements/test.txt` 读取。最小测试依赖：

```bash
uv pip install pytest pytest-asyncio tblib
```

按需补齐全部测试依赖：

```bash
uv pip install -r requirements/test.txt
```

常用命令：

```bash
pytest tests/path/to/test.py -v -s -k test_name
pytest tests/path/to/dir -v -s
pre-commit run
pre-commit run --all-files
pre-commit run ruff-check --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

## 贡献注意事项

本仓库的 `AGENTS.md` 是硬约束：

- 不要发纯代码 agent PR；提交人必须理解并能解释所有改动。
- 不要做低价值的一次性 busywork PR。
- 准备 PR 前必须查重：

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

PR 描述需要写清：

- 为什么不是重复已有 PR。
- 实际跑过哪些测试及结果。
- 明确说明使用了 AI assistance。

commit message 需要按项目要求添加合适的 attribution / sign-off trailer。
