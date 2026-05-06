# Prompt Embeds + Extract Hidden States Change Log

## 背景

目标是在 vLLM 0.18.1 或更新版本上，用 `prompt_embeds` 作为输入，并通过
`extract_hidden_states` 导出 prompt token 对应的 hidden states。当前验证重点是：

- 最小修改，优先只改 KV connector。
- 跑通单样例。
- prompt-embeds 路径和普通 `LLM.generate` token prompt 路径对比一致。
- 可选和 Hugging Face 原生 forward 保存的 ground truth 对比。

参考：

- `https://vllm-project.github.io/2026/03/30/extract-hidden-states.html`
- `https://github.com/vllm-project/vllm/blob/main/examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py`

## 修改文件

### `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`

这是核心功能修改，也是唯一的 vLLM 运行时逻辑修改。

#### 修改原因

`ExampleHiddenStatesConnector.build_connector_meta()` 原来使用：

```python
token_ids = new_req.prompt_token_ids or []
```

以及 cached request 分支里的：

```python
token_ids=cached_req.prompt_token_ids or []
```

当请求输入是纯 `prompt_embeds` 时，`prompt_token_ids` 是 `None`，这两处会把
token ids 当成空列表。因此 connector 后续保存 hidden states 时会用长度 0：

```python
hidden_states = extract_from_kv_cache(
    kv_layer, request.slot_mapping, request.token_ids.shape[0]
)
```

结果是 prompt embeddings 请求无法正确保存 prompt 长度对应的 hidden states。

#### 修改内容

新增 import：

```python
from vllm.utils import length_from_prompt_token_ids_or_embeds
```

新增 helper：

```python
def token_ids_from_request_data(request_data: NewRequestData) -> list[int]:
    """Return real token IDs, or prompt-embed placeholders when IDs are absent."""
    if request_data.prompt_token_ids is not None:
        return request_data.prompt_token_ids

    prompt_len = length_from_prompt_token_ids_or_embeds(
        request_data.prompt_token_ids,
        request_data.prompt_embeds,
    )
    return [0] * prompt_len
```

然后把新请求和 cached request 的 token ids 获取逻辑改为：

```python
token_ids = token_ids_from_request_data(new_req)
```

以及：

```python
token_ids=token_ids_from_request_data(cached_req)
```

#### 行为变化

- 普通 token prompt：保存真实 `prompt_token_ids`，行为不变。
- 纯 `prompt_embeds` prompt：没有真实 token ids 时，按 `prompt_embeds` 的长度生成
  `[0] * prompt_len` 作为 safetensors 里的占位 `token_ids`。
- hidden states 保存长度从错误的 0 变为真实 prompt embeds 长度。

当前占位 token ids 只用于让 connector 知道保存多少 token 的 hidden states；真实输入
embedding 已经通过 vLLM 的 `prompt_embeds` 路径进入模型。

### `examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py`

这是新增脚本，用 Hugging Face 原生 forward 保存 ground truth。

#### 功能

给定文本和 `seq_len`，脚本：

1. 用 HF tokenizer 编码文本。
2. 截断到 `seq_len` 个 token。
3. 用模型 input embedding 层生成 prompt embeddings。
4. 通过 `inputs_embeds=prompt_embeds` 跑一次 HF forward。
5. 设置 `output_hidden_states=True`、`use_cache=False`。
6. 分开保存 prompt、输入 tensor、输出 tensor。

#### 默认参数

默认模型：

```text
Qwen/Qwen3-0.6B
```

默认文本：

```text
vLLM can accept prompt embeddings directly and extract hidden states from selected layers for fast inference experiments.
```

默认 `seq_len`：

```text
16
```

#### 输出文件

如果传入：

```bash
--output /disk_n/zzf/tmp/qwen3_gt_gpu.pth
```

会写出：

```text
/disk_n/zzf/tmp/qwen3_gt_gpu.prompt.txt
/disk_n/zzf/tmp/qwen3_gt_gpu.input.pth
/disk_n/zzf/tmp/qwen3_gt_gpu.output.pth
```

其中：

- `.prompt.txt` 保存原始文本和截断后的 effective prompt 文本。
- `.input.pth` 保存 `input_ids`、`attention_mask`、`prompt_embeds`、metadata。
- `.output.pth` 保存 `hidden_states`、`layer_hidden_states`、metadata。

#### 已生成的 ground truth

命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py \
  --seq-len 16 \
  --device cuda \
  --dtype bfloat16 \
  --output /disk_n/zzf/tmp/qwen3_gt_gpu.pth
```

输出：

```text
/disk_n/zzf/tmp/qwen3_gt_gpu.prompt.txt
/disk_n/zzf/tmp/qwen3_gt_gpu.input.pth
/disk_n/zzf/tmp/qwen3_gt_gpu.output.pth
```

关键 tensor shape：

```text
input_ids: (16,)
prompt_embeds: (16, 1024)
hidden_states: (29, 16, 1024)
layer_hidden_states: (28, 16, 1024)
```

截断后的 effective prompt：

```text
vLLM can accept prompt embeddings directly and extract hidden states from selected layers for
```

### `examples/offline_inference/prompt_embed_extract_hidden_states.py`

这是新增验证 demo，用来跑通 vLLM 的 prompt embeddings + hidden states 提取路径。

#### 功能

脚本从 `save_prompt_embed_hidden_states_ground_truth.py` 生成的文件中读取数据，然后：

1. 从 `.input.pth` 读取 `input_ids` 和 `prompt_embeds`。
2. 自动从同名 `.prompt.txt` 读取 effective prompt 文本，也可以显式传 `--prompt-txt`。
3. 使用普通 `LLM.generate` + `prompt_token_ids` 跑 token prompt baseline。
4. 使用 `AsyncLLM.generate` + `{"prompt": ..., "prompt_embeds": ...}` 跑 prompt embeddings。
5. 两条路径都配置：

```python
speculative_config={
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": layer_ids,
        }
    },
}
```

以及：

```python
kv_transfer_config={
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "shared_storage_path": storage_path,
    },
}
```

6. demo 里的 prompt-embeds / hidden-states engine 还显式设置：

```python
enable_prefix_caching=False
```

这避免同一段 `prompt` 文本但不同 `prompt_embeds` 的请求被 token/text prefix cache
错误地视为同一前缀。当前 demo 更关注 prompt embeddings 本身的等价性和 connector
导出结果，因此关闭 prefix cache 更稳妥。

7. 从 connector 保存的 safetensors 里读取：

```text
hidden_states
token_ids
```

8. 比较：

- `LLM.generate` token prompt 生成 token id。
- `AsyncLLM.generate` prompt embeds 生成 token id。
- 两条 vLLM 路径提取的 hidden states。
- 可选和 HF `.output.pth` 中对应 layer hidden states 比较。

#### 默认行为

默认只提取第 1 层：

```bash
--layer-ids 1
```

原因是当前 GPU 1 上已有其他进程占用较多显存。全 28 层时在 warmup 末尾触发过 OOM。
需要全层验证时可以显式传：

```bash
--layer-ids all
```

默认 eager：

```bash
--enforce-eager
```

如果需要开启 vLLM compile/cudagraph，可以传：

```bash
--no-enforce-eager
```

#### 已跑通命令

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_extract_demo \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

关键输出：

```text
Prompt text: vLLM can accept prompt embeddings directly and extract hidden states from selected layers for
Prompt length: 16
Layer ids: [1]
LLM token generated ids: [279]
Async prompt-embeds generated ids: [279]
Generated ids match: True
LLM token hidden shape: (16, 1, 1024)
Prompt-embeds hidden shape: (16, 1, 1024)
Prompt-embeds token_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Prompt-embeds vs token hidden states: {'allclose': True, 'max_abs': 0.0, 'mean_abs': 0.0, 'shape': (16, 1, 1024)}
Prompt-embeds vs HF hidden states: {'allclose': True, 'max_abs': 0.00390625, 'mean_abs': 1.2211501598358154e-05, 'shape': (16, 1, 1024)}
```

生成的 safetensors 示例：

```text
/disk_n/zzf/tmp/qwen3_extract_demo/llm-token-prompt/0-b6e0ae2f.safetensors
/disk_n/zzf/tmp/qwen3_extract_demo/async-prompt-embeds/prompt-embeds-extract-hidden-states-9e33196b.safetensors
```

#### 已跑通命令：全 28 层

GPU 1 当时仍被其他进程占用，本次全层验证改用空闲的 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_extract_demo_all_layers_single \
  --layer-ids all \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7
```

关键结果：

```text
Layer ids: [1, 2, ..., 28]
LLM token generated ids: [279]
Async prompt-embeds generated ids: [279]
Generated ids match: True
LLM token hidden shape: (16, 28, 1024)
Prompt-embeds hidden shape: (16, 28, 1024)
Prompt-embeds vs token hidden states: {'allclose': True, 'max_abs': 0.0, 'mean_abs': 0.0, 'shape': (16, 28, 1024)}
```

和 HF 原生 `hidden_states` 的逐层对比中，1 到 27 层是逐层累积的数值差异；第 28 层差异很大，
更像是 vLLM `extract_hidden_states` 抽取位置和 HF `hidden_states[-1]` 是否包含 final norm
的语义差异。这个差异不影响本 demo 的核心判断：vLLM token prompt 和 vLLM prompt embeds
在同一抽取路径下全层完全一致。

### `examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py`

这是新增并发压测脚本，用来验证多个 `AsyncLLM.generate()` prompt-embeds 请求同时
运行时，connector 保存的 hidden states 是否长度正确、文件是否串线、结果是否稳定。

#### 功能

脚本启动一个 `AsyncLLM` engine，然后：

1. 从 `.input.pth` 读取同一份 `prompt_embeds`。
2. 根据 `--batch-size` 构造多个不同的 prompt embedding 变体：

```python
variant_i = prompt_embeds * (1.0 + i * variant_scale_step)
```

默认：

```bash
--variant-scale-step 0.01
```

3. 先对每个 batch 变体跑一次顺序 reference。
4. 再按 `--batch-size` 同时提交多个 `AsyncLLM.generate()` 请求。
5. 连续跑 `--num-rounds` 轮。
6. 每个并发请求都读取对应 safetensors，并检查：

- `hidden_states` shape 是否等于期望 shape。
- `token_ids` 长度是否等于 prompt 长度。
- 生成 token ids 是否和该 batch 位置的顺序 reference 一致。
- hidden states 是否和该 batch 位置的顺序 reference allclose。

这个设计比所有请求用同一份 embedding 更严格，因为如果并发时 connector 或调度把请求
混了，不同 embedding 变体的 hidden states 会和各自 reference 对不上。

#### 已跑通命令 1：batch=4，10 轮

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_concurrent_demo_variants \
  --batch-size 4 \
  --num-rounds 10 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

关键结果：

```text
Batch size: 4
Num rounds: 10
Variant scale step: 0.01
Reference variant 0 vs HF: {'allclose': True, 'max_abs': 0.00390625, 'mean_abs': 1.2211501598358154e-05, 'shape': (16, 1, 1024)}
Stress test summary:
  total_requests: 40
  failed_compares: 0
  max_abs: 0.03125
  mean_abs_avg: 0.0004284078604541719
  expected_shape: (16, 1, 1024)
  generated_ids: [[279], [279], [279], [279]]
```

#### 已跑通命令 2：batch=8，20 轮

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_concurrent_demo_variants_b8 \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

关键结果：

```text
Batch size: 8
Num rounds: 20
Variant scale step: 0.01
Reference variant 0 vs HF: {'allclose': True, 'max_abs': 0.00390625, 'mean_abs': 1.2211501598358154e-05, 'shape': (16, 1, 1024)}
Stress test summary:
  total_requests: 160
  failed_compares: 0
  max_abs: 0.03125
  mean_abs_avg: 0.0007740353757981211
  expected_shape: (16, 1, 1024)
  generated_ids: [[279], [279], [279], [279], [279], [279], [279], [279]]
```

#### 并发测试结论

- 单层 hidden state、prompt length 16、`max_tokens=1` 下，`AsyncLLM` 并发
  prompt-embeds 请求没有复现 zero-length hidden states、文件缺失或请求串线问题。
- batch=4 连续 10 轮，总 40 个并发请求通过。
- batch=8 连续 20 轮，总 160 个并发请求通过。
- 全 28 层并发也做了 b=4/r=10 的定位测试；请求没有崩溃、shape 和生成 token 正常，但
  这个脚本使用“单条 sequential reference”作为 strict reference，和 batch 并发路径的
  deep raw hidden states 存在 FlashAttention/batching 数值差，结果为 `failed_compares=30/40`、
  `max_abs=32.0`。同一批中的非首个请求彼此完全一致，未观察到请求串线。
- 对全层场景，更可靠的验证是 MEM 脚本里的 batch `LLM.generate` baseline vs batch
  `AsyncLLM.generate`，见下一节。
- 已新增 `--mode gt-only|async-only` 用于把 GT 生成和 AsyncLLM 并发验证拆到不同 Python
  进程里，避免同一进程连续创建多个 vLLM engine 后 AsyncLLM 初始化不稳定。
- 已新增两种 GT：
  - `llm-single-ground-truth`：逐条单独 `LLM.generate`。
  - `llm-batch-ground-truth`：同一批请求一次性 `LLM.generate`。
- 全 28 层、batch=4 时，batch GT 和 single GT 本身不同：

```text
Batch GT vs single GT:
  failed_compares: 3 / 4
  max_abs: 32.0
```

- 使用独立 Async-only 进程跑全 28 层、batch=4、rounds=10 后，Async batch 和 batch GT
  完全一致；Async batch 和 single GT 仍然失败：

```text
total_requests: 40
batch_gt_failed_compares: 0
batch_gt_max_abs: 0.0
batch_gt_mean_abs_avg: 0.0
single_gt_failed_compares: 30
single_gt_max_abs: 32.0
single_gt_mean_abs_avg: 0.03352775704115629
```

- 全 28 层、batch=8 时，batch GT 和 single GT 差异更大：

```text
Batch GT vs single GT:
  failed_compares: 7 / 8
  max_abs: 64.0
```

- 使用独立 Async-only 进程跑全 28 层、batch=8、rounds=20 后，160 个请求中有 2 个请求
  相比 batch GT 失败，均发生在 round 13 的 b0/b1：

```text
total_requests: 160
batch_gt_failed_compares: 2
batch_gt_max_abs: 96.0
batch_gt_mean_abs_avg: 0.0005505115259438753
single_gt_failed_compares: 141
single_gt_max_abs: 96.0
single_gt_mean_abs_avg: 0.03954133654478938
```

定位结果：

```text
round 13 b0: vs batch GT max_abs=96.0
round 13 b1: vs batch GT max_abs=32.0
```

这两个异常输出不等于 batch=2 GT，也不是简单的 batch index 互换。当前判断：

- “batch 输出 vs single GT”失败是确定的 reference mismatch。
- “Async batch vs batch GT”在 batch=4 下稳定通过。
- “Async batch vs batch GT”在 batch=8/r=20 下仍存在少量异常，需要继续修 connector
  或调度/保存路径，重点看 round 内部分请求是否发生了 prefill 分组、cached request 分支或
  connector metadata/slot mapping 状态错误。

### `examples/offline_inference/prompt_embed_async_batch_only.py`

这是新增的纯 AsyncLLM batch 输出脚本。

#### 功能

只启动 `AsyncLLM`，不 import/use `LLM`，输入一组 prompt embeddings batch，连续跑多轮，
并通过 `ExampleHiddenStatesConnector` 保存每个请求的 hidden states safetensors。

用途是把 Async batch 验证和 GT 生成拆到两个独立 Python 进程里，避免同一进程连续创建
`LLM` 和 `AsyncLLM` engine 带来的初始化干扰。

#### 已跑命令

batch=4/r=10：

```bash
CUDA_VISIBLE_DEVICES=3 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_async_batch_only.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_concurrent_batch_vs_single_gt_all_layers_b4_r1 \
  --layer-ids all \
  --batch-size 4 \
  --num-rounds 10 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7
```

batch=8/r=20：

```bash
CUDA_VISIBLE_DEVICES=3 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_async_batch_only.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_concurrent_batch_vs_single_gt_all_layers_b8_gt \
  --layer-ids all \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7
```

### `examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py`

这是新增 MEM 拼接版并发压测脚本。

#### 输入输出格式

输入 prompt embeddings 被拼成：

```text
<MEM> <prompt_emb> <MEM>
```

当前默认：

```text
MEM prefix length = 8
prompt_emb length = 16
MEM suffix length = 8
full input length = 8 + 16 + 8 = 32
```

MEM 是固定随机向量，在整个脚本运行期间保持不变：

```text
mem_seed = 1234
mem_scale = 1.0
```

输出 hidden states 的比较区域拆成两段：

```text
prefix region = <MEM prefix> <prompt hidden states>  # length 24
memory region = <MEM suffix hidden states>           # length 8
```

其中最后的 `<MEM suffix hidden states>` 是当前更关心的 memory 输出。

#### Baseline 和并发对比

脚本先用 vLLM `LLM.generate` 生成 baseline hidden states：

- 输入同样是 `<MEM> <prompt_emb> <MEM>`。
- 通过 `ExampleHiddenStatesConnector` 保存 baseline safetensors。
- baseline 每个 batch 变体各保存一份。

然后再启动一个 `AsyncLLM` engine 做并发请求：

- 同样输入 `<MEM> <prompt_emb> <MEM>`。
- batch 内 prompt 部分仍做不同 scale 变体，MEM prefix/suffix 固定不变。
- 每个并发输出都和对应 batch 位置的 `LLM.generate` baseline 对比。
- 分别统计 prefix region 和 memory region 的差值。

#### 已跑通命令 1：MEM batch=4，10 轮

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_memory_concurrent_demo_b4 \
  --batch-size 4 \
  --num-rounds 10 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

关键结果：

```text
Prompt len: 16
MEM len: 8
Full prompt len: 32
Prefix compare len: 24
Memory compare len: 8
Batch size: 4
Num rounds: 10
Memory stress test summary:
  total_requests: 40
  failed_compares: 0
  prefix_max_abs: 0.03125
  prefix_mean_abs_avg: 0.00029453232418745755
  memory_max_abs: 0.03125
  memory_mean_abs_avg: 0.001040828274562955
  expected_shape: (32, 1, 1024)
  baseline_generated_ids: [[320], [320], [320], [320]]
```

#### 已跑通命令 2：MEM batch=8，20 轮

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_memory_concurrent_demo_b8 \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

关键结果：

```text
Prompt len: 16
MEM len: 8
Full prompt len: 32
Prefix compare len: 24
Memory compare len: 8
Batch size: 8
Num rounds: 20
Memory stress test summary:
  total_requests: 160
  failed_compares: 0
  prefix_max_abs: 0.015625
  prefix_mean_abs_avg: 0.00015344673884101211
  memory_max_abs: 0.03125
  memory_mean_abs_avg: 0.000661272497382015
  expected_shape: (32, 1, 1024)
  baseline_generated_ids: [[320], [320], [320], [320], [320], [320], [320], [320]]
```

#### MEM 拼接测试结论

- 当前输入拼接 `<MEM(8)> <prompt_emb(16)> <MEM(8)>` 跑通。
- 输出比较拆成 prefix region 长度 24 和 memory region 长度 8。
- batch=4 连续 10 轮，总 40 个并发请求通过。
- batch=8 连续 20 轮，总 160 个并发请求通过。
- prefix region 和 memory region 分别和 `LLM.generate` baseline 对比，均
  `failed_compares=0`。

#### 已跑通命令 3：MEM 全 28 层，batch=4，10 轮

```bash
CUDA_VISIBLE_DEVICES=0 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_memory_concurrent_demo_all_layers_b4_noprefix \
  --layer-ids all \
  --batch-size 4 \
  --num-rounds 10 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7
```

关键结果：

```text
total_requests: 40
failed_compares: 0
prefix_max_abs: 0.0
memory_max_abs: 0.0
expected_shape: (32, 28, 1024)
baseline_generated_ids: [[320], [320], [320], [320]]
```

#### 已跑通命令 4：MEM 全 28 层，batch=8，20 轮

```bash
CUDA_VISIBLE_DEVICES=0 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_memory_concurrent_demo_all_layers_b8_noprefix \
  --layer-ids all \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7
```

关键结果：

```text
total_requests: 160
failed_compares: 0
prefix_max_abs: 0.0
memory_max_abs: 0.0
expected_shape: (32, 28, 1024)
baseline_generated_ids: [[320], [320], [320], [320], [320], [320], [320], [320]]
```

### `demo.sh`

这是新增一键启动脚本。所有参数都已经内置，用户只需要先激活环境，然后在仓库根目录
运行：

```bash
conda activate /disk_n/zzf/conda_envs/vllm_memory
cd /disk_n/zzf/vllm_memory
bash demo.sh
```

脚本内置参数：

```text
GPU = 1
MODEL = Qwen/Qwen3-0.6B
SEQ_LEN = 16
MEM_LEN = 8
BATCH_SIZE = 8
NUM_ROUNDS = 20
LAYER_IDS = 1
GPU_MEMORY_UTILIZATION = 0.2
MAX_MODEL_LEN = 512
DTYPE = bfloat16
RUN_ROOT = /disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo
```

脚本会顺序执行：

1. 打印 GPU 显存状态。
2. 生成 HF ground truth：
   - `qwen3_gt.prompt.txt`
   - `qwen3_gt.input.pth`
   - `qwen3_gt.output.pth`
3. 跑单样例 prompt-embeds vs `LLM.generate` baseline。
4. 跑普通 prompt-embeds 并发压测，默认 batch=8、rounds=20。
5. 跑 `<MEM(8)> <prompt_emb(16)> <MEM(8)>` 并发压测，默认 batch=8、rounds=20。
6. 输出关键 summary，并把完整日志保存到：

```text
/disk_n/zzf/tmp/vllm_prompt_embed_hidden_states_demo/logs/demo_*.log
```

脚本中已经内置环境变量：

```bash
HF_HOME=/disk_n/zzf/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
UV_CACHE_DIR=/disk_n/zzf/.cache/uv
PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
TMPDIR=/disk_n/zzf/tmp
HF_ENDPOINT=https://hf-mirror.com
http_proxy=http://127.0.0.1:20171
https_proxy=http://127.0.0.1:20171
no_proxy=localhost,127.0.0.1
```

脚本会使用当前激活环境里的 `python`。如果当前 `python` 无法 `import vllm`，会直接
报错并提示先激活 `/disk_n/zzf/conda_envs/vllm_memory`。

### `LOCAL_ENVIRONMENT.md`

这是环境记录文档，记录了当前用户指定的运行环境。

关键内容包括：

```bash
export HF_HOME=/disk_n/zzf/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/disk_n/zzf/.cache/huggingface/hub
export UV_CACHE_DIR=/disk_n/zzf/.cache/uv
export PIP_CACHE_DIR=/disk_n/zzf/.pip_cache
export TMPDIR=/disk_n/zzf/tmp
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://127.0.0.1:20171
export https_proxy=http://127.0.0.1:20171
export no_proxy=localhost,127.0.0.1
```

Python：

```text
/disk_n/zzf/conda_envs/vllm_memory/bin/python
```

## 验证命令

语法检查：

```bash
/disk_n/zzf/conda_envs/vllm_memory/bin/python -m py_compile \
  vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py \
  examples/offline_inference/save_prompt_embed_hidden_states_ground_truth.py \
  examples/offline_inference/prompt_embed_extract_hidden_states.py \
  examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py
```

结果：通过，无输出。

GPU 1 单层功能验证：

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_extract_demo \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

结果：通过。

GPU 1 并发压测：

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --output-pth /disk_n/zzf/tmp/qwen3_gt_gpu.output.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_concurrent_demo_variants_b8 \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

结果：通过，总 160 个并发请求，`failed_compares=0`。

GPU 1 MEM 拼接并发压测：

```bash
CUDA_VISIBLE_DEVICES=1 \
/disk_n/zzf/conda_envs/vllm_memory/bin/python \
  examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py \
  --input-pth /disk_n/zzf/tmp/qwen3_gt_gpu.input.pth \
  --work-dir /disk_n/zzf/tmp/qwen3_memory_concurrent_demo_b8 \
  --batch-size 8 \
  --num-rounds 20 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.2
```

结果：通过，总 160 个 MEM 拼接并发请求，prefix 和 memory 两段均
`failed_compares=0`。

## 当前已知限制

- 纯 `prompt_embeds` 请求没有真实 token ids，因此 connector 保存的 `token_ids` 是占位
  0。hidden states 的长度和内容是有效的。
- demo 脚本显式关闭 prefix cache：`enable_prefix_caching=False`。如果后续要在真实业务里打开
  prefix cache，需要确认 cache key 能区分 prompt embeddings，而不只是 token/text。
- 已完成单样例全 28 层验证：vLLM token prompt vs vLLM prompt embeds 完全一致。
- 已完成 batch=4/8 的单层并发压测。
- 已完成 `<MEM(8)> <prompt_emb(16)> <MEM(8)>` 的 batch=4/8 单层和全 28 层并发压测。
- 普通并发脚本的 all-layer strict sequential-reference 对比会失败，原因是单条执行和 batch
  执行的 deep raw hidden states 有数值差；MEM 脚本使用 batch baseline，可以更准确验证
  all-layer 并发路径。
- 在 GPU 1 上早先尝试 `--layer-ids all` 时，因为已有其他进程占用约 18 GiB 显存，vLLM
  warmup 末尾触发 OOM。当前全层测试已在 GPU 0 上通过。

## 当前 git 状态相关文件

```text
 M vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
 M examples/offline_inference/prompt_embed_extract_hidden_states.py
 M examples/offline_inference/prompt_embed_extract_hidden_states_concurrent.py
 M examples/offline_inference/prompt_embed_memory_extract_hidden_states_concurrent.py
 M PROMPT_EMBED_HIDDEN_STATES_CHANGES.md
?? LOCAL_ENVIRONMENT.md
```
