# vLLM prompt-embeds hidden-states docker demo

This package is a small reproduction bundle for testing prompt embeddings with
`ExampleHiddenStatesConnector`.

## Files

- `setup.sh`: installs this bundle into the container and replaces the vLLM
  connector file.
- `patches/example_hidden_states_connector.py`: patched connector.
- `docker_demo.sh`: runs the single-sample and batch prompt-embeds demos.
- `examples/offline_inference/`: demo scripts used by `docker_demo.sh`.

## Container target paths

`setup.sh` copies the patched connector to:

```bash
/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
```

It installs the runnable demo files under:

```bash
/home/dpsk_a2a/vllm-memory-demo/docker_demo.sh
/home/dpsk_a2a/vllm-memory-demo/examples/offline_inference/
```

## Usage

Run this inside the target container:

```bash
cd /path/to/this/package
bash setup.sh
```

Then configure the model/cache environment as needed and run:

```bash
cd /home/dpsk_a2a/vllm-memory-demo
bash docker_demo.sh
```

Useful overrides:

```bash
MODEL=Qwen/Qwen3-0.6B GPU=0 bash docker_demo.sh
```

By default, `docker_demo.sh` clears host proxy variables and uses
`HF_ENDPOINT=https://hf-mirror.com`. Override these with environment variables
if the container needs different network settings.
