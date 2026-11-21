# GLM RSI Feedback Lab

GLM RSI Feedback Lab explores **recursive self-improvement (RSI)** with GLM-5.3-Flash. In each operator-triggered round, the live model proposes corrections for failed user requests. An independent verifier checks the corrections before they enter the training set. If the resulting candidate passes evaluation, it becomes the live model and proposes corrections for the next round.

Two groups of four NVIDIA B300 GPUs share one server. While one group serves requests, the other trains and tests the candidate. A local router keeps the chat API at the same address on port 8000 as the groups switch roles.

| Phase | Group A · 4 B300s | Group B · 4 B300s |
|---|---|---|
| First round | Serves requests and proposes corrections | Trains, exports, and tests the candidate |
| Evaluation | Answers held-out tasks | Answers the same tasks for comparison |
| Candidate passes | Finishes in-flight responses, then takes the next training round | Takes over live requests and feedback corrections |
| Candidate fails | Continues serving | Ends the round without promotion |

## Why BF16?

NVIDIA lists inference profiles for the full 320-billion-parameter model at three precisions. NVFP4 uses two B300s per instance and downloads about 205 GB of weights. FP8 and BF16 each use four B300s, with downloads of about 330 GB and 645 GB respectively. This project uses BF16 so it can merge the trained adapter into the base weights and compare successive generations at the same precision. Four-GPU QLoRA training and serving custom checkpoints still require validation on B300 hardware. [NVIDIA support matrix](https://docs.nvidia.com/nim/large-language-models/2.1.4/reference/support-matrix.html#glm-5-3-flash) · [NVIDIA checkpoint guide](https://docs.nvidia.com/nim/large-language-models/2.1.4/get-started/advanced/get-started-glm-5.3-flash.html) · [Z.ai model overview](https://docs.z.ai/guides/vlm/glm-5.3-flash)

## Run on the server

Start from the repository root on Linux with Python 3.9 or later, Docker, the NVIDIA Container Toolkit, and a local BF16 checkpoint. Export and memory testing also require a CUDA build of llama.cpp with GLM5-Next support and the converter’s Python dependencies. Allow several terabytes of disk space for temporary BF16 copies, depending on the filesystem. [BF16 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) · [llama.cpp revision used for the memory test](https://github.com/ggml-org/llama.cpp/tree/e50c254a22c590c9d2a60db86690d48d42497e2b)

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[checkpoint]'
docker pull nvcr.io/nim/nvidia/sglang-model-free-nim:2.1.4-variant-glm-5.3-flash
docker pull axolotlai/axolotl:main-latest
glm-rotate init --state runs/rotation --model /models/GLM-5.3-Flash-BF16
glm-route --state runs/rotation --port 8000
```

Keep the router running in its own process. An existing guardrail gateway can send requests to `http://127.0.0.1:8000/v1` using the model alias `glm-5.3-flash`. In a second shell, activate the same environment and run one training step with the sample data.

```sh
glm-feedback --cases examples/cases.jsonl --oracles examples/oracles.jsonl --out runs/feedback-01
glm-rotate round --state runs/rotation --data runs/feedback-01/train.jsonl --checks examples/checks.jsonl --llama-dir /opt/llama.cpp --steps 1
```

The synthetic examples illustrate the data format. If the live model already solves both sample evaluation tasks, the candidate cannot demonstrate a gain and the live group keeps serving. For a meaningful promotion decision, provide independent held-out failure cases and controls.

The full live model runs on four B300s. Each candidate’s quantized text export must also fit on one B300, including cache and reserved memory. The training guide covers the role switch, recovery, and model growth limit. [Training guide (German)](TRAINING.md)

## Test status

Local tests use small real weight tensors, local HTTP services, and simulated Docker and GPU interfaces. Run them with `python -m unittest discover -s tests`. An end-to-end GLM run on eight physical B300s is still pending.
