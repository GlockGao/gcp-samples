#!/bin/bash

export model=meta-llama/Llama-3.1-8B-Instruct
export host=0.0.0.0
export port=30000
export tp=2

sudo docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${TOKEN}" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path ${model} --host ${host} --port ${port} --tp ${tp}