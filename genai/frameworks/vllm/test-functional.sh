#!/bin/bash

# meta-llama/Llama-3.1-8B-Instruct - L4-GPUx2
sudo docker run --runtime nvidia --gpus all --shm-size 1g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=${TOKEN}" \
    vllm/vllm-openai \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --download_dir /tmp \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size 2 \
    --max-model-len 2048


# meta-llama/Llama-3.1-8B-Instruct
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'