#!/bin/bash
export TOKEN=${HF_TOKEN}
git config --global credential.helper store
huggingface-cli login --token $TOKEN

cd ~/vllm

# meta-llama/Llama-3.1-8B
vllm serve "meta-llama/Llama-3.1-8B" --download_dir /tmp --num-scheduler-steps 4 --swap-space 16 --disable-log-requests --tensor_parallel_size=4 --max-model-len=2048 &> serve.log &

# Qwen/Qwen2.5-7B
vllm serve "Qwen/Qwen2.5-7B" --download_dir /tmp --num-scheduler-steps 4 --swap-space 16 --disable-log-requests --tensor_parallel_size=4 --max-model-len=2048 &> serve.log &

# meta-llama/Llama-3.1-8B
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

# Qwen/Qwen2.5-7B
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'