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

pip install pandas
pip install datasets

# 2. 性能测试
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $MODEL  \
  --dataset-name random \
  --random-input-len 1820 \
  --random-output-len 128 \
  --random-prefix-len 0