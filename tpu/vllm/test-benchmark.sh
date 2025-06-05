#!/bin/bash

# meta-llama/Llama-3.1-8B
export MODEL=meta-llama/Llama-3.1-8B

# Qwen/Qwen2.5-7B
export MODEL=Qwen/Qwen2.5-7B

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