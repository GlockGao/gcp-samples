#!/bin/bash

export model=meta-llama/Llama-3.1-8B-Instruct

python3 -m sglang.bench_serving --backend sglang \
        --host 0.0.0.0 \
        --port 30000 \
        --model ${model} \
        --num-prompt 1000 \
        --dataset-name random \
        --random-input-len 1860 \
        --random-output-len 128 \
        --max-concurrency 1

python3 -m sglang.bench_serving --backend sglang \
        --host 0.0.0.0 \
        --port 30000 \
        --model ${model} \
        --num-prompt 1000 \
        --dataset-name random \
        --random-input-len 1860 \
        --random-output-len 128 \
        --request-rate 16 \
        --max-concurrency 16

python3 -m sglang.bench_serving --backend sglang \
        --host 0.0.0.0 \
        --port 30000 \
        --model ${model} \
        --num-prompt 1000 \
        --dataset-name random \
        --random-input-len 1860 \
        --random-output-len 128 \
        --request-rate 64 \
        --max-concurrency 64