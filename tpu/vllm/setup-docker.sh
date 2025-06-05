#!/bin/bash
git clone https://github.com/vllm-project/vllm.git && cd vllm

sudo docker build -f docker/Dockerfile.tpu -t vllm-tpu .
sudo docker images vllm-tpu

sudo docker run --privileged --net host --shm-size=16G -it vllm-tpu