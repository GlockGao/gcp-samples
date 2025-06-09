#!/bin/bash

# Install docker
sudo apt update
sudo apt install docker.io

# Pull vllm image
sudo docker pull lmsysorg/sglang:latest
# By the end of date : 2025-06-09
sudo docker pull lmsysorg/sglang:v0.4.6.post5-cu124

# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Official nvidia image test
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi