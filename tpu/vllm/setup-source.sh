#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n vllm python=3.10 -y

conda activate vllm

git clone https://github.com/vllm-project/vllm.git && cd vllm

pip uninstall torch torch-xla -y

pip install -r requirements/tpu.txt
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev

VLLM_TARGET_DEVICE="tpu" python -m pip install -e .