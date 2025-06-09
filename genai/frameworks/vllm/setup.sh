#!/bin/bash

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install and activate venv
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vllm
uv pip install vllm --torch-backend=auto

# Install dependencies
sudo apt update
sudo apt install build-essential
sudo apt install python3.12-dev

# Download repo
git clone https://github.com/vllm-project/vllm.git && cd vllm
