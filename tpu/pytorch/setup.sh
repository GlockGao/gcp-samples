#!/bin/bash
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html