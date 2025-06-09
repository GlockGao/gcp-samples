#!/bin/bash

# Install pip
sudo apt update
sudo apt install python3-pip

# Install dependencies
pip3 install -r requirements-perf.txt