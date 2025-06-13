#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

pip install google-adk==1.2.1
pip install "anthropic[vertex]==0.54.0"

pip show google-adk
pip show anthropic
