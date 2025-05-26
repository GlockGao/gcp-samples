#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh


# MCP server
uv init weather
cd weather
uv venv
source .venv/bin/activate

# Requirements installation
uv add "mcp[cli]" httpx

# inspector
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22