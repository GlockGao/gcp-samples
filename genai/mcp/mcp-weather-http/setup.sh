#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh


# MCP server
uv init mcp-weather-http
cd mcp-weather-http
uv venv
source .venv/bin/activate

# Requirements installation
uv add "mcp[cli]" httpx

# inspector
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22

# Source directory
rm -f main.py
mkdir -p src/mcp_weather_http
cd src/mcp_weather_http/
touch __init__.py
touch __main__.py
touch server.py