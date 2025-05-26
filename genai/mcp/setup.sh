#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh


# MCP Client
uv init mcp-client
cd mcp-client
uv venv
source .venv/bin/activate

# Requirements installation
uv add mcp