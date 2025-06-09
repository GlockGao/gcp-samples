import requests
from sglang.utils import print_highlight


port = 30000

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())