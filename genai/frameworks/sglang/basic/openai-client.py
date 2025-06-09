import openai
from sglang.utils import print_highlight


host = "localhost"
host = "10.128.0.28"
port = 30000

client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print_highlight(response)