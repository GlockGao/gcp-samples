
import openai
import json
from sglang.utils import print_highlight


host = "localhost"
host = "10.128.0.28"
port = 30000

client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="None")


def get_messages():
    return [
        {
            "role": "user",
            "content": "What is 1+3?",
        }
    ]


messages = get_messages()
model_name = "Qwen/Qwen3-8B"

# Non-streaming mode test
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
print(response_non_stream.choices[0].message.tool_calls)