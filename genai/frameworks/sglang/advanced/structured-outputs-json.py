import openai
import json
from sglang.utils import print_highlight


host = "localhost"
host = "10.128.0.28"
port = 30000

client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="None")

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            "schema": json.loads(json_schema),
        },
    },
)

print_highlight(f"Validated response: {response.choices[0].message.content}")