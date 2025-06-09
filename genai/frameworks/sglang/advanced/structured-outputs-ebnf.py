import openai
import json
from sglang.utils import print_highlight


host = "localhost"
host = "10.128.0.28"
port = 30000

client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="None")

ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ],
    temperature=0,
    max_tokens=128,
    extra_body={
        "ebnf": ebnf_grammar
    },
)

print_highlight(f"Validated response: {response.choices[0].message.content}")