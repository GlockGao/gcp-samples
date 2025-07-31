from utils.time_utils import timing_decorator
from anthropic import AnthropicVertex


LOCATION="us-east5"


client = AnthropicVertex(region=LOCATION, project_id="ali-icbu-gpu-project")

# Same tool definitions as before
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

database_tool = {
    "name": "database_query",
    "description": "Query product database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            }
        },
        "required": ["query"]
    }
}

@timing_decorator
def generate(content: str,
             max_tokens: int = 1024,
             tools: list = None,
             model: str = "claude-sonnet-4@20250514"):
  message = client.messages.create(
    max_tokens=max_tokens,
    messages=[
      {
        "role": "user",
        "content": content,
      }
    ],
    model=model,
    tools=tools
  )

#   print(message.content[0].text)
#   print(message.model_dump_json(indent=2))

  return message

def main():

    content = '''What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?'''
    response = generate(content,
                        max_tokens=2048,
                        tools=[calculator_tool, database_tool])
    print(response)

    thinking_blocks = []
    tool_use_blocks = []
    print('#' * 100)
    print('Intial response')
    for block in response.content:
        if block.type == "thinking":
            thinking_blocks.append(block)
            print(f"Thinking: {block.thinking}")
        elif block.type == "tool_use":
            tool_use_blocks.append(block)
            print(f"Tool use: {block.name} with input {block.input}")
        elif block.type == "text":
            print(f"Text: {block.text}")

    # Continue with first tool result
    calculator_result = "7500"  # 150 * 50
    response2 = client.messages.create(
        model="claude-sonnet-4@20250514",
        max_tokens=16000,
        tools=[calculator_tool, database_tool],
        messages=[
            {
                "role": "user",
                "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
            },
            {
                "role": "assistant",
                "content": [thinking_blocks[0], tool_use_blocks[0]]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_blocks[0].id,
                    "content": calculator_result
                }]
            }
        ]
    )

print("\nAfter calculator result:")
# With interleaved thinking, Claude can think about the calculator result
# before deciding to query the database
for block in response2.content:
    if block.type == "thinking":
        thinking_blocks.append(block)
        print(f"Interleaved thinking: {block.thinking}")
    elif block.type == "tool_use":
        tool_use_blocks.append(block)
        print(f"Tool use: {block.name} with input {block.input}")


if __name__ == "__main__":
    main()