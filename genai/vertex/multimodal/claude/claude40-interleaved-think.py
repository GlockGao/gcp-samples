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

# First request with interleaved thinking enabled
response = client.messages.create(
    model="claude-sonnet-4@20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    # Enable interleaved thinking with beta header
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    messages=[{
        "role": "user",
        "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?"
    }]
)

print("Initial response:")
thinking_blocks = []
tool_use_blocks = []

for block in response.content:
    if block.type == "thinking":
        thinking_blocks.append(block)
        print(f"Thinking: {block.thinking}")
    elif block.type == "tool_use":
        tool_use_blocks.append(block)
        print(f"Tool use: {block.name} with input {block.input}")
    elif block.type == "text":
        print(f"Text: {block.text}")

# First tool result (calculator)
calculator_result = "7500"  # 150 * 50

# Continue with first tool result
response2 = client.messages.create(
    model="claude-sonnet-4@20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
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

# Second tool result (database)
database_result = "5200"  # Example average monthly revenue

# Continue with second tool result
response3 = client.messages.create(
    model="claude-sonnet-4@20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
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
        },
        {
            "role": "assistant",
            "content": thinking_blocks[1:] + tool_use_blocks[1:]
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_blocks[1].id,
                "content": database_result
            }]
        }
    ]
)

print("\nAfter database result:")
# With interleaved thinking, Claude can think about both results
# before formulating the final response
for block in response3.content:
    if block.type == "thinking":
        print(f"Final thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Final response: {block.text}")