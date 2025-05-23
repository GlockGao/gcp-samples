from utils.time_utils import timing_decorator
from anthropic import AnthropicVertex


LOCATION="us-east5"


client = AnthropicVertex(region=LOCATION, project_id="ali-icbu-gpu-project")


@timing_decorator
def generate(content: str,
             max_tokens: int = 1024,
             model: str = "claude-sonnet-4@20250514"):
  message = client.messages.create(
    max_tokens=max_tokens,
    messages=[
      {
        "role": "user",
        "content": content,
      }
    ],
    model=model
  )

  print(message.content[0].text)
  print(message.model_dump_json(indent=2))

def main():

    content = '''你是什么模型？告诉我具体的版本号。由哪个公司训练的？'''
    generate(content)


if __name__ == "__main__":
    main()