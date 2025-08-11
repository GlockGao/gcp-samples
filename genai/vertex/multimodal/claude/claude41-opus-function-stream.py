import os
from utils.time_utils import timing_decorator
from anthropic import AnthropicVertex


location = os.getenv('GOOGLE_CLOUD_LOCATION')
if location:
    print(f"获取到的 GOOGLE_CLOUD_LOCATION: {location}")
else:
    print("环境变量 'GOOGLE_CLOUD_LOCATION' 未设置。")

project = os.getenv('GOOGLE_CLOUD_PROJECT')
if project:
    print(f"获取到的 GOOGLE_CLOUD_PROJECT: {project}")
else:
    print("环境变量 'GOOGLE_CLOUD_PROJECT' 未设置。")


client = AnthropicVertex(region=location, project_id=project)


@timing_decorator
def generate(content: str,
             max_tokens: int = 1024,
             model: str = "claude-opus-4-1@20250805"):
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

#   print(message.content[0].text)
  print(message.model_dump_json(indent=2))

def main():

    content = '''Send me a recipe for banana bread.'''
    generate(content)


if __name__ == "__main__":
    main()