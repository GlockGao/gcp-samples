import os
from anthropic import AnthropicVertex
from utils.time_utils import timing_decorator


@timing_decorator
def claude_predict():
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')
    print(location)

    client = AnthropicVertex(region=location, project_id=project_id)

    large_text = "这是一个很长的文档内容..." * 50000  # 模拟大文本
    response = client.beta.messages.create(
        model="claude-sonnet-4@20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": large_text}
        ],
        betas=["context-1m-2025-08-07"]
    )
    print(response)


def main():

    response = claude_predict()

    print(response)


if __name__ == "__main__":
    main()