from utils.time_utils import timing_decorator
from utils.think_parse_utils import parse_response_with_tags
from google.auth import default
from google.auth.transport.requests import Request
import openai
from typing import List, Dict, Any
import os


messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello, how are you?",
        }
      ],
    },
  ]



def setup_openai_client():
    """设置OpenAI客户端连接到Google Vertex AI"""
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-easongy-poc")
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
    
    print(f"使用项目: {PROJECT_ID}, 区域: {LOCATION}")
    
    # 认证
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    # print(f"Credentials token : {credentials.token}")
    
    # 设置API主机
    api_host = "aiplatform.googleapis.com"
    if LOCATION != "global":
        api_host = f"{LOCATION}-aiplatform.googleapis.com"
    
    # 创建客户端
    client = openai.OpenAI(
        base_url=f"https://{api_host}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        api_key=credentials.token,
    )
    
    return client


@timing_decorator
def generate_with_openai(client: Any,
                         messages: List[Any],
                         model: str = "google/gemini-3-pro-preview"):
    response = client.chat.completions.create(
        max_tokens=2048,
        model=model,
        messages=messages,
        stream=False,
    )

    return response


def main():

    client = setup_openai_client()

    model = "google/gemini-3-pro-preview"

    response = generate_with_openai(client=client,
                                    messages=messages,
                                    model=model)

    print(response)


if __name__ == "__main__":
    main()