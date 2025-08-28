import os
import requests
import google.auth
import google.auth.transport.requests
from typing import List, Dict, Any
from utils.time_utils import timing_decorator


@timing_decorator
def claude_predict():
    """
    Sends a prediction request to a Claude model on Google Cloud Vertex AI.
    """

    # --- 1. 设置环境变量 (Set Environment Variables) ---
    # 您可以像这样直接在代码中设置，或者在您的运行环境中预先设置好
    # You can set them directly in the code like this, or pre-set them in your environment.
    os.environ['MODEL_ID'] = 'claude-sonnet-4@20250514'
    os.environ['LOCATION'] = 'us-east5'
    # os.environ['LOCATION'] = 'europe-west1'
    # os.environ['LOCATION'] = 'global'   # 2025-08-28, global endpoint doesn't support 1M input tokens

    # --- 2. 从环境中获取变量 (Get Variables from Environment) ---
    model_id = os.getenv('MODEL_ID')
    location = os.getenv('LOCATION')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

    if not all([model_id, location, project_id]):
        print("错误：请确保设置了 MODEL_ID, LOCATION, 和 PROJECT_ID 环境变量。")
        print("Error: Please make sure the MODEL_ID, LOCATION, and PROJECT_ID environment variables are set.")
        return

    # --- 3. 获取认证令牌 (Get Authentication Token) ---
    try:
        # 使用 Google Application Default Credentials (ADC) 获取凭证
        # Get credentials using Google Application Default Credentials (ADC)
        credentials, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
    except google.auth.exceptions.DefaultCredentialsError:
        print("错误：无法获取 Google Cloud 认证凭据。")
        print("请确保您已经通过 'gcloud auth application-default login' 命令进行认证。")
        print("Error: Could not get Google Cloud authentication credentials.")
        print("Please make sure you have authenticated via the 'gcloud auth application-default login' command.")
        return

    # --- 4. 构建请求 (Construct the Request) ---
    # 构建请求 URL
    # Construct the request URL
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict"

    # url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict"

    # 构建请求头
    # Construct the request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "anthropic-beta": "context-1m-2025-08-07",
    }

    # 构建请求体 (JSON payload)
    # Construct the request body (JSON payload)
    large_text = "这是一个很长的文档内容..." * 50000  # 模拟大文本
    data = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": [{
            "role": "user",
            "content": large_text
        }],
        "max_tokens": 2048,
        "stream": False,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024
        }
    }

    # --- 5. 发送请求并处理响应 (Send Request and Handle Response) ---
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 如果请求失败 (状态码 >= 400), 则会抛出异常 (Raises an exception for bad status codes)

        # 打印成功的响应内容
        # Print the successful response content
        print("请求成功! (Request successful!)")
        print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"请求失败 (Request failed): {e}")
        if e.response is not None:
            print("响应内容 (Response content):")
            print(e.response.text)


def main():

    response = claude_predict()

    print(response)


if __name__ == "__main__":
    main()