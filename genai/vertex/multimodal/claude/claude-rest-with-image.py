import os
import requests
import google.auth
import google.auth.transport.requests
import base64
from typing import List, Dict, Any, Optional
from utils.time_utils import timing_decorator


def encode_image_to_base64(image_path: str) -> str:
    """
    将图片文件编码为 base64 字符串
    Encode an image file to base64 string
    
    Args:
        image_path (str): 图片文件路径 (Path to the image file)
    
    Returns:
        str: base64 编码的图片字符串 (Base64 encoded image string)
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        print(f"Error: Image file {image_path} not found")
        raise
    except Exception as e:
        print(f"错误：编码图片时出现问题 {e}")
        print(f"Error: Problem encoding image {e}")
        raise


def get_image_media_type(image_path: str) -> str:
    """
    根据文件扩展名获取图片的媒体类型
    Get image media type based on file extension
    
    Args:
        image_path (str): 图片文件路径 (Path to the image file)
    
    Returns:
        str: 媒体类型 (Media type)
    """
    extension = os.path.splitext(image_path)[1].lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(extension, 'image/jpeg')


@timing_decorator
def claude_predict_with_image(image_path: Optional[str] = None, text_prompt: str = "请描述这张图片"):
    """
    发送包含图片的预测请求到 Google Cloud Vertex AI 上的 Claude 模型
    Sends a prediction request with image to a Claude model on Google Cloud Vertex AI.
    
    Args:
        image_path (Optional[str]): 图片文件路径 (Path to the image file)
        text_prompt (str): 文本提示 (Text prompt)
    """

    # --- 1. 设置环境变量 (Set Environment Variables) ---
    # 您可以像这样直接在代码中设置，或者在您的运行环境中预先设置好
    # You can set them directly in the code like this, or pre-set them in your environment.
    os.environ['MODEL_ID'] = 'claude-opus-4@20250514'
    # os.environ['MODEL_ID'] = 'claude-3-7-sonnet@20250219'
    os.environ['LOCATION'] = 'us-east5'
    os.environ['PROJECT_ID'] = 'ali-icbu-gpu-project' # 请替换为您的项目ID (Please replace with your Project ID)

    # --- 2. 从环境中获取变量 (Get Variables from Environment) ---
    model_id = os.getenv('MODEL_ID')
    location = os.getenv('LOCATION')
    project_id = os.getenv('PROJECT_ID')

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

    # 构建请求头
    # Construct the request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # 构建消息内容 (Build message content)
    content = []
    
    # 添加文本内容 (Add text content)
    content.append({
        "type": "text",
        "text": text_prompt
    })
    
    # 如果提供了图片路径，添加图片内容 (If image path is provided, add image content)
    if image_path:
        try:
            # 编码图片为 base64 (Encode image to base64)
            base64_image = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image
                }
            })
            print(f"已添加图片: {image_path} (媒体类型: {media_type})")
            print(f"Added image: {image_path} (media type: {media_type})")
        except Exception as e:
            print(f"处理图片时出错: {e}")
            print(f"Error processing image: {e}")
            return

    # 构建请求体 (JSON payload)
    # Construct the request body (JSON payload)
    data = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": [{
            "role": "user",
            "content": content
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
        response_data = response.json()
        print(response_data)
        return response_data

    except requests.exceptions.RequestException as e:
        print(f"请求失败 (Request failed): {e}")
        if e.response is not None:
            print("响应内容 (Response content):")
            print(e.response.text)
        return None


@timing_decorator
def claude_predict():
    """
    发送不包含图片的预测请求到 Google Cloud Vertex AI 上的 Claude 模型
    Sends a prediction request without image to a Claude model on Google Cloud Vertex AI.
    """
    return claude_predict_with_image(image_path=None, text_prompt="Hey Claude!")


def main():
    """
    主函数 - 演示如何使用图片功能
    Main function - demonstrates how to use image functionality
    """
    
    # 示例: 使用图片 (Example: With image)
    # 请将下面的路径替换为您的图片文件路径 (Please replace the path below with your image file path)
    image_path = "claude.png"  # 请替换为实际的图片路径
    
    # 检查图片文件是否存在 (Check if image file exists)
    if os.path.exists(image_path):
        print(f"\n=== 示例: 使用图片 (Example : With image) ===")
        response = claude_predict_with_image(
            image_path=image_path,
            text_prompt="What is in this image?"
        )
        print(response)
    else:
        print(f"\n图片文件不存在: {image_path}")
        print(f"Image file does not exist: {image_path}")
        print("请修改 main() 函数中的 image_path 变量为实际的图片路径")
        print("Please modify the image_path variable in main() function to an actual image path")


if __name__ == "__main__":
    main()
