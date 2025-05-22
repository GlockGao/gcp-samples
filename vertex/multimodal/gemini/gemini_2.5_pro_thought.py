from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import base64
import os


gemini_api_key = os.getenv('GEMINI_API_KEY')

# if gemini_api_key:
#     print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
# else:
#     print("环境变量 'GEMINI_API_KEY' 未设置。")


client = genai.Client(api_key=gemini_api_key)


@timing_decorator
def generate(prompt: str,
             config: types.GenerateContentConfig,
             model: str = "gemini-2.5-pro-preview-05-06"):

    response = client.models.generate_content(
      model=model,
      contents=prompt,
      config=config
    )

    print(response)
    print(response.text)


def main():

    prompt = '''Explain the concept of Occam's Razor and provide a simple, everyday example.'''

    thinking_config = types.ThinkingConfig(include_thoughts=True)
    config = types.GenerateContentConfig(thinking_config=thinking_config)
    generate(prompt=prompt, config=config)

    thinking_config = types.ThinkingConfig(include_thoughts=False)
    config = types.GenerateContentConfig(thinking_config=thinking_config)
    generate(prompt=prompt, config=config)


if __name__ == "__main__":
    main()