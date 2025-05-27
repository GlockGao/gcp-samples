import os
from google import genai

gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    # print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
    pass
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")

client = genai.Client(api_key=gemini_api_key)

# prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
prompt = "Hello."
response = client.models.generate_content(
    model="gemini-2.5-pro-preview-05-06",
    contents=prompt
)

print(response.text)