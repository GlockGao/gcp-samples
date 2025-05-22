import os
from google import genai

gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")

client = genai.Client(api_key=gemini_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)