import os
from google import genai


client = genai.Client()

# prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
prompt = "Hello."
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt
)

print(response.text)