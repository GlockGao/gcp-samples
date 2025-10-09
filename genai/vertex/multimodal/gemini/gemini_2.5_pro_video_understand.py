from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import base64
import os
import requests


client = genai.Client()


@timing_decorator
def generate(prompt: str,
             config: types.GenerateContentConfig,
             model: str = "gemini-2.5-pro"):

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