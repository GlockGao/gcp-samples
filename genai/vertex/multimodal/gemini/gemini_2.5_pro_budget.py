from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import base64
import os


client = genai.Client()


@timing_decorator
def generate_without_budget(prompt: str,
                            model: str = "gemini-2.5-pro"):

    response = client.models.generate_content(
      model=model,
      contents=prompt
    )

    print(response.text)


@timing_decorator
def generate_with_budget(prompt: str,
                         config: types.GenerateContentConfig,
                         model: str = "gemini-2.5-pro"):

    response = client.models.generate_content(
      model=model,
      contents=prompt,
      config=config
    )

    print(response.text)
    

def main():

    prompt = '''Hello.'''
    generate_without_budget(prompt=prompt)

    thinking_config = types.ThinkingConfig(thinking_budget=2048)
    config = types.GenerateContentConfig(thinking_config=thinking_config)
    generate_with_budget(prompt=prompt, config=config)

    # thinking_config = types.ThinkingConfig(thinking_budget=0)
    # config = types.GenerateContentConfig(thinking_config=thinking_config)
    # generate_with_budget(prompt=prompt, config=config)


if __name__ == "__main__":
    main()