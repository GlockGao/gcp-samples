from google import genai
from google.genai import types  # type: ignore
from PIL import Image           # type: ignore
from io import BytesIO
import base64
from typing import Union
import os
from utils.time_utils import timing_decorator


gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")

client = genai.Client(api_key=gemini_api_key)


@timing_decorator
def generate(prompt: str,
             model: str = "imagen-3.0-generate-002"):

    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1"
        )
    )

    for generated_image in response.generated_images:
        image = Image.open(BytesIO(generated_image.image.image_bytes))
        image.save('imagen-guide-image.png')


def main():

    prompt = '''A park in the spring next to a lake, the sun sets across the lake, golden hour, red wildflowers.'''
    generate(prompt=prompt, model="imagen-3.0-generate-002") 


if __name__ == "__main__":
    main()