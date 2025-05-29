from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from typing import Union
import os
from utils.time_utils import timing_decorator
import PIL.Image


gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")

client = genai.Client(api_key=gemini_api_key)


@timing_decorator
def edit(contents: Union[types.ContentListUnion, types.ContentListUnionDict],
         model: str = "imagen-3.0-generate-002"):

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save('gemini-native-image-edit.png')
            image.show()


def main():

    text_input = ('Hi, This is a picture of me.'
            'Can you add a llama next to me?',)
    image = PIL.Image.open('gemini-native-image.png')
    contents = [text_input, image]
    edit(contents=contents, model="imagen-3.0-generate-002") 


if __name__ == "__main__":
    main()