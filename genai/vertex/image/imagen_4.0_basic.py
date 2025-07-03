from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from typing import Union
import os

import vertexai
from typing import Any, Literal, Optional, cast
from vertexai.preview.vision_models import ImageGenerationModel
from utils.time_utils import timing_decorator


PROJECT_ID = os.getenv('PROJECT_ID')

if PROJECT_ID:
    print(f"获取到的 PROJECT_ID: {PROJECT_ID}")
else:
    print("环境变量 'PROJECT_ID' 未设置。")

vertexai.init(project=PROJECT_ID, location="us-central1")

model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-06-06")


@timing_decorator
def generate(prompt: str,
             number_of_images: int = 1,
             seed: Optional[int] = 100,
             negative_prompt: Optional[str] = None,
             aspect_ratio: Optional[str] = "1:1",
             compression_quality: Optional[int] = 75,
             language: Optional[str] = None,
             output_gcs_uri: Optional[str] = None,
             add_watermark: Optional[bool] = False,
             safety_filter_level: Optional[str] = "block_some",
             person_generation: Optional[str] = "allow_adult"):

    images = model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images,
        # Optional parameters
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        # compression_quality=compression_quality,
        language=language,
        output_gcs_uri=output_gcs_uri,

        # You can't use a seed value and watermark at the same time.
        add_watermark=add_watermark,
        seed=seed,
        safety_filter_level=safety_filter_level,
        person_generation=person_generation,
    )

    images[0].save(location='imagen4-image-basic.png', include_generation_parameters=False)


def main():

    prompt = ('Hi, can you create a 3d rendered image of a pig '
                'with wings and a top hat flying over a happy '
                'futuristic scifi city with lots of greenery?')
    generate(prompt=prompt) 


if __name__ == "__main__":
    main()