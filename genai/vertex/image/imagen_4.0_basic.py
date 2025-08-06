import os
from typing import Optional
from google import genai
from google.genai.types import GenerateImagesConfig
from utils.time_utils import timing_decorator


project = os.getenv('GOOGLE_CLOUD_PROJECT')
if project:
    print(f"获取到的 GOOGLE_CLOUD_PROJECT: {project}")
else:
    print("环境变量 'GOOGLE_CLOUD_PROJECT' 未设置。")

location = os.getenv('GOOGLE_CLOUD_LOCATION')
if location:
    print(f"获取到的 GOOGLE_CLOUD_LOCATION: {location}")
else:
    print("环境变量 'GOOGLE_CLOUD_LOCATION' 未设置。")

client = genai.Client(project=project, location=location)


@timing_decorator
def generate(
    model: str,
    prompt: str,
    config: GenerateImagesConfig,
    file: str = 'imagen4-image-basic.png'):

    image = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=config,
    )

    image.generated_images[0].image.save(file)


def main():

    # 1. 提示词和配置文件
    prompt = "a man wearing all white clothing sitting on the beach, close up, golden hour lighting"
    config = GenerateImagesConfig(
        aspect_ratio="16:9",        # Supported values are “1:1”, “3:4”, “4:3”, “9:16”, and “16:9”
        image_size="2K",            # Supported sizes are 1K and 2K
        number_of_images=1,
    )

    # 2. 调用不同的模型
    # 2.1 standard imagen 4.0
    model = "imagen-4.0-generate-preview-06-06"
    file = "imagen-4.0-generate-preview-06-06.png"
    generate(model=model, prompt=prompt, config=config, file=file)

    # 2.2 ultra imagen 4.0
    model = "imagen-4.0-ultra-generate-preview-06-06"
    file = "imagen-4.0-ultra-generate-preview-06-06.png"
    generate(model=model, prompt=prompt, config=config, file=file)

    # 2.3 fast imagen 4.0
    model = "imagen-4.0-fast-generate-preview-06-06"
    file = "imagen-4.0-fast-generate-preview-06-06.png"
    generate(model=model, prompt=prompt, config=config, file=file)


if __name__ == "__main__":
    main()