from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import time
import os


gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")


client = genai.Client(api_key=gemini_api_key)


@timing_decorator
def generate_from_text(prompt: str,
                       config: types.GenerateVideosConfig,
                       model: str = "veo-2.0-generate-001"):

    # 1. Generate video
    operation = client.models.generate_videos(
      model=model,
      prompt=prompt,
      config=config
    )

    # 2. Wait video generation
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    # 3. Save the video
    print(operation.response)
    print(operation.response.generated_videos)
    for n, generated_video in enumerate(operation.response.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(f"video{n}.mp4")  # save the video


def main():
    model = "veo-2.0-generate-001"
    prompt = '''Panning wide shot of a calico kitten sleeping in the sunshine.'''
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",  # "16:9" or "9:16"
        duration_seconds=5,
        # enhance_prompt=True,          # Not supported yet
        # fps=24, # Not supported yet
        number_of_videos=1,
        # resolution="1920x1080",          # "1280x720" or "1920x1080", Not supported yet
        person_generation="dont_allow",  # "dont_allow" or "allow_adult"
    )
    
    generate_from_text(prompt=prompt, config=config, model=model)


if __name__ == "__main__":
    main()
