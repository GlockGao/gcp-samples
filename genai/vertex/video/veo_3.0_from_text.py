from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import time
import os


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


client = genai.Client()

@timing_decorator
def generate_from_text(prompt: str,
                       config: types.GenerateVideosConfig,
                       model: str = "veo-3.0-generate-001"):

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
    if operation.response:
        print(operation.result.generated_videos[0].video.uri)


def main():
    model = "veo-3.0-generate-001"
    prompt = '''The sneaker on the billboard suddenly springs to life, its laces tying themselves. It leaps off the screen, landing on the rooftop below with a soft thud, and sprints out of frame. Audio: The sound of tying laces, a digital whoosh, a soft landing sound.'''
    output_gcs_uri = "gs://bucket-easongy-poc/veo3"
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",                # "16:9" or "9:16"
        output_gcs_uri=output_gcs_uri,
        duration_seconds=8,
        enhance_prompt=True,                # Not supported yet
        fps=24,                             # Not supported yet
        number_of_videos=1,
        generate_audio=True,
        resolution="1080p",                 # "720p" or "1080p"
        person_generation="allow_adult",    # "dont_allow" or "allow_adult"
    )
    
    generate_from_text(prompt=prompt, config=config, model=model)


if __name__ == "__main__":
    main()
