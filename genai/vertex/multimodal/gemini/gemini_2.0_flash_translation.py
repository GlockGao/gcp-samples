from utils.time_utils import timing_decorator
from google import genai
from google.genai import types
import base64


@timing_decorator
def generate():
  client = genai.Client(
      vertexai=True,
      project="ali-icbu-gpu-project",
      location="us-central1",
  )

  text1 = types.Part.from_text(text="""You are an expert Translator. You are tasked to translate documents from en to zh.Please provide an accurate translation of this document and return translation text only: Cloud Translation API uses Google's neural machine translation technology to let you dynamically translate text through the API using Google pre-trained model, custom model, or a translation specialized large language model (LLMs).""")

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        text1
      ]
    )
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0,
    top_k = 1,
    candidate_count = 1,
    max_output_tokens = 8192,
  )

  response = client.models.generate_content(
    model = model,
    contents = contents,
    config = generate_content_config,
  )
  print(response)

generate()