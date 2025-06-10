import openai

from google.auth import default
from google.auth.transport.requests import Request


credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(Request())
print(f"Credentials token : {credentials.token}")

REGION='us-central-1'
use_dedicated_endpoint = True

BASE_URL = 'https://8807964449253097472.us-central1-979398597045.prediction.vertexai.goog/v1beta1/projects/ali-icbu-gpu-project/locations/us-central1/endpoints/8807964449253097472'

user_message = "How is your day going?"
max_tokens = 5000 
temperature = 1.0
stream = False


client = openai.OpenAI(base_url=BASE_URL, api_key=credentials.token)

model_response = client.chat.completions.create(
    model="",
    messages=[{"role": "user", "content": user_message}],
    temperature=temperature,
    max_tokens=max_tokens,
    stream=stream,
)

print(model_response)
