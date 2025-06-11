from litellm import completion
import os
import sys
from utils.time_utils import timing_decorator


@timing_decorator
def check_api_key():
  """
  Check if the API key is available in the environment.
  """
  api_key = os.environ.get('GEMINI_API_KEY')
  if not api_key:
    print("Please set the GEMINI_API_KEY environment variable with the API key.")
    sys.exit(1)


@timing_decorator
def chat(prompt: str,
         model: str = "gemini-2.5-pro-preview-05-06") -> str:
  """
  Generate a response from the model based on a given prompt.
  """
  response = completion(
    model=model, 
    messages=[{"role": "user", "content": prompt}],
  )
  if response and response.choices:
    answer = response.choices[0].message.content
    return answer
  else:
    return "No response from the model"


check_api_key()

# Example usage
prompt = "Explain union types in TypeScript"
answer = chat(prompt)
print(answer)