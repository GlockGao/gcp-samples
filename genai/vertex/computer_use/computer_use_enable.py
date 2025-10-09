from google import genai
from google.genai import types
from google.genai.types import Content, Part, FunctionResponse

client = genai.Client()

# Add Computer Use model and tool to the list of tools
generate_content_config = genai.types.GenerateContentConfig(
    tools=[
        types.Tool(
            computer_use=types.ComputerUse(
                environment=types.Environment.ENVIRONMENT_BROWSER,
                )
              ),
            ]
          )

# Example request using the Computer Use model and tool
contents = [
    Content(
        role="user",
        parts=[
            Part(text="Go to google.com and search for 'weather in New York'"),
          ],
        )
      ]

response = client.models.generate_content(
    model="gemini-2.5-computer-use-preview-10-2025",
    contents=contents,
    config=generate_content_config,
  )

print(response)