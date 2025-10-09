from google import genai
from google.genai import types
from google.genai.types import Content, Part

client = genai.Client()

# Specify predefined functions to exclude (optional)
excluded_functions = ["drag_and_drop"]

# Configuration for the Computer Use model and tool with browser environment
generate_content_config = genai.types.GenerateContentConfig(
    tools=[
        # 1. Computer Use model and tool with browser environment
        types.Tool(
            computer_use=types.ComputerUse(
                environment=types.Environment.ENVIRONMENT_BROWSER,
                # Optional: Exclude specific predefined functions
                excluded_predefined_functions=excluded_functions
                )
              ),
        # 2. Optional: Custom user-defined functions (need to defined above)
        # types.Tool(
           # function_declarations=custom_functions
           #   )
    ],
)

# Create the content with user message
contents: list[Content] = [
    Content(
        role="user",
        parts=[
            Part(text="Search for highly rated smart fridges with touchscreen, 2 doors, around 25 cu ft, priced below 4000 dollars on Google Shopping. Create a bulleted list of the 3 cheapest options in the format of name, description, price in an easy-to-read layout."),
            # Optional: include a screenshot of the initial state
            # Part.from_bytes(
                 # data=screenshot_image_bytes,
                 # mime_type='image/png',
            # ),
        ],
    )
]

# Generate content with the configured settings
response = client.models.generate_content(
    model='gemini-2.5-computer-use-preview-10-2025',
    contents=contents,
    config=generate_content_config,
)

# Print the response output
print(response.text)