from anthropic import AnthropicVertex


LOCATION="us-east5"


client = AnthropicVertex(region=LOCATION, project_id="ali-icbu-gpu-project")


message = client.messages.create(
 max_tokens=1024,
 messages=[
   {
     "role": "user",
     "content": "Send me a recipe for banana bread.",
   }
 ],
 model="claude-3-7-sonnet@20250219"
)
print(message.model_dump_json(indent=2))