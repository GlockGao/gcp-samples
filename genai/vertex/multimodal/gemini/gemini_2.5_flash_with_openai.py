from utils.time_utils import timing_decorator
from google.auth import default
from google.auth.transport.requests import Request
import openai
from typing import List, Dict, Any
import os


PROJECT_ID = "ali-icbu-gpu-project"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

print(f"Using Vertex AI with project: {PROJECT_ID} in location: {LOCATION}")


prompt = """
  How many i's are in the word supercalifragilisticexpialidocious?
"""
messages = [
    {"role": "user", "content": prompt},
]

extra_body = {
    "google": {
      "thinkingConfig": {
        "includeThoughts": True,
        "thinkingBudget": 1024
      },
      "thought_tag_marker": "think"
    }
  }


def setup_openai_client():
    """设置OpenAI客户端连接到Google Vertex AI"""
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "ali-icbu-gpu-project")
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
    
    print(f"使用项目: {PROJECT_ID}, 区域: {LOCATION}")
    
    # 认证
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    
    # 设置API主机
    api_host = "aiplatform.googleapis.com"
    if LOCATION != "global":
        api_host = f"{LOCATION}-aiplatform.googleapis.com"
    
    # 创建客户端
    client = openai.OpenAI(
        base_url=f"https://{api_host}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        api_key=credentials.token,
    )
    
    return client


def parse_response_with_tags(
    response_content: str, tag_name: str
) -> tuple[str | None, str]:
    start_tag = f"<{tag_name}>"
    end_tag = f"{tag_name}>"

    start_index = response_content.find(start_tag)
    end_index = -1
    if start_index != -1:
        end_index = response_content.find(end_tag, start_index + len(start_tag))

    tagged_content = None
    main_content = response_content.strip()

    if start_index != -1 and end_index != -1:
        tagged_content = response_content[
            start_index + len(start_tag) : end_index
        ].strip()
        part_before_tag = response_content[:start_index].strip()
        part_after_tag = response_content[end_index + len(end_tag) :].strip()

        if part_before_tag and part_after_tag:
            main_content = f"{part_before_tag}\n\n{part_after_tag}"
        elif part_before_tag:
            main_content = part_before_tag
        elif part_after_tag:
            main_content = part_after_tag
        else:
            main_content = ""

    elif start_index != -1 and end_index == -1:
        pass

    return tagged_content, main_content


@timing_decorator
def generate_with_openai(client: Any,
                         messages: List[Any],
                         extra_body: Dict[Any, Any],
                         model: str = "google/gemini-2.5-flash-preview-05-20"):
    response = client.chat.completions.create(
        temperature=0,
        model=model,
        messages=messages,
        tool_choice="auto",
        # reasoning_effort="medium",    # gemini-2.5-pro not supported yet
        extra_body=extra_body
    )

    return response


def main():

    client = setup_openai_client()

    model = "google/gemini-2.5-flash-preview-05-20"

    response = generate_with_openai(client=client,
                                    messages=messages,
                                    extra_body=extra_body,
                                    model=model)


    thought, answer = parse_response_with_tags(
        response.choices[0].message.content, "think")

    print("--- Thought ---")
    print(thought)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()