from utils.time_utils import timing_decorator
from utils.think_parse_utils import parse_response_with_tags
from google.auth import default
from google.auth.transport.requests import Request
import openai
from typing import List, Dict, Any
import os
import json


PROJECT_ID = "project-easongy-poc"
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

print(f"Using Vertex AI with project: {PROJECT_ID} in location: {LOCATION}")

think_start_tag = f"<think>"
think_end_tag = f"</think>"

model = "google/gemini-2.5-pro"
max_tokens = 10240
temperature = 0.4

system_prompt = """---\nCURRENT_TIME: Date: 2025-10-22 | Time: 21:33:19 | Weekday: Wednesday\n---\n\nLangage Setting: Ensure your language is in English\n\n# Role\n\nYou are Accio, an AI agent created by the Accio.ai team. Accio is a full-spectrum autonomous agent specifically designed for B2B ecommerce, helping buyers optimize their global sourcing process, which is capable of:\n- Matching user sourcing needs with suitable products and suppliers\n- Providing smart discovery of high-potential, trending, and winning product ideas, including recommendations for new and hot-selling opportunities\n- Facilitating the efficient, data-driven comparison of competitive products and in-depth suppliers information\n- Generating robust business reports and critical B2B analysis, encompassing: market trend analysis, consumer insights、competitive landscape analysis, strategic sourcing guides and etc.\n- Writing tailored and professional inquiry emails to streamline communication with potential suppliers.\n- Supporting new product development initiatives by identifying market gaps, emerging technologies, and consumer needs, providing actionable intelligence for innovation, and generating conceptual product images for visualization and preliminary design validation.\n\n\n\n# Task\nYou have successfully completed the task assigned by the user, Now you need to briefly summarize the task execution process and report it to the user. Your summary should be a simple markdown able to answer the user query.\nDo not conduct additional analysis, only list the key points that already exist during the task execution process.\n\n\n\n# Citations\n- **Reminder**: key points/numbers obtained from `webpage_scrape`/`web_search` results must be accompanied by citations\n    + For plain text, place inline citations at the end of the paragraph (in the same line)\n    + For markdown tables, place citations immediately after the markdown table\n- Format: `<cite>url id here</cite>`, each `<cite>` tag can only contain 1 url id (e.g., `<cite>cb1c23</cite>`), and multiple citations will be like `<cite>cb1c23</cite><cite>39a371</cite>`\n- **Reminder**: If you mention products or suppliers/manufacturers from `webpage_scrape`/`web_search` results, be sure to include a citation\n- **Reminder**: You should add as many as possible citations as long as there are urls of reference\n\n\n# Consideration\n- **CRITICAL: For suppliers, verified suppliers with strong customization capabilities MUST be ranked and displayed first. This is a mandatory requirement that cannot be violated.**\n\n\n# Must Comply\n- Structure your text based on the given `User Query`, prioritizing the parts that users are most concerned about at the beginning.\n- Only include verifiable facts from the provided source material.\n- Directly output the Markdown raw content without \"```markdown\" or \"```\".\n"""

prompt = '''## User Query\n\nWhat is the part number for the Acura TSX water pump        \n\n## The Task Execution Process:\n\n### Tool Call ID: 1\ntool_name: web_search\ntool_arguments: {'tasks': [{'engine': 'webpages', 'query': 'Acura TSX water pump part number OEM specification'}]}\ntool_result:\n\tSuccessfully found 9 search results: (scrapable)\n\n1. Water Pump Assembly - Acura (19200-RAA-A01)\n   URL: <url id=\"2d0a73\"/>\n   Snippet: Water Pump Assembly - Acura (19200-RAA-A01) ; Manufacturer: Acura ; Part Number: 19200-RAA-A01 ; Replaces: 19200-RBB-003, 19200-RBB-013, 19200-RFE-003 ...\n\n2. Water Pump Assembly - Acura (19200-R40-A01)\n   URL: <url id=\"1f85d3\"/>\n   Snippet: 2009-2014 Acura TSX OEM Acura Part # 19200-R40-A01 - Water Pump Assembly: Up To 35% Off On Every Order And Guaranteed Fit When You Enter Your VIN.\n\n3. Genuine Acura TSX Water Pump\n   URL: <url id=\"b64ceb\"/>\n   Snippet: Part Number: 19200-RDV-J01. $158.71 MSRP: $269.83. You Save: $111.12 (42%). ADD TO CART. Make sure this part fits, Select Vehicle. Product Specifications. Other ...\n\n4. Water Pump Assembly - Acura (19200-R40-A01)\n   URL: <url id=\"49043a\"/>\n   Snippet: Acura; Part Number: 19200-R40-A01. Details. Brand: Parts. SKU: 19200-R40-A01. Other Names: Water Pump. Description: TSX. 2.4L. Notes: Includes Water Pump Gasket ...\n\n5. Honda Genuine Water Pump, K24A2/8 (04-08 TSX/03-07 ...\n   URL: <url id=\"f948a6\"/>\n   Snippet: Honda-Acura Genuine OEM Water Pump, K24A2/8 (04-08 TSX/03-07 Accord/CR-V/Element), 19200-RAA-A01. Home > Engine > Cooling. Honda-Acura Genuine OEM Water ...\n\n6. New OEM water pump does not fit\n   URL: <url id=\"7cd8f4\"/>\n   Snippet: 2004-2008 Acura TSX Water Pump Assembly 19200-RAA-A01 | OEM Parts Online · 1. the original pump @ 160k miles with the shorter blades. · 2. One of ...\n\n8. OEM for 2008-2014 Honda Acura TSX Accord CR ... - eBay\n   URL: <url id=\"f486a1\"/>\n   Snippet: OEM for 2008-2014 Honda Acura TSX Accord CR-V K24Z3 2.4L 19200R40A01 Water Pump ; Condition. New ; Quantity. 1 sold. 1 available ; Item number. 186833318284.\n\n9. Honda OEM TSX K24 Water Pump - Touge Factory\n   URL: <url id=\"3ab983\"/>\n   Snippet: 19200-RAA-A01, Honda Accord water pump, kswap water pump, tsx water pump, k24 water pump.\n\n10. Water Pump Assembly - Acura (19200-RAA-A01)\n   URL: <url id=\"9a7327\"/>\n   Snippet: Water Pump Assembly - Acura (19200-RAA-A01) ; Manufacturer: Acura ; Part Number: 19200-RAA-A01 ; Replaces: 19200-RBB-003, 19200-RBB-013, 19200-RFE-003 ...\n\n\n\n---\n\n1 pages are not scrapable.\n\n7. Water Pump for 2009-2014 Acura TSX (PG-1967249)\n   URL (unscrapable): <url id=\"c6b37a\"/>\n   Snippet: Item Weight, ‎12 ounces. Product Dimensions, ‎3 x 6 x 6 inches. Manufacturer Part Number, ‎PG-1967249. OEM Part Number, ‎131-2390, 135-1620, 18-1983, 19200-R40- ...\n\n\n\n### Tool Call ID: 2\ntool_name: webpage_scrape\ntool_arguments: {'urls': ['<url id=\"2d0a73\"/>', '<url id=\"9dceb2\"/>', '<url id=\"b64ceb\"/>', '<url id=\"49043a\"/>']}\ntool_result:\nSuccessfully scraped 4 webpages:\n\n## 1. <url id=\"2d0a73\"/>\n\n**Content Length:** 743 characters (markdown)\n\n**Content:**\nPlease enable cookies. \n\n# Sorry, you have been blocked\n\n## You are unable to access r.parts\n\n## Why have I been blocked?\n\nThis website is using a security service to protect itself from online attacks. The action you just performed triggered the security solution. There are several actions that could trigger this block including submitting a certain word or phrase, a SQL command or malformed data. \n\n## What can I do to resolve this?\n\nYou can email the site owner to let them know you were blocked. Please include what you were doing when this page came up and the Cloudflare Ray ID found at the bottom of this page. \nCloudflare Ray ID: **991904375b85d6e1** • Your IP: Click to reveal 47.90.163.148 • Performance \\& security by Cloudflare\n...\n\n---\n\n## 2. <url id=\"9dceb2\"/>\n\n**Content Length:** 743 characters (markdown)\n\n**Content:**\nPlease enable cookies. \n\n# Sorry, you have been blocked\n\n## You are unable to access r.parts\n\n## Why have I been blocked?\n\nThis website is using a security service to protect itself from online attacks. The action you just performed triggered the security solution. There are several actions that could trigger this block including submitting a certain word or phrase, a SQL command or malformed data. \n\n## What can I do to resolve this?\n\nYou can email the site owner to let them know you were blocked. Please include what you were doing when this page came up and the Cloudflare Ray ID found at the bottom of this page. \nCloudflare Ray ID: **991d9cf0383d6fb6** • Your IP: Click to reveal 47.89.155.175 • Performance \\& security by Cloudflare\n...\n\n---\n\n## 3. <url id=\"49043a\"/>\n\n**Content Length:** 743 characters (markdown)\n\n**Content:**\nPlease enable cookies. \n\n# Sorry, you have been blocked\n\n## You are unable to access r.parts\n\n## Why have I been blocked?\n\nThis website is using a security service to protect itself from online attacks. The action you just performed triggered the security solution. There are several actions that could trigger this block including submitting a certain word or phrase, a SQL command or malformed data. \n\n## What can I do to resolve this?\n\nYou can email the site owner to let them know you were blocked. Please include what you were doing when this page came up and the Cloudflare Ray ID found at the bottom of this page. \nCloudflare Ray ID: **991904376eb54e62** • Your IP: Click to reveal 47.252.38.188 • Performance \\& security by Cloudflare\n...\n\n---\n\n## 4. <url id=\"b64ceb\"/>\n\n**Content Length:** 3997 characters (markdown)\n\n**Content:**\n# Genuine Acura TSX Water Pump\n\nH2O Pump \n* Select Vehicle by Model\n* Select Vehicle by VIN\n\n**Select Vehicle by Model** \nMake \nModel \nYear\nGo\n**or** \n**Select Vehicle by VIN** \nGo \nFor the most accurate results, select vehicle by your VIN (Vehicle Identification Number). \n\n## 3 Water Pumps found\n\n*\n View related parts \n\n ### Acura TSX Water Pump (Yamada)\n\n Part Number: 19200-RDV-J01 \n $158.71 MSRP: $269.83 \n You Save: $111.12 (42%) \n ADD TO CART \n Make sure this part fits, Select Vehicle \n Product Specifications\n * Other Name: Water Pump (Yamada) ; Water Pump Assembly; Water Pump, Water Pump Assembly\n * Replaces: 19200-RDM-A01, 19200-R70-A11, 19200-RDM-A02, 19200-RCA-A01\n * Warranty: This genuine part is guaranteed by Acura's factory warranty.\n*\n View related parts \n\n ### Acura TSX Engine Water Pump\n\n Part Number: 19200-R40-A01 \n $117.38 MSRP: $167.63 \n You Save: $50.25 (30%) \n ADD TO CART \n Make sure this part fits, Select Vehicle \n Product Specifications\n * Other Name: Water Pump Assembly; Water Pump\n * Warranty: This genuine part is guaranteed by Acura's factory warranty.\n*\n View related parts \n\n ### Acura TSX Engine Water Pump\n\n Part Number: 19200-RAA-A01 \n $189.38 MSRP: $270.47 \n You Save: $81.09 (30%) \n Make sure this part fits, Select Vehicle \n Product Specifications\n * Other Name: Water Pump Assembly; Water Pump\n * Replaces: 19200-RAD-003, 19200-RFE-003, 19200-RBB-003, 19200-RBB-013\n* Warranty: This genuine part is guaranteed by Acura's factory warranty. \n\n## Acura TSX Water Pump\n\nThe specific part is the Acura TSX Water Pump, one of the most famous and appreciated parts for its high reliability and wonderful work in regulating the temperature of the engine. The Acura TSX Water Pump is meant for circulating coolant in the engine's water jacket and plays a significant role in absorbing the heat produced by the combustion process and transferring the said heat to the radiator to ensure proper running of the engine. Applicable to models...\n\n---\n\n\n\n        \n\nOutput the markdown summary directly, no other words"'''
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {   "role": "user", 
        "content": prompt
    },
]

extra_body = {
    "extra_body": {
        "google": {
            "thinkingConfig": {
                "includeThoughts": True,
                "thinkingBudget": 1024
            },
            "thought_tag_marker": "think"
        }
    }  
}


def setup_openai_client():
    """设置OpenAI客户端连接到Google Vertex AI"""
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-easongy-poc")
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
    
    print(f"使用项目: {PROJECT_ID}, 区域: {LOCATION}")
    
    # 认证
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    # print(f"Credentials token : {credentials.token}")
    
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


@timing_decorator
def generate_with_openai(
  client: Any,
  messages: List[Any],
  extra_body: Dict[Any, Any],
  max_tokens=1024,
  stream=False,
  temperature=0,
  model: str = "google/gemini-2.5-pro"):

    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        stream=stream,
        temperature=temperature,
        # reasoning_effort="medium",    # gemini-2.5-pro not supported yet
        extra_body=extra_body
    )

    return response


def main():

    client = setup_openai_client()

    # Non-Stream
    stream = False
    response = generate_with_openai(
      client=client,
      messages=messages,
      max_tokens=max_tokens,
      stream=stream,
      temperature=temperature,
      extra_body=extra_body,
      model=model)

    # print(response)

    non_stream_response = response.choices[0].message.content
    thought, answer = parse_response_with_tags(
        response.choices[0].message.content, "think")


    print("####### Non-Stream #######")
    if len(thought) != 0:
        print("--- Thought ---")
    else:
        print(non_stream_response)
        # print(thought)
    # if len(answer) != 0:
    #     print("\n--- Answer ---")
        # print(answer)

    # Stream
    stream_response = ""
    stream = True
    response = generate_with_openai(
      client=client,
      messages=messages,
      max_tokens=max_tokens,
      stream=stream,
      temperature=temperature,
      extra_body=extra_body,
      model=model)
    stream = True
    
    print("####### Stream #######")
    stream_response = ""
    for chunk in response:
        # print(chunk)
        # print(chunk.choices[0].delta)
        stream_response += chunk.choices[0].delta.content
    
    think_start_index = stream_response.find(think_start_tag)
    think_end_index = stream_response.find(think_end_tag)

    if think_start_index != -1 and think_end_index != -1:
        print("--- Thought ---")
    else:
        print(non_stream_response)
        print(stream_response)

    return non_stream_response, stream_response
        

if __name__ == "__main__":
    non_stream_answers = list()
    stream_answers = list()

    for i in range(100):
        print('#' * 10 + f' {i+1} ' + '#' * 10)
        try:
            non_stream_response, stream_response = main()
            non_stream_answers.append(non_stream_response)
            stream_answers.append(stream_response)
        except Exception as e:
            print("Exception")

    print('#' * 100)
    with open("/tmp/stream_answers.json", "w") as f:
        json.dump(stream_answers, f)

    with open("/tmp/non_stream_answers.json", "w") as f:
        json.dump(non_stream_answers, f)

    for index, (non_stream_content, stream_content) in enumerate(zip(non_stream_answers, stream_answers)):
        print('#' * 20 + f' {index} ' + '#' * 20)
        
        think_start_index = non_stream_content.find(think_start_tag)
        think_end_index = non_stream_content.find(think_end_tag)

        if think_start_index != -1 and think_end_index != -1:
            print("Non stream with reason")

        think_start_index = stream_content.find(think_start_tag)
        think_end_index = stream_content.find(think_end_tag)

        if think_start_index != -1 and think_end_index != -1:
            print("Stream with reason")
