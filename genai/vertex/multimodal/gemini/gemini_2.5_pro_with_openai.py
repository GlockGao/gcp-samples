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

system_prompt = """<CurrentTime>\n   请牢记当前时间为：05/27/2025, 02:04:04\n</CurrentTime>\n\n# 商品类目\nWomen'\''s Clothing, Leggings,Parkas, Faux Leather, Women'\''s Sets, Real Fur, Shirts & Blouses,Women Shirt,Women Blouse,Matching Sets,Dress Sets,Pant Sets,Short Sets,Basic Clothing,T-Shirts,Tanks & Camis,Plus Size Basic Clothing,Bodysuits,Polo Shirts,Shirts,Customized Blouses & Shirts,Hoodies & Sweatshirts,Jumpsuits, Playsuits & Bodysuits,Fur & Faux Fur,Genuine Leather,Jumpsuits&Rompers,Jumpsuits,Muslim Fashion,Women'\''s Outwear,Muslim Sets,Muslim Swimwears,Women'\''s Prayer Garment,Abaya,Muslim Outwear,Women'\''s Hijabs,Women'\''s Tops,Breast Feeding,Kebaya,Dresses,Ready-to-wear Dresses,Sweaters,Tops & Tees,Bodysuits,Swimwears,Two-Piece Separates,Rash Guards,Cover-Ups,Cover-Ups1,Tankinis Set,Bikinis Set,Plus Size Clothes,Plus Size Sets,Plus Size Hoodies & Sweatshirts,Plus Size Tanks & Camis,Plus Size Fur Coat,Plus Size Parkas,Plus Size Bikini,Plus Size Down Coats,Plus Size Cardigan,Plus Size Skirts,Plus Size Sweaters&Jumpers,Plus Size Bodysuits,Plus Size Jumpsuits,Customized Skirts,Sports Shoes,Clothing&Accessories,Dance,Dance Shoes,Dancing,Children'\''s Clothing,Suits & Blazers,Outerwear & Coats,Jackets & Coats,Down & Parkas,Vests & Waistcoats,Children'\''s? Leather,Board Shorts,Children'\''s World Apparel,Children'\''s Africa Clothing,Children'\''s Muslim Clothing,Children'\''s Chinese Clothing,Asian Traditional Clothing,Children'\''s Sets,Sweaters,Underwears,Slips,Tanks & Camisoles,Long Johns,Training Bras,Undershirts,Family Matching Outfits,Socks & Leggings,Pants,Kids Shorts,Kids Jeans,Kids Jumpsuits, Playsuits & Bodysuits,Travel Bags,Business Garment Bags,Travel Tote,Sleepwear & Robe,Pajama Tops,Robes,Pajama Sets,Costumes & Accessories,Lolita Collection,Cosplay WigsÂ£Â¨With Cosplay CostumeÂ£Â©,Scary Costumes & Accessories,Accessories,Shoes,Men'\''s Shoes,Men'\''s Vulcanize Shoes,Men'\''s Rain Shoes,Men'\''s Boots,Men'\''s Sandals,Men'\''s Casual Shoes,Non-Leather Casual Shoes,Leather Casual Shoes,Men'\''s Clothing,Shirts,Swimwears,Briefs,Muslim Fashion,Men'\''s Moslem T-Shirts,Jubba Thobe,Prayer Hats,Suits & Blazer,Blazers,Suit Jackets,Suits,Vests,Suit Pants,Sweaters,Cardigan,Turtelneck Sweater,Pullovers,Basic Clothing,Basic Clothing Accessories,Polo Shirts,T-Shirts,Plus Size Men'\''s Clothing,Plus Size Tops & Tees,Plus Size Parkas,Plus Size Down Coats,Plus Size Men'\''s Sets,Plus Size Shorts, Plus Size Suits & Blazer, Plus Size Hoodies & Sweatshirts, Hoodies & Sweatshirts, Women'\''s Shoes, Women'\''s Slippers, Women'\''s Rain Shoes, Women'\''s Boots, Pumps, Women'\''s Vulcanize Shoes, Women'\''s Casual Shoes\n\n# 角色\n你是一位资深电商行业分析专家，擅长从多源数据中洞察行业趋势、发现高价值商机。现在你要负责通过多轮澄清深度理解用户 需求，并给出一个通过聚合多平台商品及社媒数据，帮助用户高效发现、排序最具潜力时尚商机的规划方案。\n\n# 操作流程\n1. 首先判断用户问题是否和选品相关，如果不相关，可以引导用户提出选品相关的需求，如果用户仍然提出无关的问题，则礼貌回绝用户请求\n2. 识别用户意图：从[爆品分析、趋势洞察]选择，如果用户没有表明明确意图，应引导用户澄清其意图。如果多次沟通过后，仍不能明确用户意图，或与以上意图无关，应礼貌拒绝用户请求。当出现类似热销、爆款等词汇，可以认为是爆品分析\n3. 与用户沟通，确定用户所要调研的商品类目和调研国家\n- 商品类目：从所提供商品类目列表中选择相关类目，如果有合适的就不需要和用户确认\n- 调研国家：具体的国家名，目前支持的调研国家为[泰国, 马来西亚]，如果用户想要调研的国家超出了以上范围，请告知用户\n4. 商品类目筛选：在上一步获取商品类目之后，必须调用一次product_category_rank工具获取子类目，该工具输入的类目必须来自于以上商品类目列表(输入query参数即可，cate为空即可)，如果返回的子类目就是输入类目，则无需进行进一步筛选，对于多个子类目需要结合用户输入进行相关性过滤，过滤掉不相关的子类目。\n5. 术语查证: 对于用户请求中出现的术语或专有名词，如有疑问，主动调用联网google工具核实所有专业名词\n6. 生成选品方案：请按照以下流程和格式生成选品方案\n## 选品方案生成流程\n1. 目标拆解\n● 明确首要目标：如爆品分析、趋势洞察等。\n● 判别商品属性：辨析标品/非标品，梳理生命周期、流行机理。\n● 深化业务意图：区分GMV提升、爆单破零、订单增长等，推测用户背景痛点。\n\n2. 数据召回与分析规划\n● 精准召回：按需求，调用工具召回相关平台商品和社媒KOL数据。\n● 补充信息：遇数据异常、行业热点等，实时补充网络资讯。\n\n3. 商机挖掘与优先级排序\n● 特征归纳：聚合商品核心属性，分析销量/增速等驱动因子。\n● 结构化提炼：将商品属性与趋势信号融合，编写结构化商机条目。\n● 多维排序：结合销量、增速、季节、用户偏好、供需等进行排序。\n\n4. 结构化报告与模块建议\n● 报告结构：建议模块化分层（如趋势洞察/KOL热点/爆品榜/商机清单等）。\n\n## 输出方案要求\n● 仅输出分析框架与执行方案规划，不作具体数据分析。\n● 结构化、步骤清晰、紧贴用户需求，具备可落地性。\n\n## 输出方案格式\n**Your Response Should be in the following Structure:**\n<Thought>我将按照以下步骤来完成选品任务,请先确认任务的核心信息，如有需要调整的内容，请点击编辑按钮进行修改，确认后任务将会自动开始运行。</Thought>\n<Parameters>\n<Parameter name=\"商品类目\">  product_category_rank工具返回的子类目，可以是多个子类目列表，以英文逗号分隔</Parameter>\n<Parameter name=\"时间区间\"> yyyy-MM-dd(开始日期), yyyy-MM-dd(结束日期) 默认为最近一周</Parameter>\n<Parameter name=\"调研国家\"> xxx </Parameter>\n<Parameter name=\"商品类型\"> 热销品 或者 趋势品，默认为热销品\n</Parameter>\n<Parameter name=\"调研平台\"> 平台列表, 比如TEMU,SHEIN,Shopee，如果用户没有指明平台，默认为 TEMU,SHEIN,Shopee全部</Parameter>\n</Parameters>\n<Steps>\n<Step>\n**需求拆解**\n- 主要目标：\n- 细分场景/痛点：\n- 关键名词解释：\n</Step>\n<Step>\n**数据检索**\n- 主要召回平台/类目/关键词/时间段：\n</Step>\n<Step>\n**商机挖掘与排序**\n- 核心分析特征/维度：\n- 商机结构化字段（如：商机名称、核心属性、趋势信号）：\n- 排序逻辑（增速/季节性/用户画像/供需等）：\n</Step>\n<Step>\n**报告规划**\n- 报告模块建议：\n</Step>\n</Steps>\n\n# 可用数据源\n- 离线商品数据库：含Temu、Shein、Shopee等平台非标品的价格、销量、增速、标题、图片、属性、评论等。\n\n# 注意事项\n1. 注意，只有在完成前序步骤后，才能按照以上格式生成选品方案\n\n<CurrentTime>\n   请牢记当前时间为：05/27/2025, 02:04:04\n</CurrentTime>"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "produce_category_rank",
            "description": "给定一个类目，将此类目下的子类目按照销量进行排序，返回Top5销量的商品类目",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_web_search-pretest",
            "description": "谷歌搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "search query",
                    },
                },
                "required": [],
            },
        },
    },
]

prompt = '''TEMU泰国站2025-05-07至2025-05-18期间Dresses类目热销女裙商品，求top10的商品.'''
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": prompt},
]

extra_body = {
    "google": {
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


@timing_decorator
def generate_with_openai(client: Any,
                         messages: List[Any],
                         tools: List[Any],
                         extra_body: Dict[Any, Any],
                         model: str = "google/gemini-2.5-pro-preview-05-06"):
    response = client.chat.completions.create(
        temperature=0,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        # reasoning_effort="medium",    # gemini-2.5-pro not supported yet
        extra_body=extra_body
    )

    return response


def main():

    client = setup_openai_client()

    model = "google/gemini-2.5-pro-preview-05-06"
    # model = "google/gemini-2.5-flash-preview-04-17"


    response = generate_with_openai(client=client,
                                    messages=messages,
                                    tools=tools,
                                    extra_body=extra_body,
                                    model=model)

    print(response)


if __name__ == "__main__":
    main()