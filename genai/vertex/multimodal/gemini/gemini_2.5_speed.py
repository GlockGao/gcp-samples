"""
简化版本 - 快速获取 Gemini Output Token 数量
包含 Token 生成速率统计
"""

from google import genai
import os
import time
from utils.time_utils import timing_decorator


@timing_decorator
def get_output_tokens(prompt: str, model: str = "gemini-2.5-pro"):
    """
    快速获取 output token 数量和生成速率
    
    Args:
        prompt: 输入提示
        model: 模型名称
    
    Returns:
        dict: 包含各种 token 数量和生成速率的字典
    """
    # 设置客户端
    client = genai.Client()
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成内容
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    
    # 记录结束时间
    end_time = time.time()
    generation_time = end_time - start_time
    
    # print(f'response : {response}')
    
    usage = response.usage_metadata

    # print(f'usage    : {usage}')
    # 获取 prompt tokens
    prompt_token_count = getattr(usage, 'prompt_token_count', 0)

    # 获取 output tokens
    candidates_token_count = getattr(usage, 'candidates_token_count', 0)

    # 获取 thinking tokens
    thoughts_token_count = getattr(usage, 'thoughts_token_count', 0)

    # 获取 total tokens
    total_token_count = getattr(usage, 'total_token_count', 0)
    output_token_count = total_token_count - prompt_token_count

    # print(f'prompt_token_count     : {prompt_token_count}')
    # print(f'candidates_token_count : {candidates_token_count}')
    # print(f'thoughts_token_count   : {thoughts_token_count}')
    # print(f'output_token_count     : {output_token_count}')
    # print(f'total_token_count      : {total_token_count}')

    
    # 计算生成速率
    output_tokens_per_sec = output_token_count / generation_time if generation_time > 0 else 0
    total_tokens_per_sec = total_token_count / generation_time if generation_time > 0 else 0

    # print(f'output_tokens_per_sec      : {output_tokens_per_sec}')
    # print(f'total_tokens_per_sec      : {total_tokens_per_sec}')
    
    # 返回详细的 token 信息
    return {
        'output_token_count': output_token_count,
        'candidates_token_count': candidates_token_count,
        'thoughts_token_count': thoughts_token_count,
        'prompt_token_count': prompt_token_count,
        'total_token_count': total_token_count,
        'generation_time': generation_time,
        'output_tokens_per_sec': output_tokens_per_sec,
        'total_tokens_per_sec': total_tokens_per_sec,
        'response_text': response.text
    }


def generate_optimal_prompts():
    """
    生成目标为 input token ~20, output token ~700 的提示词
    
    Returns:
        list: 优化的提示词列表
    """
    # 设计的提示词，目标：简短输入，长输出
    optimal_prompts = [
        # 故事创作类 (短提示，长输出)
        "写一个科幻短故事, 字数在700字左右",
        # "创作一个悬疑故事",
        # "写个童话故事",
        
        # # 详细解释类
        # "详细解释机器学习",
        # "介绍区块链技术",
        # "解释量子计算原理",
        
        # # 分析评论类
        # "分析人工智能的影响",
        # "评论气候变化问题",
        # "讨论远程工作趋势",
        
        # # 指南教程类
        # "写Python入门教程",
        # "制作健身计划指南",
        # "烹饪技巧大全",
        
        # # 创意写作类
        # "写一首现代诗",
        # "创作广告文案",
        # "设计产品介绍"
    ]
    
    return optimal_prompts
    

def main():
    """主函数 - 测试Token生成速度"""
    print("input ~20 tokens, output ~700 tokens\n")
    
    try:
        optimal_prompts = generate_optimal_prompts()
        for i, prompt in enumerate(optimal_prompts, 1):
            print(f"\n测试提示 {i}: '{prompt}'")

            print('#' * 100)
            print(f"速度测试 - gemini-2.5-pro")
            token_info = get_output_tokens(prompt, model="gemini-2.5-pro")
            print(f"输入 tokens           : {token_info['prompt_token_count']:>10}")
            print(f"思考 tokens           : {token_info['thoughts_token_count']:>10}")
            print(f"输出 tokens           : {token_info['candidates_token_count']:>10}")
            print(f"总输出 tokens         : {token_info['output_token_count']:>10}")
            print(f"输出生成速率          : {token_info['output_tokens_per_sec']:>10.2f} tokens/秒")
            print(f"输入&输出生成速率      : {token_info['total_tokens_per_sec']:>10.2f} tokens/秒")

            print('#' * 100)
            print(f"速度测试 - gemini-2.5-flash")
            token_info = get_output_tokens(prompt, model="gemini-2.5-flash")
            print(f"输入 tokens           : {token_info['prompt_token_count']:>10}")
            print(f"思考 tokens           : {token_info['thoughts_token_count']:>10}")
            print(f"输出 tokens           : {token_info['candidates_token_count']:>10}")
            print(f"总输出 tokens         : {token_info['output_token_count']:>10}")
            print(f"输出生成速率          : {token_info['output_tokens_per_sec']:>10.2f} tokens/秒")
            print(f"输入&输出生成速率      : {token_info['total_tokens_per_sec']:>10.2f} tokens/秒")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
