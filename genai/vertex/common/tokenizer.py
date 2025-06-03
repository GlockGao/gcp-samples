import os
from typing import List, Union, Optional, Dict, Any
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
from utils.time_utils import timing_decorator
import requests
from PIL import Image
from io import BytesIO


class GeminiTokenizer:
    """GCP Gemini模型的Token计数器类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Gemini Tokenizer
        
        Args:
            api_key: Gemini API密钥，如果不提供则从环境变量GEMINI_API_KEY获取
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("未找到GEMINI_API_KEY，请设置环境变量或传入api_key参数")
            
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=HttpOptions(api_version="v1")
        )
    
    @timing_decorator
    def count_tokens_text(self, 
                         content: str, 
                         model: str = "gemini-2.0-flash-preview") -> Dict[str, Any]:
        """
        计算文本内容的token数量
        
        Args:
            content: 要计算token的文本内容
            model: 使用的Gemini模型名称
            
        Returns:
            包含token计数信息的字典
        """
        try:
            response = self.client.models.count_tokens(
                model=model,
                contents=content
            )
            
            result = {
                "total_tokens": response.total_tokens,
                "model": model,
                "content_type": "text",
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
            
            print(f"文本Token计数结果: {result['total_tokens']} tokens")
            return result
            
        except Exception as e:
            print(f"计算文本token时出错: {str(e)}")
            raise
    
    def _load_image_from_url(self, image_url: str) -> Image.Image:
        """
        从URL加载图片
        
        Args:
            image_url: 图片URL
            
        Returns:
            PIL Image对象
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"无法加载图片 {image_url}: {str(e)}")
    
    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """
        从本地路径加载图片
        
        Args:
            image_path: 本地图片路径
            
        Returns:
            PIL Image对象
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            raise ValueError(f"无法加载图片 {image_path}: {str(e)}")
    
    @timing_decorator
    def count_tokens_multimodal(self, 
                               contents: List[Union[str, Image.Image]], 
                               model: str = "gemini-2.0-flash-preview-image-generation") -> Dict[str, Any]:
        """
        计算多模态内容（文本+图片）的token数量
        
        Args:
            contents: 包含文本和PIL Image对象的内容列表
            model: 使用的Gemini模型名称
            
        Returns:
            包含token计数信息的字典
        """
        try:
            response = self.client.models.count_tokens(
                model=model,
                contents=contents
            )
            
            # 分析内容类型
            content_types = []
            for item in contents:
                if isinstance(item, str):
                    content_types.append("text")
                elif isinstance(item, Image.Image):
                    content_types.append("image")
                else:
                    content_types.append("unknown")
            
            result = {
                "total_tokens": response.total_tokens,
                "model": model,
                "content_type": "multimodal",
                "content_types": content_types,
                "content_count": len(contents)
            }
            
            print(f"多模态Token计数结果: {result['total_tokens']} tokens")
            print(f"内容类型: {', '.join(content_types)}")
            return result
            
        except Exception as e:
            print(f"计算多模态token时出错: {str(e)}")
            raise
    
    @timing_decorator
    def count_tokens_image_from_url(self, 
                                   image_url: str, 
                                   prompt: str = "请描述这张图片",
                                   model: str = "gemini-2.0-flash-preview-image-generation") -> Dict[str, Any]:
        """
        计算图片(从URL)+文本提示的token数量
        
        Args:
            image_url: 图片的HTTP URL
            prompt: 文本提示
            model: 使用的Gemini模型名称
            
        Returns:
            包含token计数信息的字典
        """
        try:
            image = self._load_image_from_url(image_url)
            contents = [prompt, image]
            
            result = self.count_tokens_multimodal(contents, model)
            result["image_url"] = image_url
            result["prompt"] = prompt
            return result
        except Exception as e:
            print(f"计算图片token时出错: {str(e)}")
            raise
    
    @timing_decorator
    def count_tokens_image_from_path(self, 
                                    image_path: str, 
                                    prompt: str = "请描述这张图片",
                                    model: str = "gemini-2.0-flash-preview-image-generation") -> Dict[str, Any]:
        """
        计算图片(从本地路径)+文本提示的token数量
        
        Args:
            image_path: 本地图片路径
            prompt: 文本提示
            model: 使用的Gemini模型名称
            
        Returns:
            包含token计数信息的字典
        """
        try:
            image = self._load_image_from_path(image_path)
            contents = [prompt, image]
            
            result = self.count_tokens_multimodal(contents, model)
            result["image_path"] = image_path
            result["prompt"] = prompt
            return result
        except Exception as e:
            print(f"计算图片token时出错: {str(e)}")
            raise
    
    @timing_decorator
    def count_tokens_image_object(self, 
                                 image: Image.Image, 
                                 prompt: str = "请描述这张图片",
                                 model: str = "gemini-2.0-flash-preview-image-generation") -> Dict[str, Any]:
        """
        计算图片对象+文本提示的token数量
        
        Args:
            image: PIL Image对象
            prompt: 文本提示
            model: 使用的Gemini模型名称
            
        Returns:
            包含token计数信息的字典
        """
        try:
            contents = [prompt, image]
            
            result = self.count_tokens_multimodal(contents, model)
            result["image_size"] = image.size
            result["image_mode"] = image.mode
            result["prompt"] = prompt
            return result
        except Exception as e:
            print(f"计算图片token时出错: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的Gemini模型列表
        
        Returns:
            可用模型名称的列表
        """
        try:
            models = self.client.models.list()
            model_names = [model.name for model in models]
            print(f"可用的Gemini模型: {len(model_names)}个")
            for name in model_names:
                print(f"  - {name}")
            return model_names
        except Exception as e:
            print(f"获取模型列表时出错: {str(e)}")
            return []


# 便捷函数，保持向后兼容性
gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    # 创建全局tokenizer实例
    _global_tokenizer = GeminiTokenizer(gemini_api_key)
else:
    _global_tokenizer = None
    print("警告: 环境变量 'GEMINI_API_KEY' 未设置，无法创建全局tokenizer实例")


# 便捷函数，使用全局tokenizer实例
def count_tokens_text(content: str, model: str = "gemini-2.0-flash-preview-image-generation") -> Optional[Dict[str, Any]]:
    """
    便捷函数：计算文本内容的token数量
    
    Args:
        content: 要计算token的文本内容
        model: 使用的Gemini模型名称
        
    Returns:
        包含token计数信息的字典，如果tokenizer未初始化则返回None
    """
    if _global_tokenizer is None:
        print("错误: 全局tokenizer未初始化，请设置GEMINI_API_KEY环境变量")
        return None
    return _global_tokenizer.count_tokens_text(content, model)


def count_tokens_image_from_url(image_url: str, 
                               prompt: str = "请描述这张图片",
                               model: str = "gemini-2.0-flash-preview-image-generation") -> Optional[Dict[str, Any]]:
    """
    便捷函数：计算图片(从URL)+文本提示的token数量
    
    Args:
        image_url: 图片的HTTP URL
        prompt: 文本提示
        model: 使用的Gemini模型名称
        
    Returns:
        包含token计数信息的字典，如果tokenizer未初始化则返回None
    """
    if _global_tokenizer is None:
        print("错误: 全局tokenizer未初始化，请设置GEMINI_API_KEY环境变量")
        return None
    return _global_tokenizer.count_tokens_image_from_url(image_url, prompt, model)


def count_tokens_image_from_path(image_path: str, 
                                prompt: str = "请描述这张图片",
                                model: str = "gemini-2.0-flash-preview-image-generation") -> Optional[Dict[str, Any]]:
    """
    便捷函数：计算图片(从本地路径)+文本提示的token数量
    
    Args:
        image_path: 本地图片路径
        prompt: 文本提示
        model: 使用的Gemini模型名称
        
    Returns:
        包含token计数信息的字典，如果tokenizer未初始化则返回None
    """
    if _global_tokenizer is None:
        print("错误: 全局tokenizer未初始化，请设置GEMINI_API_KEY环境变量")
        return None
    return _global_tokenizer.count_tokens_image_from_path(image_path, prompt, model)


# 保持向后兼容的函数名
count_token_text = count_tokens_text
count_token_image = count_tokens_image_from_url  # 默认使用URL方式


def main():
    """示例用法"""
    print("=== GCP Gemini Token计数器示例 (修复版) ===")
    
    # 示例1: 计算文本token
    print("\n1. 文本Token计数:")
    text_content = "Add some chocolate drizzle to the croissants."
    result = count_tokens_text(text_content)
    if result:
        print(f"文本内容: {result['content_preview']}")
        print(f"Token数量: {result['total_tokens']}")
    
    # 示例2: 计算图片+文本token (使用HTTP URL)
    print("\n2. 图片+文本Token计数 (HTTP URL):")
    # 使用一个公开的示例图片URL
    image_url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/croissant.jpeg"
    result = count_tokens_image_from_url(image_url, "Add some chocolate drizzle to the croissants.")
    if result:
        print(f"图片URL: {result['image_url']}")
        print(f"提示文本: {result['prompt']}")
        print(f"Token数量: {result['total_tokens']}")
    
    # 示例3: 使用类实例
    print("\n3. 使用GeminiTokenizer类:")
    try:
        tokenizer = GeminiTokenizer()
        
        # 计算长文本
        long_text = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"""
        
        result = tokenizer.count_tokens_text(long_text, "gemini-2.0-flash-preview-image-generation")
        print(f"长文本Token数量: {result['total_tokens']}")
        
        # 如果有本地图片文件，可以测试
        # result = tokenizer.count_tokens_image_from_path("path/to/local/image.jpg", "描述这张图片")
        
    except Exception as e:
        print(f"创建tokenizer实例失败: {str(e)}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
