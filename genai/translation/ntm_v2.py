#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCP Cloud Translation Basic Edition API 调用示例
支持多种语言翻译，包含错误处理和配置选项
"""

import os
import sys
from typing import List, Optional, Dict, Any
from google.cloud import translate_v2 as translate
from google.api_core import exceptions

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.time_utils import timing_decorator


class CloudTranslationBasic:
    """GCP Cloud Translation Basic Edition 客户端类"""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        初始化翻译客户端
        
        Args:
            project_id: GCP项目ID，如果不提供则从环境变量获取
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("请提供project_id或设置GOOGLE_CLOUD_PROJECT环境变量")
        
        try:
            # 使用Basic Edition (v2) API
            self.client = translate.Client()
            print(f"✅ 成功初始化Cloud Translation Basic Edition客户端")
        except Exception as e:
            print(f"❌ 初始化翻译客户端失败: {e}")
            raise

    @timing_decorator
    def get_supported_languages(self, target_language: str = 'zh') -> List[Dict[str, str]]:
        """
        获取支持的语言列表
        
        Args:
            target_language: 目标语言代码，用于显示语言名称
            
        Returns:
            支持的语言列表
        """
        try:
            languages = self.client.get_languages(target_language=target_language)
            print(f"📋 支持的语言数量: {len(languages)}")
            return languages
        except exceptions.GoogleAPIError as e:
            print(f"❌ 获取支持语言失败: {e}")
            return []

    @timing_decorator
    def detect_language(self, text: str) -> Optional[Dict[str, Any]]:
        """
        检测文本语言
        
        Args:
            text: 要检测的文本
            
        Returns:
            检测结果，包含语言代码和置信度
        """
        try:
            result = self.client.detect_language(text)
            print(f"🔍 检测到语言: {result['language']} (置信度: {result['confidence']:.2f})")
            return result
        except exceptions.GoogleAPIError as e:
            print(f"❌ 语言检测失败: {e}")
            return None

    @timing_decorator
    def translate_text(
        self, 
        texts: List[str], 
        target_language: str = 'zh',
        source_language: Optional[str] = None,
        format_: str = 'text'
    ) -> List[Dict[str, Any]]:
        """
        翻译文本
        
        Args:
            texts: 要翻译的文本列表
            target_language: 目标语言代码 (如: 'zh', 'en', 'ja', 'ko')
            source_language: 源语言代码，如果不指定则自动检测
            format_: 文本格式 ('text' 或 'html')
            
        Returns:
            翻译结果列表
        """
        try:
            results = []
            
            for text in texts:
                if not text.strip():
                    continue
                    
                # 执行翻译
                result = self.client.translate(
                    text,
                    target_language=target_language,
                    source_language=source_language,
                    format_=format_
                )
                
                results.append(result)
                
                # 打印翻译结果
                detected_lang = result.get('detectedSourceLanguage', source_language or 'auto')
                print(f"🌐 [{detected_lang} → {target_language}] {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"📝 翻译结果: {result['translatedText']}")
                print("-" * 80)
            
            return results
            
        except exceptions.GoogleAPIError as e:
            print(f"❌ 翻译失败: {e}")
            return []
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            return []

    @timing_decorator
    def translate_single(
        self, 
        text: str, 
        target_language: str = 'zh',
        source_language: Optional[str] = None
    ) -> Optional[str]:
        """
        翻译单个文本的便捷方法
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言代码
            source_language: 源语言代码
            
        Returns:
            翻译后的文本
        """
        results = self.translate_text([text], target_language, source_language)
        return results[0]['translatedText'] if results else None


def demo_basic_translation():
    """演示基本翻译功能"""
    print("=" * 80)
    print("🚀 GCP Cloud Translation Basic Edition 演示")
    print("=" * 80)
    
    # 初始化翻译客户端
    translator = CloudTranslationBasic(project_id="ali-icbu-gpu-project")
    
    # 演示文本
    demo_texts = [
        "Hello, how are you today?",
        "Cloud Translation API utilise la technologie de traduction automatique neuronale de Google.",
        "こんにちは、今日はいかがですか？",
        "안녕하세요, 오늘 어떻게 지내세요?",
        "Hola, ¿cómo estás hoy?"
    ]
    
    print("\n📋 演示文本:")
    for i, text in enumerate(demo_texts, 1):
        print(f"{i}. {text}")
    
    # 1. 检测语言
    print("\n" + "=" * 50)
    print("🔍 1. 语言检测演示")
    print("=" * 50)
    
    for text in demo_texts[:3]:  # 只检测前3个
        translator.detect_language(text)
    
    # 2. 翻译为中文
    print("\n" + "=" * 50)
    print("🌐 2. 翻译为中文演示")
    print("=" * 50)
    
    translator.translate_text(demo_texts, target_language='zh')
    
    # 3. 翻译为英文
    print("\n" + "=" * 50)
    print("🌐 3. 翻译为英文演示")
    print("=" * 50)
    
    translator.translate_text(demo_texts, target_language='en')
    
    # 4. 单个文本翻译
    print("\n" + "=" * 50)
    print("📝 4. 单个文本翻译演示")
    print("=" * 50)
    
    single_text = "人工智能正在改变我们的世界"
    result = translator.translate_single(single_text, target_language='en')
    print(f"原文: {single_text}")
    print(f"译文: {result}")


def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 80)
    print("🔧 高级功能演示")
    print("=" * 80)
    
    translator = CloudTranslationBasic(project_id="ali-icbu-gpu-project")
    
    # 1. 获取支持的语言
    print("\n📋 支持的语言列表 (前10个):")
    languages = translator.get_supported_languages()
    for lang in languages[:10]:
        print(f"  {lang['language']}: {lang['name']}")
    
    # 2. HTML格式翻译
    print("\n🌐 HTML格式翻译:")
    html_text = "<p>This is a <strong>bold</strong> text with <em>emphasis</em>.</p>"
    result = translator.translate_text([html_text], target_language='zh', format_='html')
    if result:
        print(f"原文: {html_text}")
        print(f"译文: {result[0]['translatedText']}")
    
    # 3. 指定源语言翻译
    print("\n🎯 指定源语言翻译:")
    french_text = "Bonjour le monde"
    result = translator.translate_single(
        french_text, 
        target_language='zh', 
        source_language='fr'
    )
    print(f"法语原文: {french_text}")
    print(f"中文译文: {result}")


if __name__ == "__main__":
    try:
        # 基本功能演示
        demo_basic_translation()
        
        # 高级功能演示
        demo_advanced_features()
        
        print("\n" + "=" * 80)
        print("✅ 演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        sys.exit(1)
