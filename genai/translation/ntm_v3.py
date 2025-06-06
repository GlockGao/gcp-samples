#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCP Cloud Translation Advanced Edition API 调用示例
使用 Translation v3 API，支持更多高级功能如自定义模型、术语表等
"""

import os
import sys
from typing import List, Optional, Dict, Any
from google.cloud import translate_v3 as translate
from google.api_core import exceptions

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.time_utils import timing_decorator


class CloudTranslationAdvanced:
    """GCP Cloud Translation Advanced Edition 客户端类"""
    
    def __init__(self, project_id: Optional[str] = None, location: str = "global"):
        """
        初始化翻译客户端
        
        Args:
            project_id: GCP项目ID，如果不提供则从环境变量获取
            location: 翻译服务的位置，默认为"global"
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("请提供project_id或设置GOOGLE_CLOUD_PROJECT环境变量")
        
        self.location = location
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        
        try:
            # 使用Advanced Edition (v3) API
            self.client = translate.TranslationServiceClient()
            print(f"✅ 成功初始化Cloud Translation Advanced Edition客户端")
            print(f"📍 项目: {self.project_id}, 位置: {self.location}")
        except Exception as e:
            print(f"❌ 初始化翻译客户端失败: {e}")
            raise

    @timing_decorator
    def get_supported_languages(self, display_language_code: str = 'zh-CN') -> List[Dict[str, Any]]:
        """
        获取支持的语言列表
        
        Args:
            display_language_code: 用于显示语言名称的语言代码
            
        Returns:
            支持的语言列表
        """
        try:
            request = translate.GetSupportedLanguagesRequest(
                parent=self.parent,
                display_language_code=display_language_code,
            )
            
            response = self.client.get_supported_languages(request=request)
            languages = []
            
            for language in response.languages:
                languages.append({
                    'language_code': language.language_code,
                    'display_name': language.display_name,
                    'support_source': language.support_source,
                    'support_target': language.support_target
                })
            
            print(f"📋 支持的语言数量: {len(languages)}")
            return languages
            
        except exceptions.GoogleAPIError as e:
            print(f"❌ 获取支持语言失败: {e}")
            return []

    @timing_decorator
    def detect_language(self, content: str, mime_type: str = "text/plain") -> Optional[Dict[str, Any]]:
        """
        检测文本语言
        
        Args:
            content: 要检测的文本
            mime_type: MIME类型，支持 "text/plain" 或 "text/html"
            
        Returns:
            检测结果，包含语言代码和置信度
        """
        try:
            request = translate.DetectLanguageRequest(
                parent=self.parent,
                content=content,
                mime_type=mime_type,
            )
            
            response = self.client.detect_language(request=request)
            
            if response.languages:
                detected = response.languages[0]  # 取置信度最高的结果
                result = {
                    'language_code': detected.language_code,
                    'confidence': detected.confidence
                }
                print(f"🔍 检测到语言: {result['language_code']} (置信度: {result['confidence']:.2f})")
                return result
            else:
                print("❌ 未检测到语言")
                return None
                
        except exceptions.GoogleAPIError as e:
            print(f"❌ 语言检测失败: {e}")
            return None

    @timing_decorator
    def translate_text(
        self, 
        contents: List[str], 
        target_language_code: str = 'zh-CN',
        source_language_code: Optional[str] = None,
        mime_type: str = "text/plain",
        model: Optional[str] = None,
        glossary_config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        翻译文本
        
        Args:
            contents: 要翻译的文本列表
            target_language_code: 目标语言代码 (如: 'zh-CN', 'en-US', 'ja', 'ko')
            source_language_code: 源语言代码，如果不指定则自动检测
            mime_type: MIME类型，支持 "text/plain" 或 "text/html"
            model: 自定义模型ID (可选)
            glossary_config: 术语表配置 (可选)
            
        Returns:
            翻译结果列表
        """
        try:
            # 构建请求
            request = translate.TranslateTextRequest(
                parent=self.parent,
                contents=contents,
                mime_type=mime_type,
                target_language_code=target_language_code,
            )
            
            # 可选参数
            if source_language_code:
                request.source_language_code = source_language_code
            
            if model:
                request.model = f"projects/{self.project_id}/locations/{self.location}/models/{model}"
            
            if glossary_config:
                request.glossary_config = glossary_config
            
            # 执行翻译
            response = self.client.translate_text(request=request)
            
            results = []
            for i, translation in enumerate(response.translations):
                result = {
                    'translated_text': translation.translated_text,
                    'detected_language_code': translation.detected_language_code,
                    'model': translation.model if hasattr(translation, 'model') else None,
                    'glossary_config': translation.glossary_config if hasattr(translation, 'glossary_config') else None
                }
                results.append(result)
                
                # 打印翻译结果
                detected_lang = translation.detected_language_code or source_language_code or 'auto'
                original_text = contents[i]
                print(f"🌐 [{detected_lang} → {target_language_code}] {original_text[:50]}{'...' if len(original_text) > 50 else ''}")
                print(f"📝 翻译结果: {translation.translated_text}")
                if translation.model:
                    print(f"🤖 使用模型: {translation.model}")
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
        target_language_code: str = 'zh-CN',
        source_language_code: Optional[str] = None,
        mime_type: str = "text/plain"
    ) -> Optional[str]:
        """
        翻译单个文本的便捷方法
        
        Args:
            text: 要翻译的文本
            target_language_code: 目标语言代码
            source_language_code: 源语言代码
            mime_type: MIME类型
            
        Returns:
            翻译后的文本
        """
        results = self.translate_text([text], target_language_code, source_language_code, mime_type)
        return results[0]['translated_text'] if results else None

    @timing_decorator
    def batch_translate_text(
        self,
        input_configs: List[Dict[str, str]],
        output_config: Dict[str, str],
        target_language_codes: List[str],
        source_language_code: Optional[str] = None,
        models: Optional[Dict[str, str]] = None,
        glossaries: Optional[Dict[str, str]] = None
    ) -> str:
        """
        批量翻译文本文件
        
        Args:
            input_configs: 输入配置列表，每个配置包含 gcs_source 或 mime_type
            output_config: 输出配置，包含 gcs_destination
            target_language_codes: 目标语言代码列表
            source_language_code: 源语言代码
            models: 语言对应的模型映射
            glossaries: 语言对应的术语表映射
            
        Returns:
            操作名称，可用于跟踪批量翻译状态
        """
        try:
            request = translate.BatchTranslateTextRequest(
                parent=self.parent,
                source_language_code=source_language_code,
                target_language_codes=target_language_codes,
                input_configs=input_configs,
                output_config=output_config,
            )
            
            if models:
                request.models = models
            if glossaries:
                request.glossaries = glossaries
            
            operation = self.client.batch_translate_text(request=request)
            print(f"🚀 批量翻译已启动，操作名称: {operation.operation.name}")
            
            return operation.operation.name
            
        except exceptions.GoogleAPIError as e:
            print(f"❌ 批量翻译失败: {e}")
            return ""

    @timing_decorator
    def list_glossaries(self) -> List[Dict[str, Any]]:
        """
        列出项目中的术语表
        
        Returns:
            术语表列表
        """
        try:
            request = translate.ListGlossariesRequest(parent=self.parent)
            page_result = self.client.list_glossaries(request=request)
            
            glossaries = []
            for glossary in page_result:
                glossaries.append({
                    'name': glossary.name,
                    'language_codes': list(glossary.language_codes_set.language_codes) if glossary.language_codes_set else [],
                    'input_config': glossary.input_config,
                    'entry_count': glossary.entry_count
                })
            
            print(f"📚 找到 {len(glossaries)} 个术语表")
            return glossaries
            
        except exceptions.GoogleAPIError as e:
            print(f"❌ 获取术语表列表失败: {e}")
            return []


def demo_basic_translation():
    """演示基本翻译功能"""
    print("=" * 80)
    print("🚀 GCP Cloud Translation Advanced Edition 演示")
    print("=" * 80)
    
    # 初始化翻译客户端
    translator = CloudTranslationAdvanced(project_id="ali-icbu-gpu-project")
    
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
    
    translator.translate_text(demo_texts, target_language_code='zh-CN')
    
    # 3. 翻译为英文
    print("\n" + "=" * 50)
    print("🌐 3. 翻译为英文演示")
    print("=" * 50)
    
    translator.translate_text(demo_texts, target_language_code='en-US')
    
    # 4. 单个文本翻译
    print("\n" + "=" * 50)
    print("📝 4. 单个文本翻译演示")
    print("=" * 50)
    
    single_text = "人工智能正在改变我们的世界"
    result = translator.translate_single(single_text, target_language_code='en-US')
    print(f"原文: {single_text}")
    print(f"译文: {result}")


def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 80)
    print("🔧 高级功能演示")
    print("=" * 80)
    
    translator = CloudTranslationAdvanced(project_id="ali-icbu-gpu-project")
    
    # 1. 获取支持的语言
    print("\n📋 支持的语言列表 (前10个):")
    languages = translator.get_supported_languages()
    for lang in languages[:10]:
        support_info = []
        if lang['support_source']:
            support_info.append("源语言")
        if lang['support_target']:
            support_info.append("目标语言")
        support_str = ", ".join(support_info) if support_info else "不支持"
        print(f"  {lang['language_code']}: {lang['display_name']} ({support_str})")
    
    # 2. HTML格式翻译
    print("\n🌐 HTML格式翻译:")
    html_text = "<p>This is a <strong>bold</strong> text with <em>emphasis</em>.</p>"
    result = translator.translate_text([html_text], target_language_code='zh-CN', mime_type='text/html')
    if result:
        print(f"原文: {html_text}")
        print(f"译文: {result[0]['translated_text']}")
    
    # 3. 指定源语言翻译
    print("\n🎯 指定源语言翻译:")
    french_text = "Bonjour le monde"
    result = translator.translate_single(
        french_text, 
        target_language_code='zh-CN', 
        source_language_code='fr'
    )
    print(f"法语原文: {french_text}")
    print(f"中文译文: {result}")
    
    # 4. 列出术语表
    print("\n📚 术语表列表:")
    glossaries = translator.list_glossaries()
    if glossaries:
        for glossary in glossaries:
            print(f"  名称: {glossary['name']}")
            print(f"  语言: {', '.join(glossary['language_codes'])}")
            print(f"  条目数: {glossary['entry_count']}")
    else:
        print("  当前项目中没有术语表")


def demo_language_comparison():
    """演示多语言翻译对比"""
    print("\n" + "=" * 80)
    print("🌍 多语言翻译对比演示")
    print("=" * 80)
    
    translator = CloudTranslationAdvanced(project_id="ali-icbu-gpu-project")
    
    # 测试文本
    test_text = "Artificial intelligence is transforming our world in unprecedented ways."
    
    # 目标语言列表
    target_languages = [
        ('zh-CN', '简体中文'),
        ('zh-TW', '繁体中文'),
        ('ja', '日语'),
        ('ko', '韩语'),
        ('fr', '法语'),
        ('de', '德语'),
        ('es', '西班牙语'),
        ('ru', '俄语')
    ]
    
    print(f"\n📝 原文 (英语): {test_text}")
    print("\n🌐 翻译结果:")
    
    for lang_code, lang_name in target_languages:
        result = translator.translate_single(test_text, target_language_code=lang_code, source_language_code='en')
        if result:
            print(f"  {lang_name} ({lang_code}): {result}")


if __name__ == "__main__":
    try:
        # 基本功能演示
        demo_basic_translation()
        
        # 高级功能演示
        demo_advanced_features()
        
        # 多语言对比演示
        demo_language_comparison()
        
        print("\n" + "=" * 80)
        print("✅ 演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        sys.exit(1)
