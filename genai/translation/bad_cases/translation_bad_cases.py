#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译测试案例 - 复用ntm_v2.py和ntm_v3.py进行多个字符串翻译测试
测试字符串列表包含多种语言和特殊符号，用于全面测试翻译API的处理能力
源语言检测是“日语”，目标语言是“日语”，此时不会翻译，原样返回；"你好，[秘伝·鮮魚のごちそう]" -> "你好，[秘伝·鮮魚のごちそう]"
Workaround是随意指定源语言是“非日语”（例如英语），"你好，[秘伝·鮮魚のごちそう]" -> "こんにちは。[秘伝・鮮魚のごちそう]"
"""

import os
import sys
from typing import Dict, Any, Optional

# 添加父目录到路径以导入ntm_v2和ntm_v3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntm_v2 import CloudTranslationBasic
from ntm_v3 import CloudTranslationAdvanced


class TranslationBadCases:
    """翻译测试案例类"""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        初始化翻译测试客户端
        
        Args:
            project_id: GCP项目ID，如果不提供则从环境变量获取
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'ali-icbu-gpu-project')
        self.test_strings = [
            # 源语言检测是“日语”，目标语言是“日语”，此时不会翻译，原样返回；"你好，[秘伝·鮮魚のごちそう]" -> "你好，[秘伝·鮮魚のごちそう]"
            # Workaround是随意指定源语言是“非日语”（例如英语），"你好，[秘伝·鮮魚のごちそう]" -> "こんにちは。[秘伝・鮮魚のごちそう]"
            "你好，[秘伝·鮮魚のごちそう]", 
            # "Hello, [秘伝·鮮魚のごちそう]",
        ]
        
        # 初始化两个翻译客户端
        try:
            self.basic_translator = CloudTranslationBasic(project_id=self.project_id)
            self.advanced_translator = CloudTranslationAdvanced(project_id=self.project_id)
            print(f"✅ 成功初始化翻译测试客户端")
        except Exception as e:
            print(f"❌ 初始化翻译客户端失败: {e}")
            raise

    def detect_language_comparison(self) -> Dict[str, Any]:
        """
        使用两个版本的API检测语言并对比结果
        
        Returns:
            检测结果对比
        """
        print("\n" + "=" * 60)
        print("🔍 语言检测对比测试")
        print("=" * 60)
        
        all_results = {}
        
        for i, test_string in enumerate(self.test_strings, 1):
            print(f"\n📝 测试字符串 {i}: {test_string}")
            
            results = {}
            
            # Basic Edition (v2) 检测
            print("\n🔹 Basic Edition (v2) 检测结果:")
            basic_result = self.basic_translator.detect_language(test_string)
            results['basic'] = basic_result
            
            # Advanced Edition (v3) 检测
            print("\n🔹 Advanced Edition (v3) 检测结果:")
            advanced_result = self.advanced_translator.detect_language(test_string)
            results['advanced'] = advanced_result
            
            # 对比结果
            print("\n📊 检测结果对比:")
            if basic_result and advanced_result:
                print(f"  Basic (v2):    {basic_result.get('language', 'N/A')} (置信度: {basic_result.get('confidence', 0):.2f})")
                print(f"  Advanced (v3): {advanced_result.get('language_code', 'N/A')} (置信度: {advanced_result.get('confidence', 0):.2f})")
                
                # 检查是否一致
                basic_lang = basic_result.get('language', '')
                advanced_lang = advanced_result.get('language_code', '')
                if basic_lang == advanced_lang:
                    print("  ✅ 两个版本检测结果一致")
                else:
                    print("  ⚠️ 两个版本检测结果不一致")
            else:
                print("  ❌ 部分检测失败")
            
            all_results[f"string_{i}"] = {
                'text': test_string,
                'results': results
            }
        
        return all_results

    def translate_to_multiple_languages(self) -> Dict[str, Dict[str, Any]]:
        """
        使用两个版本的API翻译到多种语言并对比结果
        
        Returns:
            翻译结果对比
        """
        print("\n" + "=" * 60)
        print("🌐 多语言翻译对比测试")
        print("=" * 60)
        
        # 目标语言列表
        target_languages = [
            ('zh', 'zh-CN', '汉语'),
            ('en', 'en-US', '英语'),
            ('ja', 'ja', '日语'),
            # ('ko', 'ko', '韩语'),
            # ('fr', 'fr', '法语'),
            # ('de', 'de', '德语'),
            # ('es', 'es', '西班牙语')
        ]
        
        all_results = {}
        
        for i, test_string in enumerate(self.test_strings, 1):
            print(f"\n📝 测试字符串 {i}: {test_string}")
            
            string_results = {}
            
            for basic_lang, advanced_lang, lang_name in target_languages:
                print(f"\n🔸 翻译到{lang_name}:")
                
                # Basic Edition (v2) 翻译
                print(f"  🔹 Basic Edition (v2) → {basic_lang}:")
                basic_result = self.basic_translator.translate_single(
                    test_string, 
                    target_language=basic_lang,
                    source_language='fr'
                )
                
                # Advanced Edition (v3) 翻译
                print(f"  🔹 Advanced Edition (v3) → {advanced_lang}:")
                advanced_result = self.advanced_translator.translate_single(
                    test_string, 
                    target_language_code=advanced_lang
                )
                
                # 保存结果
                string_results[lang_name] = {
                    'basic': basic_result,
                    'advanced': advanced_result,
                    'basic_lang_code': basic_lang,
                    'advanced_lang_code': advanced_lang
                }
                
                # 对比结果
                print(f"  📊 翻译结果对比:")
                print(f"    Basic (v2):    {basic_result or 'N/A'}")
                print(f"    Advanced (v3): {advanced_result or 'N/A'}")
                
                if basic_result and advanced_result:
                    if basic_result.strip() == advanced_result.strip():
                        print(f"    ✅ 两个版本翻译结果一致")
                    else:
                        print(f"    ⚠️ 两个版本翻译结果不同")
                else:
                    print(f"    ❌ 部分翻译失败")
            
            all_results[f"string_{i}"] = {
                'text': test_string,
                'translations': string_results
            }
        
        return all_results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行综合测试
        
        Returns:
            所有测试结果
        """
        print("=" * 80)
        print("🚀 翻译测试案例 - 综合测试")
        print("=" * 80)
        print(f"🎯 测试字符串列表: {len(self.test_strings)} 个字符串")
        for i, test_string in enumerate(self.test_strings, 1):
            print(f"  {i}. {test_string}")
        print(f"📋 说明: 这些字符串包含多种语言和特殊符号，用于测试翻译API的处理能力")
        
        all_results = {}
        
        try:
            # 1. 语言检测对比
            detection_results = self.detect_language_comparison()
            all_results['language_detection'] = detection_results
            
            # 2. 多语言翻译对比
            translation_results = self.translate_to_multiple_languages()
            all_results['multi_language_translation'] = translation_results
            
            # 3. 生成测试报告
            self.generate_test_report(all_results)
            
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            all_results['error'] = str(e)
        
        return all_results

    def generate_test_report(self, results: Dict[str, Any]) -> None:
        """
        生成测试报告
        
        Args:
            results: 所有测试结果
        """
        print("\n" + "=" * 80)
        print("📊 测试报告总结")
        print("=" * 80)
        
        # 语言检测报告
        if 'language_detection' in results:
            detection = results['language_detection']
            print("\n🔍 语言检测结果:")
            detection_consistent_count = 0
            detection_total_count = 0
            
            for string_key, string_data in detection.items():
                test_string = string_data.get('text', '')
                detection_results = string_data.get('results', {})
                
                print(f"\n  📝 {test_string}:")
                if detection_results.get('basic') and detection_results.get('advanced'):
                    basic_lang = detection_results['basic'].get('language', 'N/A')
                    advanced_lang = detection_results['advanced'].get('language_code', 'N/A')
                    print(f"    Basic Edition:    {basic_lang}")
                    print(f"    Advanced Edition: {advanced_lang}")
                    
                    detection_total_count += 1
                    is_consistent = basic_lang == advanced_lang
                    if is_consistent:
                        detection_consistent_count += 1
                    
                    status = "✅ 一致" if is_consistent else "⚠️ 不一致"
                    print(f"    结果一致性: {status}")
                else:
                    print(f"    ❌ 检测失败")
            
            if detection_total_count > 0:
                detection_rate = (detection_consistent_count / detection_total_count) * 100
                print(f"\n  📈 语言检测一致性率: {detection_rate:.1f}% ({detection_consistent_count}/{detection_total_count})")
        
        # 翻译质量报告
        if 'multi_language_translation' in results:
            translation = results['multi_language_translation']
            print(f"\n🌐 翻译测试结果:")
            
            overall_consistent_count = 0
            overall_total_count = 0
            
            for string_key, string_data in translation.items():
                test_string = string_data.get('text', '')
                translations = string_data.get('translations', {})
                
                print(f"\n  📝 {test_string}:")
                string_consistent_count = 0
                string_total_count = 0
                
                for lang_name, lang_results in translations.items():
                    basic_result = lang_results.get('basic')
                    advanced_result = lang_results.get('advanced')
                    
                    if basic_result and advanced_result:
                        string_total_count += 1
                        overall_total_count += 1
                        is_consistent = basic_result.strip() == advanced_result.strip()
                        if is_consistent:
                            string_consistent_count += 1
                            overall_consistent_count += 1
                        
                        status = "✅ 一致" if is_consistent else "⚠️ 不同"
                        print(f"    {lang_name}: {status}")
                    else:
                        print(f"    {lang_name}: ❌ 翻译失败")
                
                if string_total_count > 0:
                    string_rate = (string_consistent_count / string_total_count) * 100
                    print(f"    📊 该字符串一致性率: {string_rate:.1f}% ({string_consistent_count}/{string_total_count})")
            
            if overall_total_count > 0:
                overall_rate = (overall_consistent_count / overall_total_count) * 100
                print(f"\n  📈 总体翻译一致性率: {overall_rate:.1f}% ({overall_consistent_count}/{overall_total_count})")
        
        print(f"\n✅ 测试完成！所有 {len(self.test_strings)} 个测试字符串已通过所有测试项目。")


def main():
    """主函数"""
    try:
        # 创建测试实例
        test_case = TranslationBadCases()
        
        # 运行综合测试
        results = test_case.run_comprehensive_test()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试已完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
