from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from typing import Union, List
import os
from utils.time_utils import timing_decorator
import PIL.Image
from pathlib import Path


gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"获取到的 GEMINI_API_KEY: {gemini_api_key}")
else:
    print("环境变量 'GEMINI_API_KEY' 未设置。")

client = genai.Client(api_key=gemini_api_key)


@timing_decorator
def edit_single_image(contents: Union[types.ContentListUnion, types.ContentListUnionDict],
                      output_path: str,
                      model: str = "gemini-2.0-flash-preview-image-generation"):
    """
    编辑单个图片
    
    Args:
        contents: 包含文本指令和图片的内容
        output_path: 输出图片的路径
        model: 使用的模型名称
    """
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"模型响应: {part.text}")
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save(output_path)
            print(f"编辑后的图片已保存到: {output_path}")
            return image
    
    return None


@timing_decorator
def edit_multiple_images(image_paths: List[str], 
                         edit_instruction: str,
                         output_dir: str = "edited_images",
                         model: str = "gemini-2.0-flash-preview-image-generation",
                         show_images: bool = False):
    """
    编辑多个图片
    
    Args:
        image_paths: 图片文件路径列表
        edit_instruction: 编辑指令
        output_dir: 输出目录
        model: 使用的模型名称
        show_images: 是否显示编辑后的图片
    
    Returns:
        List[str]: 编辑后图片的路径列表
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    edited_image_paths = []
    
    print(f"开始编辑 {len(image_paths)} 张图片...")
    print(f"编辑指令: {edit_instruction}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片文件不存在: {image_path}")
                continue
            
            print(f"正在处理第 {i}/{len(image_paths)} 张图片: {image_path}")
            
            # 加载图片
            image = PIL.Image.open(image_path)
            
            # 准备内容
            contents = [edit_instruction, image]
            
            # 生成输出文件名
            original_name = Path(image_path).stem
            output_filename = f"{original_name}_output_{i}.png"
            output_file_path = output_path / output_filename
            
            # 编辑图片
            edited_image = edit_single_image(
                contents=contents,
                output_path=str(output_file_path),
                model=model
            )
            
            if edited_image:
                edited_image_paths.append(str(output_file_path))
                
                # 可选择显示图片
                if show_images:
                    edited_image.show()
            
            print(f"第 {i} 张图片处理完成")
            print("-" * 30)
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            continue
    
    print(f"批量编辑完成! 成功处理了 {len(edited_image_paths)} 张图片")
    return edited_image_paths


@timing_decorator
def edit_multiple_images_once(image_paths: List[str], 
                              edit_instruction: str,
                              output_dir: str = "edited_images",
                              model: str = "gemini-2.0-flash-preview-image-generation",
                              show_images: bool = False):
    """
    编辑多个图片
    
    Args:
        image_paths: 图片文件路径列表
        edit_instruction: 编辑指令
        output_dir: 输出目录
        model: 使用的模型名称
        show_images: 是否显示编辑后的图片
    
    Returns:
        List[str]: 编辑后图片的路径列表
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    edited_image_paths = []
    
    print(f"开始编辑 {len(image_paths)} 张图片...")
    print(f"编辑指令: {edit_instruction}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    contents = list()

    contents.append(edit_instruction)
    
    try:
        for i, image_path in enumerate(image_paths, 1):
        
            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片文件不存在: {image_path}")
                continue
            
            print(f"正在处理第 {i}/{len(image_paths)} 张图片: {image_path}")
            
            # 加载图片
            image = PIL.Image.open(image_path)

            # 准备内容
            contents.append(image)

        print(f"准备内容: {contents}")
        
        # 生成输出文件名
        original_name = Path(image_path).stem
        output_filename = f"{original_name}_output.png"
        output_file_path = output_path / output_filename
        
        # 编辑图片
        edited_image = edit_single_image(
            contents=contents,
            output_path=str(output_file_path),
            model=model
        )
        
        if edited_image:
            edited_image_paths.append(str(output_file_path))
            
            # 可选择显示图片
            if show_images:
                edited_image.show()
        
        print(f"第 {i} 张图片处理完成")
        print("-" * 30)
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
    
    print(f"批量编辑完成! 成功处理了 {len(edited_image_paths)} 张图片")
    return edited_image_paths


def main():
    """
    主函数 - 演示多种使用方式
    """
    
    # 方式1: 编辑指定的多个图片文件
    print("=== 方式1: 编辑指定的多个图片文件 ===")
    image_paths = [
        'gemini-native-image.png',
        # 'image2.jpg',
        # 'image3.png'
    ]
    
    edit_instruction = "Can you add a cute llama next to the person in the image?"
    
    # 检查图片文件是否存在
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if existing_images:
        edited_paths = edit_multiple_images(
            image_paths=existing_images,
            edit_instruction=edit_instruction,
            output_dir="edited_images",
            show_images=False  # 设置为True可以显示编辑后的图片
        )
        
        print(f"编辑完成的图片路径: {edited_paths}")
    else:
        print("未找到可用的图片文件，请确保图片文件存在")


if __name__ == "__main__":
    main()
