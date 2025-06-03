"""
多图片编辑使用示例
"""

import sys
import os
import glob
from pathlib import Path

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_2_0_flash_multi_image_edit import edit_multiple_images, edit_multiple_images_once


def edit_specific_images():
    """
    示例1: 编辑指定的图片文件列表
    """
    print("=== 示例1: 编辑指定的图片文件 ===")
    
    # 指定要编辑的图片路径列表
    image_paths = [
        '9-model-input.png',
        # 添加更多图片路径
        # 'path/to/image2.jpg',
        # 'path/to/image3.png',
    ]
    
    # 编辑指令
    edit_instruction = """Angle of the Model: The model is shown from a side angle, facing left. This full side profile reveals the clean lines of the silhouette while allowing for a partial turn of the face toward the viewer-enough to suggest engagement without confronting directly. Pose & Posture: The posture is tall and composed, with the spine straight and the shoulders relaxed. The alignment is classical, nearly vertical, yet softened by a slight curve through the hip. The weight appears to rest evenly on both feet, suggesting stillness rather than movement. Hand Gesture: The visible hand, resting naturally alongside the thigh, is relaxed with fingers gently curled. There is no tension in the wrist or fingers, no sign of gesturing-merely the hand's quiet participation in the overall calm of the pose. Model-to-Image Ratio: The model fills approximately 90% of the vertical space, well-centered in the frame with a comfortable margin above the head and below the feet. This tight framing draws attention to the side profile, creating a portrait that feels architectural in its clarity, yet human in its subtle gesture. Keep the style, background, and clothing of the image"""
    
    # 执行编辑
    edited_paths = edit_multiple_images(
        image_paths=image_paths,
        edit_instruction=edit_instruction,
        output_dir="output_images",
        show_images=False  # 显示编辑后的图片
    )
    
    print(f"编辑完成! 输出文件: {edited_paths}")


def different_edit_instructions():
    """
    示例: 对不同图片使用不同的编辑指令
    """
    print("\n=== 示例: 对不同图片使用不同编辑指令 ===")
    
    # 图片和对应的编辑指令
    multiple_images_edit_pairs = [
        ('2-bag.png', '2-model.png', """Replace the model's bag without changing the model."""),
        ('3-shoe.png', '3-model.png', """Replace the model's shoes without changing the model."""),
    ]
    
    for i, (image_path1, image_path2, instruction) in enumerate(multiple_images_edit_pairs):
        if os.path.exists(image_path1) and os.path.exists(image_path2):
            print(f"处理图片 {i+1}: {image_path1} and {image_path2}")
            edited_paths = edit_multiple_images_once(
                image_paths=[image_path1, image_path2],
                edit_instruction=instruction,
                output_dir=f"custom_edit",
                show_images=False
            )
            print(f"完成: {edited_paths}")
        else:
            print(f"图片文件不存在: {image_path1} or {image_path2}")

    single_image_edit_pairs = [
        ('4-shoe.png', """cô gái trẻ việt nam 22t làn da trắng tự nhiên, tạo dáng với đôi dép này dưới nền nhà đẹp."""),
        ('5-palm.png', """can you enhance this image without changing the logo with white background part?"""),
        ('6-necklace.png', """A professional studio photo of a stylish woman with a neutral expression, wearing a 14K yellow gold Y-shaped lariat necklace with elongated paper clip chain links. The woman is dressed in a simple, elegant black V-neck top that highlights the necklace. She has medium skin tone and softly styled hair, with a soft neutral background and natural lighting that accentuates the shine and detail of the gold necklace. The focus is on the necklace as the centerpiece of the image, suitable for online jewelry store product listing."""),
        ('7-shirt.png', """A half-Length shot of a female indian model in a standing upright pose with white background, little closer shot and focus on tshirt, wearing a tshirt which i uploaded image shows -bottom wearing black jeans pant dress soft fabric. The model has straight, dark hair parted in the center, and she is wearing small white statement earrings. She is wearing casual white sneakers. Her hands are by her sides, and her posture is relaxed with a neutral smile facial expression. model should stand stylish posture"""),
        ('8-model.png', """Angle of the Model: The model is shown from a side angle, facing left. This full side profile reveals the clean lines of the silhouette while allowing for a partial turn of the face toward the viewer-enough to suggest engagement without confronting directly. Pose & Posture: The posture is tall and composed, with the spine straight and the shoulders relaxed. The alignment is classical, nearly vertical, yet softened by a slight curve through the hip. The weight appears to rest evenly on both feet, suggesting stillness rather than movement. Hand Gesture: The visible hand, resting naturally alongside the thigh, is relaxed with fingers gently curled. There is no tension in the wrist or fingers, no sign of gesturing-merely the hand's quiet participation in the overall calm of the pose. Model-to-Image Ratio: The model fills approximately 90% of the vertical space, well-centered in the frame with a comfortable margin above the head and below the feet. This tight framing draws attention to the side profile, creating a portrait that feels architectural in its clarity, yet human in its subtle gesture. Keep the style, background, and clothing of the image"""),
        ('9-model.png', """Angle of the Model: The model is shown from a side angle, facing left. This full side profile reveals the clean lines of the silhouette while allowing for a partial turn of the face toward the viewer-enough to suggest engagement without confronting directly. Pose & Posture: The posture is tall and composed, with the spine straight and the shoulders relaxed. The alignment is classical, nearly vertical, yet softened by a slight curve through the hip. The weight appears to rest evenly on both feet, suggesting stillness rather than movement. Hand Gesture: The visible hand, resting naturally alongside the thigh, is relaxed with fingers gently curled. There is no tension in the wrist or fingers, no sign of gesturing-merely the hand's quiet participation in the overall calm of the pose. Model-to-Image Ratio: The model fills approximately 90% of the vertical space, well-centered in the frame with a comfortable margin above the head and below the feet. This tight framing draws attention to the side profile, creating a portrait that feels architectural in its clarity, yet human in its subtle gesture. Keep the style, background, and clothing of the image"""),
    ]
    
    for i, (image_path, instruction) in enumerate(single_image_edit_pairs):
        if os.path.exists(image_path):
            print(f"处理图片 {i+1}: {image_path}")
            edited_paths = edit_multiple_images_once(
                image_paths=[image_path],
                edit_instruction=instruction,
                output_dir=f"custom_edit",
                show_images=False
            )
            print(f"完成: {edited_paths}")
        else:
            print(f"图片文件不存在: {image_path}")


def main():
    """
    运行示例
    """
    print("多图片编辑功能演示")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv('GEMINI_API_KEY'):
        print("错误: 请设置 GEMINI_API_KEY 环境变量")
        return
    
    try:        
        # # 运行示例
        different_edit_instructions()
        
        
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
