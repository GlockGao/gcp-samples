from PIL import Image
import os

def resize_image(input_image_path, output_image_path, size):
    """
    将指定图片调整为特定分辨率并保存。

    Args:
        input_image_path (str): 输入图片的路径。
        output_image_path (str): 输出图片的保存路径。
        size (tuple): 一个包含宽度和高度的元组, 例如 (1024, 1024)。
    """
    try:
        # 打开原始图片
        original_image = Image.open(input_image_path)
        print(f"成功打开图片: {input_image_path}")
        print(f"原始尺寸: {original_image.size}")

        # 将图片从低分辨率放大到高分辨率，使用高质量的重采样算法
        # Image.Resampling.LANCZOS 是目前 Pillow 中公认的、用于放大或缩小图像时质量最高的算法之一
        resized_image = original_image.resize(size, resample=Image.Resampling.LANCZOS)
        print(f"图片已调整至: {size}")

        # 保存调整大小后的图片
        resized_image.save(output_image_path)
        print(f"已成功保存至: {output_image_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_image_path}'。请确保文件路径和名称正确。")
    except Exception as e:
        print(f"处理图片时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 定义输入和输出文件名
    #    请将 'input_300x300.jpg' 替换为您自己的 300x300 图片文件名
    # input_filename = 'input_300x300.jpg'
    input_filename = 'input.png'
    
    #    定义您希望保存的新文件名
    output_filename = 'output_1024x1024.jpg'

    # 2. 定义目标分辨率
    target_size = (1024, 1024)

    # 3. 调用函数执行缩放
    resize_image(input_filename, output_filename, target_size)