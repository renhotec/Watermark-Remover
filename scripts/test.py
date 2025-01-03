import os
import torch
from PIL import Image
import numpy as np
from scripts.model import EnhancedGenerator as Generator
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F  # 新增导入
from PIL import ImageEnhance
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def pad_to_block_size(image, block_size=16):
    """
    将输入图像填充到能够被 block_size 整除的尺寸。
    :param image: 输入图像张量，形状为 [B, C, H, W]
    :param block_size: 块大小
    :return: 填充后的图像
    """
    _, _, h, w = image.shape
    new_h = (h + block_size - 1) // block_size * block_size  # 向上取整
    new_w = (w + block_size - 1) // block_size * block_size
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)  # 左右、上下
    image = F.pad(image, padding, mode="reflect")
    #print(f"Padded image from ({h}, {w}) to ({new_h}, {new_w})")  # 调试信息
    return image

def test_model(model_path, input_image_path="data/train/test", output_folder="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_path}, Input: {input_image_path}, Output: {output_folder}, Device: {device}")

    # 加载生成器模型
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    generator.eval()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 和训练时的标准化一致
    ])
    unnormalize = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2))  # 反归一化
    ])

    def get_generator_epoch_number(model_path):
        return int(model_path.split("_")[-1].split(".")[0])

    def post_process(image_np):
        """后处理：去噪、轻微锐化，不改变原始颜色"""
        image_np = (image_np * 255).astype(np.uint8)

        # 去噪
        denoised = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)

        # 锐化（保守调整）
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 避免对比度过高，保留原有亮度范围
        sharpened = np.clip(sharpened, 0, 255)

        return sharpened

    def process_image(filename, output_image_path):
        try:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return
            input_image = Image.open(filename).convert("RGB")
            original_size = input_image.size
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            # 调整输入尺寸
            input_tensor = pad_to_block_size(input_tensor, block_size=16)

            with torch.no_grad():
                output_tensor = generator(input_tensor)

            output_tensor = unnormalize(output_tensor.squeeze(0)).clamp(0, 1)
            output_image_np = output_tensor.permute(1, 2, 0).cpu().numpy()

            # 后处理步骤
            output_image_np = post_process(output_image_np)

            output_image = Image.fromarray(output_image_np.astype(np.uint8)).resize(original_size, Image.LANCZOS)

            if output_image_path != "":
                output_image_path = os.path.join(output_folder, output_image_path)
            else:
                output_image_path = os.path.join(output_folder, filename)
            enhancer = ImageEnhance.Sharpness(output_image)
            output_image = enhancer.enhance(2.0)  # 增强锐度（值可以调节）

            output_image.save(output_image_path, quality=100)
            # print(f"Processed {filename} and saved to {output_image_path}")
        except Exception as e:
            print(f"Failed to process {filename} to {output_image_path}: {e}")

    if input_image_path != "":
        # 如何 input_image_path 是文件路径，则直接处理该文件
        if os.path.isfile(input_image_path):
            output_image_path = f"{os.path.splitext(os.path.basename(input_image_path))[0]}_gen{get_generator_epoch_number(model_path)}{os.path.splitext(input_image_path)[1]}"
            process_image(input_image_path, output_image_path)
        elif os.path.isdir(input_image_path):
            model_gen = get_generator_epoch_number(model_path)
            start_time = time.time()  # Start time
            filenames = [f for f in os.listdir(input_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, os.path.join(input_image_path, filename), f"{os.path.splitext(filename)[0]}_gen{model_gen}{os.path.splitext(filename)[1]}"): filename for filename in filenames}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    future.result()

            end_time = time.time()  # End time
            print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    else:
        print("No image input")

    # start_time = time.time()  # Start time

    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(process_image, filename, ""): filename for filename in filenames}
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         future.result()

    # end_time = time.time()  # End time
    # print(f"Total elapsed time: {end_time - start_time:.2f} seconds")