"""
测试VL模型多图区域描述能力

用法:
python test_multi_image_region.py
自动从 self-optim/data 加载前两张图进行测试

或手动指定:
python test_multi_image_region.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg \
    --box1 0.1,0.2,0.5,0.6 --box2 0.3,0.4,0.7,0.8
"""
import json
import os
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed

seed = 5
set_seed(seed)

model_name = "Qwen3-VL-8B-Instruct"
max_memory = {
    0: "24576MiB",
    "cpu": "64GiB"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(model_name)

_model = None
_processor = None
_model_config = None

# 从self_optim_config.json读取配置
with open("self_optim_config.json", encoding='utf-8') as f:
    config = json.load(f)
class_name = config["class_name"]
src_path = "self-optim/data"


def load_all_images_data(src_path: str) -> list:
    """
    加载src_path下所有图像及其标注

    Returns:
        list: [{"image": PIL.Image, "filename": str, "true_boxes": [(label, points), ...]}, ...]
    """
    all_data = []
    for root, dirs, files in os.walk(src_path):
        for file in sorted(files):
            if file.endswith(".json"):
                continue
            img_file = os.path.join(root, file)
            json_file = os.path.join(root, file.replace(".jpg", ".json").replace(".png", ".json"))

            if not os.path.exists(json_file):
                continue

            image = Image.open(img_file).convert("RGB")
            with open(json_file, "r") as f:
                shapes = json.load(f)["shapes"]
            true_boxes = [(shape['label'], shape["points"]) for shape in shapes]

            all_data.append({
                "image": image,
                "filename": file,
                "true_boxes": true_boxes,
                "path": img_file
            })
            print(f"加载图像: {file}, 目标数量: {len(true_boxes)}")

    return all_data


def load_model():
    """加载Qwen3-VL模型"""
    global _model, _processor, _model_config
    if _model is None:
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            max_memory=max_memory
        )
        _processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        _model_config = {"max_new_tokens": 1024, "temperature": 0.1}
        logger.info("模型加载完成")
    return _model, _processor, _model_config


def generate(model, processor, inputs, model_config):
    """生成文本输出"""
    generated_ids = model.generate(**inputs, **model_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


def describe_regions(prompt: str, images: list) -> str:
    """
    用单prompt描述多张图中指定区域的内容

    Args:
        prompt: 提示词，需包含对各区域的描述指令
        images: PIL图像列表 [image1, image2, ...]

    Returns:
        str: 模型输出文本
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, processor, model_config = load_model()

    # 预处理图像
    min_width, min_height = 800, 600
    processed_images = []
    for image in images:
        width, height = image.size
        if width < min_width or height < min_height:
            scale = max(min_width / width, min_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"图像分辨率过低，已放大至: ({new_width}, {new_height})")
        processed_images.append(image)

    # 构建消息：图像在前，prompt在后
    user_content = [{"type": "text", "text": prompt}]
    for img in processed_images:
        user_content.insert(0, {"type": "image", "image": img})

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    output_text = generate(model, processor, inputs, model_config)
    logger.info(f"Raw output: {output_text}")

    return output_text


if __name__ == "__main__":

    # 加载图像数据
    print("加载图像数据...")
    all_data = load_all_images_data(src_path)
    print(f"共加载 {len(all_data)} 张图像\n")

    if len(all_data) < 2:
        print("需要至少2张图像进行测试！")
        exit(1)

    # 取前两张图
    img1_data = all_data[0]
    img2_data = all_data[1]

    img1 = img1_data["image"]
    img2 = img2_data["image"]

    # 如果没指定坐标，用第一张图第一个目标的坐标
    pts = img1_data["true_boxes"][0][1]
    box1 = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
    pts = img2_data["true_boxes"][0][1]
    box2 = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]

    # 默认提示词
    prompt = f"图1在[{box1[0]:.3f},{box1[1]:.3f},{box1[2]:.3f},{box1[3]:.3f}]坐标位置的内容是什么？\
    图2在[{box2[0]:.3f},{box2[1]:.3f},{box2[2]:.3f},{box2[3]:.3f}]坐标位置的内容是什么？\
    请分别描述图1和图2在各自指定位置看到的内容。"


    print(f"{'='*60}")
    print(f"测试图像1: {img1_data['filename']}")
    print(f"测试图像2: {img2_data['filename']}")
    print(f"图1坐标: {box1}")
    print(f"图2坐标: {box2}")
    print(f"{'='*60}")
    print(f"提示词:\n{prompt}")
    print(f"{'='*60}")

    output = describe_regions(prompt, [img1, img2])
    print(f"\n模型输出:\n{output}")
