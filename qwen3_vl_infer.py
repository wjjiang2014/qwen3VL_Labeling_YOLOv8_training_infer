import json
import os
import re
import logging
import copy
import base64
import numpy as np
import requests
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qwen3_vl_infer")

# 图像预处理参数
MIN_WIDTH, MIN_HEIGHT = 800, 600
MAX_EDGE = 1024

def preprocess_image(image: Image.Image) -> tuple:
    """
    图像预处理：放大过小的图像，缩小过大的图像

    Returns:
        tuple: (处理后的图像, 缩放比例scale)
        - scale = new_size / original_size，用于坐标转换
    """
    width, height = image.size
    scale = 1.0

    # 先处理最小尺寸限制（放大）
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        resize_scale = max(MIN_WIDTH / width, MIN_HEIGHT / height)
        new_width = int(width * resize_scale)
        new_height = int(height * resize_scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        scale *= resize_scale
        # logger.info(f"图像分辨率过低，已放大至: ({new_width}, {new_height}), scale={resize_scale:.4f}")

    # 再处理最大尺寸限制（缩小）
    width, height = image.size
    max_edge = max(width, height)
    if max_edge > MAX_EDGE:
        resize_scale = MAX_EDGE / max_edge
        new_width = int(width * resize_scale)
        new_height = int(height * resize_scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        scale *= resize_scale
        # logger.info(f"图像分辨率过高，已缩小至: ({new_width}, {new_height}), scale={resize_scale:.4f}")

    return image, scale

# 从config.json读取Ollama配置
_ollama_config = None

def get_ollama_config():
    """获取Ollama配置"""
    global _ollama_config
    if _ollama_config is None:
        with open("self_optim_config.json", encoding='utf-8') as f:
            config = json.load(f)
        _ollama_config = {
            "ollama_url": config.get("ollama_url", "http://192.168.159.179:11434/api/generate"),
            "model_name": config.get("qwen_model", "qwen3-vl:30b")
        }
    return _ollama_config

def encode_image_to_base64(image: Image.Image) -> str:
    """将PIL图像编码为base64字符串"""
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama_api(prompt: str, images: list, isdets: bool = True) -> tuple:
    """
    调用Ollama API进行推理

    Returns:
        tuple: (结果, show_path, crops, original_sizes, raw_output)
    """
    config = get_ollama_config()
    ollama_url = config["ollama_url"]
    model_name = config["model_name"]

    # 编码所有图像为base64
    images_base64 = [encode_image_to_base64(img) for img in images]

    # 构建请求载荷
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": images_base64,
        "stream": False,
        "options": {
            "temperature": 0.1,  # 降低随机性，0=完全确定，可选0.1-0.3
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "seed": 23
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        output_text = result.get('response', '')
    except Exception as e:
        logger.error(f"Ollama API调用失败: {e}")
        output_text = ""

    raw_output = output_text

    if isdets:
        # 解析检测结果
        dets = None
        json_text = output_text
        if "```json" in output_text:
            json_text = output_text.strip('```json\n').strip('```')
            raw_output = json_text
        try:
            dets = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")

        # 裁剪小图（基于第一张图）
        crops = []
        if dets:
            img_w, img_h = images[0].size
            for det in dets:
                bbox = np.array(det["bbox_2d"]).reshape((-1, 2))
                bbox_norm = bbox / 1000
                (xmin, ymin), (xmax, ymax) = bbox_norm * (img_w, img_h)
                crop_img = images[0].crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                crops.append(crop_img)

        return dets, None, crops, [(img.size[0], img.size[1]) for img in images], raw_output
    else:
        # 返回原始文本
        return output_text, None, None, [(img.size[0], img.size[1]) for img in images], raw_output

def infer(user_prompt: str, images: list, system_prompt: str = None, show: bool = False, crop: bool = False, isdets: bool = True) -> tuple:
    """
    推理接口 - 使用Ollama API调用Qwen3-VL模型

    Args:
        user_prompt: 提示词
        images: PIL图像列表，支持单图或多图
        system_prompt: 系统提示词（可选，暂未使用）
        show: 是否展示效果（绘制检测框并显示）
        crop: 是否裁剪检测区域小图
        isdets: 是否解析为检测结果，否则返回原始文本

    Returns:
        tuple (isdets=True):
            - dets: 检测结果列表，bbox_2d为归一化坐标[0,1]，解析失败时为None
            - crops: 裁剪小图列表
            - raw_output: 原始模型输出
        tuple (isdets=False):
            - text_result: 模型原始文本输出
            - raw_output: 原始模型输出
    """
    # 预处理：确保images是列表，并进行尺寸调整
    if isinstance(images, Image.Image):
        images = [images]

    # 保存原始图像副本（用于crop/show等需要原始图像的操作）
    original_images = [img.copy() for img in images]

    # 图像预处理（放大过小图像，缩小过大图像）
    processed_images = []
    scales = []
    original_sizes = []
    for img in images:
        original_sizes.append(img.size)  # 保存原始尺寸
        processed_img, scale = preprocess_image(img)
        processed_images.append(processed_img)
        scales.append(scale)
    images = processed_images

    # 保存预处理后的图像尺寸（用于API调用和坐标转换）
    preprocessed_sizes = [img.size for img in images]  # [(w,h), ...]

    # 调用Ollama API
    result, _, crops, _, raw_output = call_ollama_api(user_prompt, images, isdets)

    if isdets:
        dets = result

        # 坐标归一化：bbox_2d从[0,1000]转为[0,1]，基于预处理图像尺寸
        if dets:
            for det in dets:
                bbox = np.array(det["bbox_2d"]).reshape((-1, 2))
                bbox_norm = (bbox / 1000.0).tolist()
                det["bbox_2d"] = [coord for point in bbox_norm for coord in point]

        # 裁剪小图（基于预处理后的第一张图）
        crops = []
        if crop and dets:
            prep_w, prep_h = preprocessed_sizes[0]
            for det in dets:
                bbox = np.array(det["bbox_2d"]).reshape((-1, 2))
                (xmin, ymin), (xmax, ymax) = bbox * (prep_w, prep_h)
                crop_img = images[0].crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                crops.append(crop_img)

        # 展示效果（基于预处理后的第一张图）
        show_path = None
        if show:
            prep_w, prep_h = preprocessed_sizes[0]
            image_copy = copy.deepcopy(images[0])
            draw = ImageDraw.Draw(image_copy)
            if dets:
                for det in dets:
                    label = det["label"]
                    bbox = np.array(det["bbox_2d"]).reshape((-1, 2))
                    (xmin, ymin), (xmax, ymax) = bbox * (prep_w, prep_h)
                    draw.rectangle([int(xmin), int(ymin), int(xmax), int(ymax)], outline=(255, 0, 0), width=3)
                    draw.text((int(xmin), int(ymin) - 10), label, fill=(255, 0, 0))
            import uuid
            os.makedirs("tmp", exist_ok=True)
            show_path = f"tmp/qwen3vl_infer_{uuid.uuid4().hex[:8]}.jpg"
            image_copy.save(show_path)

        return dets, crops, raw_output
    else:
        return result, raw_output


if __name__ == "__main__":
    # 测试代码
    with open("self_optim_config.json", encoding='utf-8') as f:
        config = json.load(f)

    with open("prompt.json", "r") as f:
        orig_prompt = json.load(f)["draw_box"]

    class_names = config["class_name"]
    orig_prompt = orig_prompt.format(class_names=class_names)
    print(f"提示词: {orig_prompt}")

    src_path = "self-optim/data"
    results = []

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".json"):
                continue
            img_file = os.path.join(root, file)
            json_file = os.path.join(root, file.replace(".jpg", ".json").replace(".png", ".json"))

            if not os.path.exists(json_file):
                continue

            image = Image.open(img_file).convert("RGB")

            # 调用推理接口
            pred_dets, _, _ = infer(orig_prompt, image)

            # 读取真实标签
            with open(json_file, "r") as f:
                true_shapes = json.load(f)["shapes"]
            true_boxes = [(shape['label'], shape["points"]) for shape in true_shapes]

            # 记录结果
            result = {
                "image": file,
                "true_boxes": true_boxes,
                "pred_boxes": pred_dets if pred_dets else []
            }
            results.append(result)
            print(f"图片: {result['image']}")
            print(f"  真实框: {result['true_boxes']}")
            print(f"  预测框: {result['pred_boxes']}")
