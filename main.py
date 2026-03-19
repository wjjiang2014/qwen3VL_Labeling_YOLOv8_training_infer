#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 标注 + YOLOv8 训练推理完整工作流

功能流程:
    1. 读取类别配置，使用 Qwen3-VL 对图片进行目标检测标注
    2. 裁剪检测区域，使用 Qwen3-VL 重新验证类别是否正确
    3. 将标注转换为 YOLO 格式
    4. 使用 YOLOv8 进行训练
    5. 使用训练好的模型进行推理
    6. 计算 Qwen3-VL 标注与 YOLO 推理结果的 Precision/Recall/F1

使用方法:
    python main.py --config config.json --image-dir /path/to/images --epochs 100

作者: willj
日期: 2026-03-20
"""

import os
import json
import base64
import shutil
import time
from pathlib import Path
from datetime import datetime
import logging

# 第三方库
import cv2
import numpy as np
import requests
from PIL import Image
import torch
from ultralytics import YOLO
import labelme
from labelme import utils

# 配置日志
# 使用说明: 日志级别默认为 INFO，可改为 DEBUG 查看更详细信息
#   DEBUG: 详细日志，包含所有调试信息
#   INFO: 一般信息，如步骤进度、统计等
#   WARNING: 警告信息
#   ERROR: 错误信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Qwen3VLAnnotator:
    """
    Qwen3-VL 标注器
    
    负责与 Ollama 部署的 Qwen3-VL 模型交互，完成图像目标检测任务。
    
    主要功能:
        - 从配置文件加载类别列表
        - 构建目标检测 Prompt
        - 调用 Ollama API 进行图像识别
        - 将结果转换为 Labelme 格式 JSON
    
    属性:
        ollama_url (str): Ollama API 地址，默认为 http://localhost:11434/api/generate
        model_name (str): Qwen3-VL 模型名称，默认为 qwen2-vl:7b
    
    示例:
        >>> annotator = Qwen3VLAnnotator(
        ...     ollama_url="http://localhost:11434/api/generate",
        ...     model_name="qwen2-vl:7b"
        ... )
        >>> classes = annotator.load_classes("config.json")
        >>> result = annotator.call_qwen3vl("image.jpg", prompt)
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "qwen2-vl:7b"):
        """
        初始化 Qwen3-VL 标注器
        
        参数:
            ollama_url: Ollama API 地址
                - 本地默认: http://localhost:11434/api/generate
                - 远程示例: http://192.168.1.100:11434/api/generate
            model_name: 已下载的 Qwen3-VL 模型名称
                - 可用: qwen2-vl:7b, qwen2-vl:4b, qwen2.5-vl:7b 等
                - 需提前通过 ollama pull <model> 下载
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        logger.info(f"Qwen3-VL 标注器初始化完成，模型: {self.model_name}")
    
    def load_classes(self, config_path: str) -> list:
        """
        从 JSON 配置文件加载类别名称列表
        
        支持三种配置格式:
            1. 简单数组: ["person", "car", "dog"]
            2. classes 键: {"classes": ["person", "car"]}
            3. names 键: {"names": {"0": "person", "1": "car"}}
        
        参数:
            config_path: 配置文件路径 (JSON 格式)
        
        返回:
            list: 类别名称列表，如 ["person", "car", "dog"]
        
        异常:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置格式不支持
        
        示例:
            >>> classes = annotator.load_classes("config.json")
            >>> print(classes)
            ['person', 'car', 'dog']
        """
        logger.info(f"加载类别配置: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 兼容多种配置格式
        if isinstance(config, dict):
            if 'classes' in config:
                classes = config['classes']
            elif 'names' in config:
                # names 可以是列表或字典
                names = config['names']
                if isinstance(names, list):
                    classes = names
                else:
                    classes = list(names.values())
            else:
                # 假设所有键都是类别名
                classes = list(config.keys())
        elif isinstance(config, list):
            classes = config
        else:
            raise ValueError(f"不支持的配置格式: {type(config)}")
        
        logger.info(f"加载到 {len(classes)} 个类别: {classes}")
        return classes
    
    def build_prompt(self, classes: list) -> str:
        """
        构建目标检测提示词
        
        Prompt 包含:
            - 类别列表
            - 输出格式要求 (JSON)
            - 坐标归一化说明
        
        参数:
            classes: 类别名称列表
        
        返回:
            str: 完整的提示词
        
        示例:
            >>> prompt = annotator.build_prompt(["person", "car"])
            >>> print(prompt[:100])
            请识别图片中的对象，从以下类别中选择: ["person", "car"]...
        """
        # 将类别列表转为带引号的字符串
        class_str = ", ".join([f'"{c}"' for c in classes])
        
        prompt = f"""请识别图片中的对象，从以下类别中选择: [{class_str}]。
请以JSON格式返回结果，格式如下：
{{
    "objects": [
        {{"class_name": "类别名", "bbox": [x1, y1, x2, y2]}}
    ]
}}
其中bbox是边界框的左上角和右下角坐标（归一化到0-1）。
只返回JSON，不要其他内容。"""
        
        return prompt
    
    def encode_image(self, image_path: str) -> str:
        """
        将图片文件编码为 base64 字符串
        
        用于 Ollama API 请求中传递图片数据。
        
        参数:
            image_path: 图片文件路径
        
        返回:
            str: base64 编码的字符串
        
        异常:
            FileNotFoundError: 图片文件不存在
        """
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def call_qwen3vl(self, image_path: str, prompt: str) -> dict:
        """
        调用 Ollama API 进行图像目标检测
        
        向 Qwen3-VL 模型发送请求，解析返回的 JSON 结果。
        
        参数:
            image_path: 输入图片路径
            prompt: 目标检测提示词 (由 build_prompt 生成)
        
        返回:
            dict: 检测结果，格式为
                {
                    "objects": [
                        {"class_name": "类别名", "bbox": [x1, y1, x2, y2]},
                        ...
                    ]
                }
                - bbox 坐标已归一化到 0-1 范围
        
        异常:
            requests.exceptions.RequestException: API 调用失败
            json.JSONDecodeError: 返回结果解析失败
        
        注意:
            - 超时时间设置为 120 秒 (大图可能较慢)
            - 解析失败时返回空 objects 列表
        """
        # 编码图片为 base64
        image_base64 = self.encode_image(image_path)
        
        # 构建请求载荷
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False  # 关闭流式输出
        }
        
        try:
            # 发送 POST 请求到 Ollama API
            # 超时时间 120 秒 (大图识别可能需要更长时间)
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()  # 检查 HTTP 错误
            
            result = response.json()
            text = result.get('response', '')
            
            # 尝试解析返回的 JSON
            try:
                # 方法1: 直接解析 (最理想情况)
                return json.loads(text)
            except json.JSONDecodeError:
                # 方法2: 尝试提取 ```json ... ``` 块
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # 无法解析，返回空结果
                    logger.warning(f"无法解析Qwen3-VL返回: {text[:200]}...")
                    return {"objects": []}
                    
        except requests.exceptions.Timeout:
            logger.error(f"Qwen3-VL API 超时: {image_path}")
            return {"objects": []}
        except requests.exceptions.RequestException as e:
            logger.error(f"调用Qwen3-VL失败: {e}")
            return {"objects": []}
    
    def denormalize_bbox(self, bbox: list, width: int, height: int) -> list:
        """
        反归一化边界框坐标
        
        将归一化坐标 (0-1) 转换为像素坐标。
        
        参数:
            bbox: 归一化边界框 [x1, y1, x2, y2]，范围 0-1
            width: 图片宽度 (像素)
            height: 图片高度 (像素)
        
        返回:
            list: 像素坐标 [x1, y1, x2, y2]
        
        示例:
            >>> denormalize_bbox([0.1, 0.2, 0.5, 0.8], 640, 480)
            [64, 96, 320, 384]
        """
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * width),   # x1 像素坐标
            int(y1 * height),  # y1 像素坐标
            int(x2 * width),   # x2 像素坐标
            int(y2 * height)   # y2 像素坐标
        ]
    
    def convert_to_labelme(self, qwen_result: dict, image_path: str, 
                          output_path: str, classes: list) -> bool:
        """
        将 Qwen3-VL 检测结果转换为 Labelme 格式 JSON
        
        Labelme 是一种常用的图像标注格式，支持矩形、多边形等多种形状。
        转换后的 JSON 可用于 labelme 工具查看和编辑。
        
        参数:
            qwen_result: Qwen3-VL 返回的检测结果
            image_path: 对应的图片路径
            output_path: 输出的 Labelme JSON 文件路径
            classes: 类别列表 (用于过滤)
        
        返回:
            bool: 转换成功返回 True，否则返回 False
        
        Labelme JSON 格式:
            {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                    {
                        "label": "person",
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle"
                    }
                ],
                "imagePath": "image.jpg",
                "imageHeight": 480,
                "imageWidth": 640
            }
        """
        try:
            # 读取图片获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return False
            
            height, width = img.shape[:2]
            
            # 构建 shapes 列表
            shapes = []
            for obj in qwen_result.get('objects', []):
                class_name = obj.get('class_name', '')
                
                # 过滤不在类别列表中的检测结果
                if class_name not in classes:
                    continue
                
                bbox = obj.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # 反归一化坐标
                x1, y1, x2, y2 = self.denormalize_bbox(bbox, width, height)
                
                shapes.append({
                    "label": class_name,
                    "points": [[x1, y1], [x2, y2]],  # 左上角、右下角
                    "shape_type": "rectangle",
                    "flags": {}
                })
            
            # 没有有效检测结果
            if not shapes:
                logger.warning(f"图片无有效检测结果: {image_path}")
                return False
            
            # 构建 Labelme JSON
            labelme_json = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": os.path.basename(image_path),
                "imageData": None,  # 不存储图片数据，只存储路径
                "imageHeight": height,
                "imageWidth": width
            }
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_json, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"转换成功: {output_path}, 检测到 {len(shapes)} 个目标")
            return True
            
        except Exception as e:
            logger.error(f"转换Labelme格式失败: {e}")
            return False


class LabelVerifier:
    """
    标签验证器
    
    对初步标注进行二次验证，提高标注质量。
    流程: 裁剪检测区域 -> 调用 Qwen3-VL 重新识别 -> 比较类别
    
    主要功能:
        - 裁剪边界框区域图片
        - 使用 Qwen3-VL 重新验证类别
        - 修正错误或删除无效标注
    
    示例:
        >>> verifier = LabelVerifier(annotator)
        >>> result = verifier.verify_and_crop("image.jpg", "label.json", classes, "crop_dir")
    """
    
    def __init__(self, annotator: Qwen3VLAnnotator):
        """
        初始化标签验证器
        
        参数:
            annotator: Qwen3VLAnnotator 实例，用于调用 Qwen3-VL API
        """
        self.annotator = annotator
    
    def verify_and_crop(self, image_path: str, label_path: str, 
                       classes: list, output_crop_dir: str) -> bool:
        """
        验证标签并裁剪区域
        
        验证流程:
            1. 读取 Labelme JSON 标注文件
            2. 对每个边界框:
                a. 裁剪图片区域并保存
                b. 调用 Qwen3-VL 重新识别
                c. 比较识别结果与标注类别
            3. 只保留验证通过的标注
        
        参数:
            image_path: 原始图片路径
            label_path: Labelme JSON 标注文件路径
            classes: 类别列表
            output_crop_dir: 裁剪图片输出目录
        
        返回:
            bool: 验证通过返回 True，否则返回 False (标注被删除)
        
        验证逻辑:
            - 类别正确: 保留
            - 类别错误但识别到正确类别: 修正类别并保留
            - 无法识别或识别为非目标类别: 删除
        """
        try:
            # 1. 读取标签
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            valid_shapes = []  # 验证通过的 shapes
            
            # 2. 遍历每个标注
            for shape in label_data.get('shapes', []):
                label = shape.get('label', '')
                points = shape.get('points', [])
                
                if len(points) != 2:
                    continue
                
                # 获取边界框坐标
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                
                # 边界检查，防止越界
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                y1, y2 = max(0, y1), min(img.shape[0], y1)
                
                # 过滤无效框
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 3. 裁剪区域
                crop_img = img[y1:y2, x1:x2]
                
                # 保存裁剪图 (可选，用于调试)
                crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}.jpg"
                crop_path = os.path.join(output_crop_dir, crop_filename)
                cv2.imwrite(crop_path, crop_img)
                
                # 4. 重新验证 - 构建验证 prompt
                # 简化验证：直接问是否为指定类别
                prompt = f"""请识别这张图片中的对象，从以下类别中选择: {classes}。
如果图片中包含{label}类别的物体，请返回JSON格式：
{{"correct": true}}
否则返回：
{{"correct": false, "actual_class": "实际类别"}}
只返回JSON。"""
                
                # 调用 Qwen3-VL
                result = self.annotator.call_qwen3vl(crop_path, prompt)
                
                # 5. 分析验证结果
                is_correct = result.get('correct', False)
                
                if not is_correct and label in classes:
                    # 类别错误，检查实际类别
                    actual = result.get('actual_class', '')
                    if actual and actual in classes:
                        # 修正类别
                        shape['label'] = actual
                        valid_shapes.append(shape)
                        logger.info(f"类别修正: {label} -> {actual}")
                    else:
                        # 无法确认类别，删除标注
                        logger.warning(f"标签验证失败，将删除: {label_path}")
                        return False
                else:
                    # 验证通过
                    valid_shapes.append(shape)
            
            # 6. 保存验证后的标注
            if valid_shapes:
                label_data['shapes'] = valid_shapes
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                return True
            else:
                # 所有标注都被删除
                return False
                
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False


class YOLOTrainer:
    """
    YOLOv8 训练与推理器
    
    负责 YOLO 格式转换、模型训练和推理。
    
    主要功能:
        - Labelme 格式 -> YOLO 格式转换
        - YOLOv8 模型训练
        - 模型推理与结果输出
    
    属性:
        model: ultralytics YOLO 模型实例
    
    YOLO 格式说明:
        每个图片对应一个 .txt 文件，每行一个目标:
        class_id x_center y_center width height
        - 坐标已归一化到 0-1 范围
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        初始化 YOLOv8 训练器
        
        参数:
            model_path: 预训练模型路径或模型名称
                - 本地: /path/to/yolov8n.pt
                - 远程: yolov8n.pt (自动下载)
                - 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        
        注意:
            - 首次使用会自动从 ultralytics 下载模型
            - GPU 可用时会自动使用 CUDA 加速
        """
        self.model = YOLO(model_path)
        logger.info(f"YOLO 模型加载: {model_path}")
        
        # 检查 GPU 可用性
        if torch.cuda.is_available():
            logger.info(f"GPU 可用: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("使用 CPU 进行训练")
    
    def convert_labelme_to_yolo(self, labelme_dir: str, output_dir: str, 
                                 image_dir: str, classes: list):
        """
        将 Labelme 格式标注转换为 YOLO 格式
        
        转换流程:
            1. 遍历所有 Labelme JSON 文件
            2. 读取图片获取尺寸
            3. 转换坐标格式:
               - Labelme: [x1, y1, x2, y2] (像素，绝对坐标)
               - YOLO: [x_center, y_center, w, h] (归一化，相对坐标)
            4. 生成 YOLO 格式 txt 文件
        
        参数:
            labelme_dir: Labelme JSON 文件目录
            output_dir: YOLO 数据集输出目录
            image_dir: 原始图片目录
            classes: 类别列表
        
        输出目录结构:
            output_dir/
            ├── images/train/   # 图片副本
            ├── labels/train/   # YOLO 标注文件
            └── data.yaml       # YOLO 数据配置
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        
        labelme_files = list(Path(labelme_dir).glob('*.json'))
        logger.info(f"开始转换 {len(labelme_files)} 个 Labelme 文件...")
        
        for labelme_file in labelme_files:
            # 读取 labelme JSON
            with open(labelme_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取对应图片
            img_name = data.get('imagePath', labelme_file.stem + '.jpg')
            img_path = os.path.join(image_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.warning(f"图片不存在: {img_path}")
                continue
            
            # 复制图片到 YOLO 目录
            dest_img = os.path.join(output_dir, 'images', 'train', img_name)
            if not os.path.exists(dest_img):
                shutil.copy(img_path, dest_img)
            
            # 读取图片获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            # 转换标注
            yolo_lines = []
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                
                # 过滤无效类别
                if label not in classes:
                    continue
                
                class_id = classes.index(label)
                points = shape.get('points', [])
                
                if len(points) != 2:
                    continue
                
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 转换为 YOLO 格式 (中心点 + 宽高，归一化)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}\n")
            
            # 写入 YOLO 标注文件
            yolo_filename = os.path.splitext(img_name)[0] + '.txt'
            yolo_file = os.path.join(output_dir, 'labels', 'train', yolo_filename)
            with open(yolo_file, 'w') as f:
                f.writelines(yolo_lines)
        
        # 创建 data.yaml (YOLO 训练需要)
        yaml_content = f"""# YOLO 数据集配置
# 由 Qwen3VL 自动生成

# 训练数据路径 (相对于此文件)
train: images/train

# 验证数据路径 (此处与训练相同，实际可分开)
val: images/train

# 类别数量
nc: {len(classes)}

# 类别名称
names: {classes}
"""
        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"转换完成: {len(labelme_files)} 个文件 -> YOLO格式")
        logger.info(f"输出目录: {output_dir}")
    
    def train(self, data_yaml: str, epochs: int = 100, 
              imgsz: int = 640) -> str:
        """
        训练 YOLOv8 模型
        
        参数:
            data_yaml: YOLO 数据集配置文件路径
            epochs: 训练轮数，默认 100
                - 小数据集: 50-100
                - 中等数据集: 100-200
                - 大数据集: 200-500
            imgsz: 输入图片尺寸，默认 640
                - 可选: 320, 416, 640, 1280
                - 越小越快，但精度可能降低
        
        返回:
            str: 最佳模型路径 (best.pt)
        
        训练特性:
            - 早停: patience=20 (连续 20 轮无提升则停止)
            - 保存最佳: 自动保存 best.pt
            - 日志: 保存训练过程到 runs/detect/train/
        
        注意:
            - GPU 训练会快很多
            - 可在训练过程中查看 runs/detect/train/results.png
        """
        logger.info(f"开始训练 YOLOv8, epochs={epochs}, imgsz={imgsz}")
        
        results = self.model.train(
            data=data_yaml,          # 数据集配置
            epochs=epochs,           # 训练轮数
            imgsz=imgsz,             # 输入尺寸
            project='runs/detect',   # 输出目录
            name='train',            # 实验名称
            exist_ok=True,           # 覆盖已有结果
            patience=20,             # 早停耐心值
            save=True,              # 保存模型
            plots=True,             # 生成图表
            # 可选: 进一步优化参数
            # batch=16,              # batch size
            # optimizer='SGD',       # 优化器
            # lr0=0.01,              # 初始学习率
        )
        
        # 返回最佳模型路径
        best_model = os.path.join('runs/detect', 'train', 'weights', 'best.pt')
        logger.info(f"训练完成，模型保存于: {best_model}")
        
        return best_model
    
    def infer(self, model_path: str, image_dir: str, 
              output_dir: str, classes: list) -> str:
        """
        使用训练好的模型进行推理
        
        参数:
            model_path: 模型文件路径 (best.pt)
            image_dir: 待推理图片目录
            output_dir: 输出目录
            classes: 类别列表
        
        返回:
            str: 推理结果 JSON 文件路径
        
        输出 JSON 格式:
            [
                {
                    "image": "/path/to/image1.jpg",
                    "detections": [
                        {
                            "class_name": "person",
                            "confidence": 0.95,
                            "bbox": [x1, y1, x2, y2]  # 像素坐标
                        }
                    ]
                }
            ]
        """
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有 jpg 图片
        image_files = list(Path(image_dir).glob('*.jpg'))
        logger.info(f"开始推理 {len(image_files)} 张图片...")
        
        results_json = []
        
        for img_path in image_files:
            # 推理
            result = model(img_path, verbose=False)
            
            # 提取检测结果
            boxes = result[0].boxes
            detections = []
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])      # 类别 ID
                    conf = float(box.conf[0])       # 置信度
                    xyxy = box.xyxy[0].tolist()     # 边界框 (像素坐标)
                    
                    detections.append({
                        "class_name": classes[class_id],
                        "confidence": round(conf, 4),
                        "bbox": [int(x) for x in xyxy]
                    })
            
            results_json.append({
                "image": str(img_path),
                "detections": detections
            })
        
        # 保存推理结果
        output_file = os.path.join(output_dir, 'inference_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"推理完成: {len(image_files)} 张图片")
        logger.info(f"结果保存于: {output_file}")
        
        return output_file


class Evaluator:
    """
    评估器
    
    计算 YOLO 推理结果与 Qwen3-VL 标注之间的差异，
    使用 Precision (精确率)、Recall (召回率)、F1 分数作为指标。
    
    指标说明:
        - Precision: YOLO 正确检测数 / YOLO 总检测数
        - Recall: YOLO 正确检测数 / Qwen3-VL 标注总数
        - F1: Precision 和 Recall 的调和平均
    
    IOU (Intersection over Union):
        两个边界框的交并比，用于判断是否为同一目标。
        IOU >= threshold (默认 0.5) 视为匹配成功。
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def calculate_iou(self, box1: list, box2: list) -> float:
        """
        计算两个边界框的 IOU (Intersection over Union)
        
        IOU = 交集面积 / 并集面积
        
        参数:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
        
        返回:
            float: IOU 值，范围 [0, 1]
                - 0: 完全不重叠
                - 1: 完全重叠
        
        示例:
            >>> calculate_iou([0, 0, 100, 100], [50, 50, 150, 150])
            0.1428...
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集区域
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        # 无交集
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        # 交集面积
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # 并集面积 = 面积1 + 面积2 - 交集面积
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_boxes(self, pred_boxes: list, gt_boxes: list, 
                    iou_threshold: float = 0.5) -> tuple:
        """
        匹配预测框与真实框，计算 TP/FP/FN
        
        匹配规则:
            1. 对每个预测框，找 IOU 最大的真实框
            2. IOU >= threshold 且类别相同 -> TP (True Positive)
            3. 否则 -> FP (False Positive)
            4. 未匹配的真实框 -> FN (False Negative)
        
        参数:
            pred_boxes: YOLO 预测框列表
                [{"class_name": "person", "bbox": [x1,y1,x2,y2]}, ...]
            gt_boxes: Qwen3-VL 标注框列表
                [{"class_name": "person", "bbox": [x1,y1,x2,y2]}, ...]
            iou_threshold: IOU 阈值，默认 0.5
        
        返回:
            tuple: (TP, FP, FN)
                - TP: 正确检测 (预测正确且匹配)
                - FP: 误检 (预测错误或无法匹配)
                - FN: 漏检 (未被预测到的真实框)
        """
        matched_gt = set()  # 已匹配的真实框索引
        tp = 0              # True Positive
        fp = 0              # False Positive
        
        # 遍历每个预测框
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            # 找最佳匹配的真实框
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue  # 已匹配，跳过
                
                # 计算 IOU
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # 判断是否为有效匹配
            if (best_iou >= iou_threshold and 
                pred['class_name'] == gt_boxes[best_gt_idx]['class_name']):
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        # 未匹配的真实框为 FN
        fn = len(gt_boxes) - len(matched_gt)
        
        return tp, fp, fn
    
    def evaluate(self, qwen_labels_path: str, yolo_results_path: str,
                 iou_threshold: float = 0.5) -> dict:
        """
        评估 YOLO 推理结果与 Qwen3-VL 标注的差异
        
        参数:
            qwen_labels_path: Qwen3-VL 标注结果 JSON 文件路径
                格式: [{"image": "path", "detections": [...]}, ...]
            yolo_results_path: YOLO 推理结果 JSON 文件路径
                格式: [{"image": "path", "detections": [...]}, ...]
            iou_threshold: IOU 阈值，默认 0.5
        
        返回:
            dict: 评估指标
                {
                    "precision": 0.85,
                    "recall": 0.78,
                    "f1": 0.8135,
                    "total_tp": 85,
                    "total_fp": 15,
                    "total_fn": 24
                }
        
        计算公式:
            Precision = TP / (TP + FP)
            Recall    = TP / (TP + FN)
            F1        = 2 * P * R / (P + R)
        """
        
        # 加载 Qwen3-VL 标注
        with open(qwen_labels_path, 'r', encoding='utf-8') as f:
            qwen_data = json.load(f)
        
        # 加载 YOLO 推理结果
        with open(yolo_results_path, 'r', encoding='utf-8') as f:
            yolo_data = json.load(f)
        
        # 构建图片名 -> 检测结果 映射
        yolo_dict = {}
        for item in yolo_data:
            img_name = os.path.basename(item['image'])
            yolo_dict[img_name] = item['detections']
        
        # 统计
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # 逐图匹配
        for qwen_item in qwen_data:
            img_name = os.path.basename(qwen_item['image'])
            
            pred_boxes = yolo_dict.get(img_name, [])
            gt_boxes = qwen_item['detections']
            
            # 跳过空标注
            if not gt_boxes:
                continue
            
            # 匹配计算
            tp, fp, fn = self.match_boxes(pred_boxes, gt_boxes, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # 计算指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        }


class Pipeline:
    """
    完整工作流管道
    
    整合所有模块，执行端到端的标注、训练、推理、评估流程。
    
    工作流程:
        1. Qwen3-VL 标注 (labelme_output/)
        2. 标签验证 (verified_output/)
        3. YOLO 数据转换 (yolo_dataset/)
        4. 模型训练 (runs/detect/train/)
        5. 模型推理 (inference_output/)
        6. 结果评估 (输出 Precision/Recall/F1)
    
    特性:
        - 自动创建带时间戳的输出目录，避免覆盖
        - 每个步骤独立，可单独运行
        - 详细日志记录进度
    """
    
    def __init__(self, config: dict):
        """
        初始化工作流管道
        
        参数:
            config: 配置字典，应包含:
                - classes: 类别列表
                - ollama_url: Ollama API 地址
                - qwen_model: Qwen3-VL 模型名
                - yolo_model: YOLO 模型名
                - base_dir: 输出根目录
        """
        self.config = config
        self.base_dir = config.get('base_dir', '.')
        
        # 创建带时间戳的输出目录
        # 格式: labelme_output_20260320_123456
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.labelme_dir = os.path.join(self.base_dir, f"labelme_output_{timestamp}")
        self.verified_dir = os.path.join(self.base_dir, f"verified_output_{timestamp}")
        self.crop_dir = os.path.join(self.base_dir, f"crop_verify_{timestamp}")
        self.yolo_dir = os.path.join(self.base_dir, f"yolo_dataset_{timestamp}")
        self.infer_dir = os.path.join(self.base_dir, f"inference_output_{timestamp}")
        
        # 确保目录存在
        for d in [self.labelme_dir, self.verified_dir, self.crop_dir, 
                  self.yolo_dir, self.infer_dir]:
            os.makedirs(d, exist_ok=True)
            logger.debug(f"创建目录: {d}")
        
        # 初始化各组件
        self.annotator = Qwen3VLAnnotator(
            ollama_url=config.get('ollama_url', 'http://localhost:11434/api/generate'),
            model_name=config.get('qwen_model', 'qwen2-vl:7b')
        )
        
        self.verifier = LabelVerifier(self.annotator)
        self.trainer = YOLOTrainer(model_path=config.get('yolo_model', 'yolov8n.pt'))
        self.evaluator = Evaluator()
        
        # 加载类别
        self.classes = self.annotator.load_classes(config['class_config'])
        logger.info(f"=" * 50)
        logger.info(f"工作流初始化完成")
        logger.info(f"类别: {self.classes}")
        logger.info(f"输出目录: {self.base_dir}")
        logger.info(f"=" * 50)
    
    def step1_annotate(self, image_dir: str) -> str:
        """
        步骤1: 使用 Qwen3-VL 进行图像标注
        
        遍历图片目录中的所有 JPG 文件，调用 Qwen3-VL 进行目标检测，
        生成 Labelme 格式的标注文件。
        
        参数:
            image_dir: 图片目录路径
        
        返回:
            str: 标注输出目录路径
        
        输出:
            labelme_output_YYYYMMDD_HHMMSS/*.json
        """
        logger.info("=" * 50)
        logger.info("步骤1: Qwen3-VL 标注")
        logger.info("=" * 50)
        
        # 构建检测 Prompt
        prompt = self.annotator.build_prompt(self.classes)
        
        # 获取所有 JPG 图片
        image_files = list(Path(image_dir).glob('*.jpg'))
        logger.info(f"找到 {len(image_files)} 张图片")
        
        success_count = 0
        for img_path in image_files:
            # 输出标注文件路径
            label_path = os.path.join(self.labelme_dir, f"{img_path.stem}.json")
            
            # 调用 Qwen3-VL 进行检测
            result = self.annotator.call_qwen3vl(str(img_path), prompt)
            
            # 转换为 Labelme 格式
            if self.annotator.convert_to_labelme(result, str(img_path), 
                                                   label_path, self.classes):
                success_count += 1
        
        logger.info(f"标注完成: {success_count}/{len(image_files)} 成功")
        
        # 返回标注目录供下一步使用
        return self.labelme_dir
    
    def step2_verify(self, image_dir: str) -> str:
        """
        步骤2: 验证标注并裁剪
        
        对每个标注的边界框进行裁剪，使用 Qwen3-VL 重新验证类别。
        修正或删除错误的标注。
        
        参数:
            image_dir: 原始图片目录
        
        返回:
            str: 验证后标注的输出目录
        
        输出:
            verified_output_YYYYMMDD_HHMMSS/*.json
            crop_verify_YYYYMMDD_HHMMSS/*.jpg (裁剪图片)
        """
        logger.info("=" * 50)
        logger.info("步骤2: 标签验证")
        logger.info("=" * 50)
        
        # 获取所有标注文件
        label_files = list(Path(self.labelme_dir).glob('*.json'))
        logger.info(f"验证 {len(label_files)} 个标注文件...")
        
        success_count = 0
        for label_path in label_files:
            # 对应图片路径
            img_name = label_path.stem + '.jpg'
            img_path = os.path.join(image_dir, img_name)
            
            if os.path.exists(img_path):
                # 验证并裁剪
                if self.verifier.verify_and_crop(
                    img_path, str(label_path), self.classes, self.crop_dir
                ):
                    # 验证通过，复制到验证目录
                    shutil.copy(
                        str(label_path),
                        os.path.join(self.verified_dir, label_path.name)
                    )
                    success_count += 1
        
        logger.info(f"验证完成: {success_count}/{len(label_files)} 通过")
        return self.verified_dir
    
    def step3_train(self, image_dir: str, epochs: int = 100) -> str:
        """
        步骤3: 训练 YOLOv8 模型
        
        将验证后的标注转换为 YOLO 格式，然后训练模型。
        
        参数:
            image_dir: 原始图片目录
            epochs: 训练轮数
        
        返回:
            str: 训练好的模型路径 (best.pt)
        
        输出:
            yolo_dataset_YYYYMMDD_HHMMSS/ (数据集)
            runs/detect/train/weights/best.pt (模型)
        """
        logger.info("=" * 50)
        logger.info("步骤3: YOLOv8 训练")
        logger.info("=" * 50)
        
        # 1. 转换数据格式
        logger.info("转换数据格式: Labelme -> YOLO")
        self.trainer.convert_labelme_to_yolo(
            self.verified_dir, self.yolo_dir, image_dir, self.classes
        )
        
        # 2. 训练模型
        data_yaml = os.path.join(self.yolo_dir, 'data.yaml')
        model_path = self.trainer.train(data_yaml, epochs=epochs)
        
        logger.info(f"训练完成，模型保存于: {model_path}")
        return model_path
    
    def step4_infer(self, image_dir: str, model_path: str) -> str:
        """
        步骤4: YOLO 模型推理
        
        使用训练好的模型对所有图片进行推理。
        
        参数:
            image_dir: 待推理图片目录
            model_path: 模型文件路径
        
        返回:
            str: 推理结果 JSON 文件路径
        
        输出:
            inference_output_YYYYMMDD_HHMMSS/inference_results.json
        """
        logger.info("=" * 50)
        logger.info("步骤4: YOLO 推理")
        logger.info("=" * 50)
        
        result_path = self.trainer.infer(
            model_path, image_dir, self.infer_dir, self.classes
        )
        
        return result_path
    
    def step5_evaluate(self, inference_results: str) -> dict:
        """
        步骤5: 评估
        
        计算 YOLO 推理结果与 Qwen3-VL 标注的差异。
        
        参数:
            inference_results: YOLO 推理结果 JSON 文件路径
        
        返回:
            dict: 评估指标 (precision, recall, f1, tp, fp, fn)
        """
        logger.info("=" * 50)
        logger.info("步骤5: 评估")
        logger.info("=" * 50)
        
        # 转换标注结果为统一格式
        qwen_labels = []
        for label_file in Path(self.verified_dir).glob('*.json'):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取检测结果
            detections = []
            for shape in data.get('shapes', []):
                points = shape['points']
                detections.append({
                    "class_name": shape['label'],
                    "bbox": [
                        points[0][0], points[0][1],  # x1, y1
                        points[1][0], points[1][1]   # x2, y2
                    ]
                })
            
            qwen_labels.append({
                "image": str(label_file.with_suffix('.jpg')),
                "detections": detections
            })
        
        # 临时保存
        qwen_temp = os.path.join(self.base_dir, 'qwen_labels_temp.json')
        with open(qwen_temp, 'w', encoding='utf-8') as f:
            json.dump(qwen_labels, f, ensure_ascii=False)
        
        # 评估
        metrics = self.evaluator.evaluate(qwen_temp, inference_results)
        
        # 输出结果
        logger.info(f"=" * 50)
        logger.info("评估结果:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
        logger.info(f"=" * 50)
        
        # 清理临时文件
        os.remove(qwen_temp)
        
        return metrics
    
    def run(self, image_dir: str, epochs: int = 100):
        """
        运行完整工作流
        
        执行步骤 1-5，完成端到端的标注、训练、推理、评估流程。
        
        参数:
            image_dir: 图片目录路径
            epochs: 训练轮数
        
        返回:
            dict: 包含所有输出路径和评估结果
                {
                    "labelme_dir": "...",
                    "verified_dir": "...",
                    "model_path": "...",
                    "inference_results": "...",
                    "metrics": {...}
                }
        """
        logger.info("=" * 50)
        logger.info("开始执行完整工作流...")
        logger.info(f"图片目录: {image_dir}")
        logger.info(f"训练轮数: {epochs}")
        logger.info("=" * 50)
        
        # 步骤1: 标注
        labelme_dir = self.step1_annotate(image_dir)
        
        # 步骤2: 验证
        verified_dir = self.step2_verify(image_dir)
        
        # 步骤3: 训练
        model_path = self.step3_train(image_dir, epochs=epochs)
        
        # 步骤4: 推理
        inference_results = self.step4_infer(image_dir, model_path)
        
        # 步骤5: 评估
        metrics = self.step5_evaluate(inference_results)
        
        # 输出汇总
        logger.info("=" * 50)
        logger.info("工作流完成!")
        logger.info("=" * 50)
        logger.info(f"原始标注: {labelme_dir}")
        logger.info(f"验证后标注: {verified_dir}")
        logger.info(f"训练模型: {model_path}")
        logger.info(f"推理结果: {inference_results}")
        logger.info(f"评估指标: Precision={metrics['precision']}, Recall={metrics['recall']}, F1={metrics['f1']}")
        
        return {
            "labelme_dir": labelme_dir,
            "verified_dir": verified_dir,
            "model_path": model_path,
            "inference_results": inference_results,
            "metrics": metrics
        }


def main():
    """
    主函数 - 命令行入口
    
    使用方法:
        python main.py --config config.json --image-dir /path/to/images --epochs 100
    
    参数:
        --config, -c: 配置文件路径 (JSON 格式) [必填]
        --image-dir, -i: 图片目录路径 [必填]
        --epochs, -e: 训练轮数，默认 100 [可选]
    
    配置文件格式 (config.json):
        {
            "classes": ["person", "car"],     // 类别列表 [必填]
            "ollama_url": "...",              // Ollama API 地址
            "qwen_model": "qwen2-vl:7b",      // Qwen3-VL 模型
            "yolo_model": "yolov8n.pt",        // YOLO 模型
            "base_dir": "./output"             // 输出目录
        }
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Qwen3-VL + YOLOv8 训练推理工作流',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --config config.json --image-dir /data/images
  python main.py -c config.json -i /data/images -e 50
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置文件路径 (JSON 格式)'
    )
    
    parser.add_argument(
        '--image-dir', '-i',
        type=str,
        required=True,
        help='图片目录路径 (包含 JPG 文件)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='训练轮数 (默认: 100)'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    # 检查图片目录是否存在
    if not os.path.exists(args.image_dir):
        logger.error(f"图片目录不存在: {args.image_dir}")
        return
    
    # 加载配置
    logger.info(f"加载配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 添加 class_config 到配置 (兼容旧版本)
    if 'class_config' not in config:
        config['class_config'] = args.config
    
    # 验证必填参数
    if 'classes' not in config:
        logger.error("配置文件中缺少 'classes' 字段")
        return
    
    # 运行工作流
    logger.info("=" * 50)
    logger.info("启动工作流")
    logger.info("=" * 50)
    
    try:
        pipeline = Pipeline(config)
        results = pipeline.run(args.image_dir, epochs=args.epochs)
        
        # 输出最终结果
        print("\n" + "=" * 50)
        print("最终结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        import traceback
        traceback.print_exc()


# 程序入口
if __name__ == '__main__':
    main()
