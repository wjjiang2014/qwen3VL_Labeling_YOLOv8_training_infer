"""
Qwen3-VL 标注 + YOLOv8 训练推理完整工作流
"""
import os
import json
import base64
import shutil
import time
from pathlib import Path
from datetime import datetime
import requests
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import labelme
from labelme import utils
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Qwen3VLAnnotator:
    """Qwen3-VL 标注器"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate",
                 model_name: str = "qwen2-vl:7b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
    
    def load_classes(self, config_path: str) -> list:
        """从JSON配置文件加载类别名称"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 支持多种格式: {"classes": [...]} 或 ["class1", "class2"]
        if isinstance(config, dict):
            if 'classes' in config:
                return config['classes']
            elif 'names' in config:
                return config['names']
            else:
                # 假设所有键都是类别名
                return list(config.keys())
        elif isinstance(config, list):
            return config
        else:
            raise ValueError("配置格式不支持")
    
    def build_prompt(self, classes: list) -> str:
        """构建识别提示词"""
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
        """将图片编码为base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def call_qwen3vl(self, image_path: str, prompt: str) -> dict:
        """调用 Ollama API 进行图像识别"""
        image_base64 = self.encode_image(image_path)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # 解析返回的JSON
            text = result.get('response', '')
            # 尝试提取JSON部分
            try:
                # 尝试直接解析
                return json.loads(text)
            except:
                # 尝试提取 ```json ... ``` 块
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.warning(f"无法解析Qwen3-VL返回: {text[:200]}")
                    return {"objects": []}
        except Exception as e:
            logger.error(f"调用Qwen3-VL失败: {e}")
            return {"objects": []}
    
    def denormalize_bbox(self, bbox: list, width: int, height: int) -> list:
        """反归一化边界框"""
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        ]
    
    def convert_to_labelme(self, qwen_result: dict, image_path: str, 
                          output_path: str, classes: list) -> bool:
        """转换为 Labelme 格式 JSON"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            height, width = img.shape[:2]
            
            shapes = []
            for obj in qwen_result.get('objects', []):
                class_name = obj.get('class_name', '')
                if class_name not in classes:
                    continue
                
                bbox = obj.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # 反归一化
                x1, y1, x2, y2 = self.denormalize_bbox(bbox, width, height)
                
                shapes.append({
                    "label": class_name,
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "flags": {}
                })
            
            if not shapes:
                return False
            
            labelme_json = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": os.path.basename(image_path),
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_json, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"转换Labelme格式失败: {e}")
            return False


class LabelVerifier:
    """标签验证器 - 裁剪后重新验证"""
    
    def __init__(self, annotator: Qwen3VLAnnotator):
        self.annotator = annotator
    
    def verify_and_crop(self, image_path: str, label_path: str, 
                       classes: list, output_crop_dir: str) -> bool:
        """验证标签并裁剪区域"""
        try:
            # 读取标签
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            valid_shapes = []
            
            for shape in label_data.get('shapes', []):
                label = shape.get('label', '')
                points = shape.get('points', [])
                
                if len(points) != 2:
                    continue
                
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                
                # 裁剪区域
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                y1, y2 = max(0, y1), min(img.shape[0], y1)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop_img = img[y1:y2, x1:x2]
                
                # 保存裁剪图
                crop_path = os.path.join(output_crop_dir, 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}.jpg")
                cv2.imwrite(crop_path, crop_img)
                
                # 重新验证
                prompt = f"""请识别这张图片中的对象，从以下类别中选择: {classes}。
如果图片中包含{label}类别的物体，请返回JSON格式：
{{"correct": true}}
否则返回：
{{"correct": false, "actual_class": "实际类别"}}
只返回JSON。"""
                
                result = self.annotator.call_qwen3vl(crop_path, prompt)
                
                is_correct = result.get('correct', False)
                if not is_correct and label in classes:
                    # 检查实际类别
                    actual = result.get('actual_class', '')
                    if actual and actual in classes:
                        # 更新标签
                        shape['label'] = actual
                        valid_shapes.append(shape)
                    else:
                        logger.warning(f"标签验证失败，将删除: {label_path}")
                        return False
                else:
                    valid_shapes.append(shape)
            
            if valid_shapes:
                label_data['shapes'] = valid_shapes
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False


class YOLOTrainer:
    """YOLOv8 训练器"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
    
    def convert_labelme_to_yolo(self, labelme_dir: str, output_dir: str, 
                                 image_dir: str, classes: list):
        """将 Labelme 格式转换为 YOLO 格式"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        
        labelme_files = list(Path(labelme_dir).glob('*.json'))
        
        for labelme_file in labelme_files:
            # 读取labelme JSON
            with open(labelme_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 对应的图片文件
            img_name = data.get('imagePath', labelme_file.stem + '.jpg')
            img_path = os.path.join(image_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            # 复制图片
            dest_img = os.path.join(output_dir, 'images', 'train', img_name)
            if not os.path.exists(dest_img):
                shutil.copy(img_path, dest_img)
            
            # 转换标注
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            yolo_lines = []
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                if label not in classes:
                    continue
                
                class_id = classes.index(label)
                points = shape.get('points', [])
                
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 转换为 YOLO 格式 (中心点 + 宽高，归一化)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}\n")
            
            # 写入 YOLO 标注文件
            yolo_file = os.path.join(output_dir, 'labels', 'train', 
                                     os.path.splitext(img_name)[0] + '.txt')
            with open(yolo_file, 'w') as f:
                f.writelines(yolo_lines)
        
        # 创建 data.yaml
        yaml_content = f"""train: images/train
val: images/train

nc: {len(classes)}
names: {classes}
"""
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"转换完成: {len(labelme_files)} 个文件 -> YOLO格式")
    
    def train(self, data_yaml: str, epochs: int = 100, 
              imgsz: int = 640) -> str:
        """训练 YOLOv8 模型"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project='runs/detect',
            name='train',
            exist_ok=True,
            patience=20,  # 早停
            save=True,
            plots=True
        )
        
        # 返回最佳模型路径
        best_model = os.path.join('runs/detect', 'train', 'weights', 'best.pt')
        return best_model
    
    def infer(self, model_path: str, image_dir: str, 
              output_dir: str, classes: list):
        """使用训练好的模型进行推理"""
        model = YOLO(model_path)
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = list(Path(image_dir).glob('*.jpg'))
        
        results_json = []
        
        for img_path in image_files:
            result = model(img_path, verbose=False)
            
            # 提取检测结果
            boxes = result[0].boxes
            detections = []
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    detections.append({
                        "class_name": classes[class_id],
                        "confidence": conf,
                        "bbox": xyxy
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
        return output_file


class Evaluator:
    """评估器 - 计算召回率和精确率"""
    
    def __init__(self):
        pass
    
    def calculate_iou(self, box1: list, box2: list) -> float:
        """计算两个边界框的IOU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # 计算并集
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_boxes(self, pred_boxes: list, gt_boxes: list, 
                    iou_threshold: float = 0.5) -> tuple:
        """匹配预测框和真实框，返回TP, FP, FN"""
        matched_gt = set()
        tp = 0
        fp = 0
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold and pred['class_name'] == gt_boxes[best_gt_idx]['class_name']:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        
        return tp, fp, fn
    
    def evaluate(self, qwen_labels_path: str, yolo_results_path: str,
                 iou_threshold: float = 0.5) -> dict:
        """评估 YOLO 和 Qwen3-VL 标注的差异"""
        
        # 加载 Qwen3-VL 标签
        with open(qwen_labels_path, 'r', encoding='utf-8') as f:
            qwen_data = json.load(f)
        
        # 加载 YOLO 推理结果
        with open(yolo_results_path, 'r', encoding='utf-8') as f:
            yolo_data = json.load(f)
        
        # 按图片名称匹配
        yolo_dict = {os.path.basename(item['image']): item['detections'] 
                     for item in yolo_data}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for qwen_item in qwen_data:
            img_name = os.path.basename(qwen_item['image'])
            
            pred_boxes = yolo_dict.get(img_name, [])
            gt_boxes = qwen_item['detections']
            
            if gt_boxes:
                tp, fp, fn = self.match_boxes(pred_boxes, gt_boxes, iou_threshold)
                total_tp += tp
                total_fp += fp
                total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        }


class Pipeline:
    """完整工作流管道"""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_dir = config.get('base_dir', '.')
        
        # 创建带时间戳的输出目录
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
        logger.info(f"类别: {self.classes}")
    
    def step1_annotate(self, image_dir: str) -> str:
        """步骤1: 使用 Qwen3-VL 标注"""
        logger.info("=" * 50)
        logger.info("步骤1: Qwen3-VL 标注")
        logger.info("=" * 50)
        
        prompt = self.annotator.build_prompt(self.classes)
        
        image_files = list(Path(image_dir).glob('*.jpg'))
        logger.info(f"找到 {len(image_files)} 张图片")
        
        success_count = 0
        for img_path in image_files:
            label_path = os.path.join(self.labelme_dir, 
                                       img_path.stem + '.json')
            
            result = self.annotator.call_qwen3vl(str(img_path), prompt)
            
            if self.annotator.convert_to_labelme(result, str(img_path), 
                                                   label_path, self.classes):
                success_count += 1
        
        logger.info(f"标注完成: {success_count}/{len(image_files)} 成功")
        
        # 保存标注结果摘要
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "annotated": success_count,
            "output_dir": self.labelme_dir
        }
        
        return self.labelme_dir
    
    def step2_verify(self, image_dir: str) -> str:
        """步骤2: 验证并裁剪"""
        logger.info("=" * 50)
        logger.info("步骤2: 标签验证")
        logger.info("=" * 50)
        
        label_files = list(Path(self.labelme_dir).glob('*.json'))
        success_count = 0
        
        for label_path in label_files:
            img_name = label_path.stem + '.jpg'
            img_path = os.path.join(image_dir, img_name)
            
            if os.path.exists(img_path):
                if self.verifier.verify_and_crop(img_path, str(label_path),
                                                   self.classes, self.crop_dir):
                    # 移动到验证目录
                    shutil.copy(str(label_path), 
                               os.path.join(self.verified_dir, label_path.name))
                    success_count += 1
        
        logger.info(f"验证完成: {success_count}/{len(label_files)} 通过")
        return self.verified_dir
    
    def step3_train(self, image_dir: str, epochs: int = 100) -> str:
        """步骤3: 训练 YOLOv8"""
        logger.info("=" * 50)
        logger.info("步骤3: YOLOv8 训练")
        logger.info("=" * 50)
        
        # 转换数据格式
        self.trainer.convert_labelme_to_yolo(
            self.verified_dir, self.yolo_dir, image_dir, self.classes
        )
        
        # 训练
        data_yaml = os.path.join(self.yolo_dir, 'data.yaml')
        model_path = self.trainer.train(data_yaml, epochs=epochs)
        
        logger.info(f"训练完成，模型保存于: {model_path}")
        return model_path
    
    def step4_infer(self, image_dir: str, model_path: str) -> str:
        """步骤4: YOLO 推理"""
        logger.info("=" * 50)
        logger.info("步骤4: YOLO 推理")
        logger.info("=" * 50)
        
        result_path = self.trainer.infer(model_path, image_dir, 
                                          self.infer_dir, self.classes)
        
        logger.info(f"推理结果保存于: {result_path}")
        return result_path
    
    def step5_evaluate(self, inference_results: str) -> dict:
        """步骤5: 评估"""
        logger.info("=" * 50)
        logger.info("步骤5: 评估")
        logger.info("=" * 50)
        
        # 转换标注结果为统一格式
        qwen_labels = []
        for label_file in Path(self.verified_dir).glob('*.json'):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            detections = []
            for shape in data.get('shapes', []):
                points = shape['points']
                detections.append({
                    "class_name": shape['label'],
                    "bbox": [points[0][0], points[0][1], 
                             points[1][0], points[1][1]]
                })
            
            qwen_labels.append({
                "image": str(label_file.with_suffix('.jpg')),
                "detections": detections
            })
        
        # 临时保存Qwen标注
        qwen_temp = os.path.join(self.base_dir, 'qwen_labels_temp.json')
        with open(qwen_temp, 'w', encoding='utf-8') as f:
            json.dump(qwen_labels, f, ensure_ascii=False)
        
        metrics = self.evaluator.evaluate(qwen_temp, inference_results)
        
        logger.info(f"评估结果: Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        # 清理临时文件
        os.remove(qwen_temp)
        
        return metrics
    
    def run(self, image_dir: str, epochs: int = 100):
        """运行完整工作流"""
        logger.info("开始执行完整工作流...")
        logger.info(f"图片目录: {image_dir}")
        
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
        
        logger.info("=" * 50)
        logger.info("工作流完成!")
        logger.info("=" * 50)
        logger.info(f"标注输出: {labelme_dir}")
        logger.info(f"验证输出: {verified_dir}")
        logger.info(f"推理输出: {inference_results}")
        logger.info(f"评估指标: {metrics}")
        
        return {
            "labelme_dir": labelme_dir,
            "verified_dir": verified_dir,
            "model_path": model_path,
            "inference_results": inference_results,
            "metrics": metrics
        }


def main():
    """主函数 - 通过配置文件运行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Qwen3-VL + YOLOv8 训练推理工作流')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径 (JSON)')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='图片目录路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 运行工作流
    pipeline = Pipeline(config)
    results = pipeline.run(args.image_dir, epochs=args.epochs)
    
    print("\n" + "=" * 50)
    print("最终结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
