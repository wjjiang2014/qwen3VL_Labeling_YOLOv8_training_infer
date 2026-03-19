# Qwen3VL Labeling + YOLOv8 Training Inference

## 📋 项目概述

使用 Qwen3-VL 进行图像半自动标注，验证后训练 YOLOv8 目标检测模型，并评估两者差异的完整端到端工作流。

### 核心功能流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  输入图片   │───▶│ Qwen3-VL   │───▶│ 标签验证   │───▶│ YOLOv8     │───▶│ 差异评估   │
│ + 类别配置  │    │ 自动标注   │    │ + 裁剪确认 │    │ 训练+推理   │    │ (P/R/F1)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wjjiang2014/qwen3VL_Labeling_YOLOv8_training_infer.git
cd qwen3VL_Labeling_YOLOv8_training_infer

# 安装 Python 依赖
pip install -r requirements.txt

# 启动 Ollama 服务 (在另一个终端)
ollama serve
# 下载 Qwen2-VL 模型
ollama pull qwen2-vl:7b
```

### 2. 配置参数

编辑 `config.json`：

```json
{
    "classes": ["person", "car", "dog", "cat", "bicycle"],
    "ollama_url": "http://localhost:11434/api/generate",
    "qwen_model": "qwen2-vl:7b",
    "yolo_model": "yolov8n.pt",
    "base_dir": "./output"
}
```

### 3. 运行

```bash
python main.py --config config.json --image-dir /path/to/images --epochs 100
```

---

## 📁 目录结构

```
qwen3VL_Labeling_YOLOv8_training_infer/
├── docs/
│   ├── DESIGN.md           # 本文档 - 设计与开发文档
│   └── USAGE.md            # 使用说明详解
├── main.py                 # 主程序入口
├── config.json             # 配置文件 ⭐ 需要修改
├── requirements.txt        # Python 依赖
├── README.md               # 项目简介
└── output/                 # 运行输出目录 (自动生成)
    ├── labelme_output_YYYYMMDD_HHMMSS/
    ├── verified_output_YYYYMMDD_HHMMSS/
    ├── crop_verify_YYYYMMDD_HHMMSS/
    ├── yolo_dataset_YYYYMMDD_HHMMSS/
    ├── inference_output_YYYYMMDD_HHMMSS/
    └── runs/detect/train/  # YOLO 训练输出
```

---

## ⚙️ 配置参数说明

### config.json 完整配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `classes` | Array | ✅ | - | 目标检测类别列表，如 `["person", "car"]` |
| `ollama_url` | String | ❌ | `http://localhost:11434/api/generate` | Ollama API 地址 |
| `qwen_model` | String | ❌ | `qwen2-vl:7b` | Qwen3-VL 模型名称（需确保已通过 `ollama pull` 下载） |
| `yolo_model` | String | ❌ | `yolov8n.pt` | YOLOv8 预训练模型路径，可选 `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` |
| `base_dir` | String | ❌ | `./output` | 输出根目录 |

### YOLOv8 模型选择建议

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| `yolov8n.pt` | 6.3M | 最快 | 基础 | 快速测试 |
| `yolov8s.pt` | 25.9M | 快 | 中等 | 平衡场景 |
| `yolov8m.pt` | 78.9M | 中 | 较高 | 生产环境 |
| `yolov8l.pt` | 198M | 慢 | 高 | 高精度需求 |
| `yolov8x.pt` | 344M | 最慢 | 最高 | 极致精度 |

### 类别配置文件格式

支持三种格式：

**格式 1: 数组**
```json
["person", "car", "dog"]
```

**格式 2: 对象 (classes 键)**
```json
{
    "classes": ["person", "car", "dog"]
}
```

**格式 3: 对象 (names 键)**
```json
{
    "names": {
        "0": "person",
        "1": "car",
        "2": "dog"
    }
}
```

---

## 📖 使用说明

### 完整工作流程

#### Step 1: Qwen3-VL 标注

读取配置文件中的类别列表，构建 Prompt，调用 Ollama API 对每张图片进行目标检测，输出 Labelme 格式的 JSON 标注文件。

**输出**: `labelme_output_YYYYMMDD_HHMMSS/*.json`

#### Step 2: 标签验证

对每个标注的边界框进行裁剪，使用 Qwen3-VL 重新验证类别是否正确：
- ✅ 正确 → 保留
- ❌ 错误 → 删除该标注
- 类别修正 → 更新类别

**输出**: `verified_output_YYYYMMDD_HHMMSS/*.json`

#### Step 3: 转换为 YOLO 格式

将 Labelme 格式转换为 YOLO 格式：
- 边界框转为 (class_id, x_center, y_center, width, height) 格式
- 坐标归一化到 0-1

**输出**: `yolo_dataset_YYYYMMDD_HHMMSS/`

```
yolo_dataset/
├── images/train/      # 图片
├── labels/train/      # YOLO 格式标注
└── data.yaml          # YOLO 数据配置
```

#### Step 4: 训练 YOLOv8

使用 `ultralytics` 库训练 YOLOv8 模型，支持早停 (patience=20)。

**输出**: `runs/detect/train/weights/best.pt`

#### Step 5: YOLO 推理

使用训练好的模型对所有图片进行推理，输出 JSON 格式的检测结果。

**输出**: `inference_output_YYYYMMDD_HHMMSS/inference_results.json`

#### Step 6: 评估

计算 YOLO 推理结果与 Qwen3-VL 标注的差异：
- **Precision**: YOLO 正确检测 / YOLO 总检测数
- **Recall**: YOLO 正确检测 / Qwen3-VL 标注总数
- **F1**: Precision 和 Recall 的调和平均

---

## 🔧 命令行参数

| 参数 | 简写 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--config` | -c | ✅ | - | 配置文件路径 (JSON) |
| `--image-dir` | -i | ✅ | - | JPG 图片目录路径 |
| `--epochs` | -e | ❌ | 100 | 训练轮数 |

### 示例

```bash
# 基本用法
python main.py --config config.json --image-dir /data/images

# 指定训练轮数
python main.py -c config.json -i /data/images -e 50
```

---

## 🏗️ 系统架构

### 模块设计

```
Pipeline (主管道)
├── Qwen3VLAnnotator (Qwen3-VL 标注)
│   ├── load_classes()        # 加载类别配置
│   ├── build_prompt()        # 构建识别 Prompt
│   ├── call_qwen3vl()        # 调用 Ollama API
│   └── convert_to_labelme()  # 转换为 Labelme 格式
│
├── LabelVerifier (标签验证)
│   └── verify_and_crop()    # 裁剪 + 重新验证
│
├── YOLOTrainer (YOLO 训练推理)
│   ├── convert_labelme_to_yolo()  # 格式转换
│   ├── train()                # 模型训练
│   └── infer()               # 模型推理
│
└── Evaluator (评估)
    └── evaluate()             # 计算 P/R/F1
```

### 核心类说明

#### Qwen3VLAnnotator

负责与 Qwen3-VL 模型交互。

```python
class Qwen3VLAnnotator:
    def __init__(self, ollama_url: str, model_name: str)
        # 初始化 Ollama 连接
        
    def load_classes(self, config_path: str) -> list
        # 从 JSON 配置文件加载类别列表
        
    def build_prompt(self, classes: list) -> str
        # 构建目标检测 Prompt
        
    def call_qwen3vl(self, image_path: str, prompt: str) -> dict
        # 调用 Ollama API，返回检测结果
        
    def convert_to_labelme(self, qwen_result: dict, ...) -> bool
        # 转换为 Labelme JSON 格式
```

#### LabelVerifier

对初步标注进行二次验证，提高标注质量。

```python
class LabelVerifier:
    def __init__(self, annotator: Qwen3VLAnnotator)
    
    def verify_and_crop(self, image_path: str, label_path: str, ...) -> bool
        # 1. 读取 Labelme JSON
        # 2. 对每个边界框裁剪图片区域
        # 3. 调用 Qwen3-VL 重新识别
        # 4. 比较类别，一致则保留，否则删除或修正
```

#### YOLOTrainer

负责 YOLO 格式转换、训练和推理。

```python
class YOLOTrainer:
    def __init__(self, model_path: str = "yolov8n.pt")
    
    def convert_labelme_to_yolo(self, labelme_dir: str, ...)
        # Labelme → YOLO 格式转换
        
    def train(self, data_yaml: str, epochs: int, imgsz: int) -> str
        # 训练 YOLOv8，返回最佳模型路径
        
    def infer(self, model_path: str, image_dir: str, ...) -> str
        # 推理，返回结果 JSON 路径
```

#### Evaluator

评估 YOLO 与 Qwen3-VL 标注的差异。

```python
class Evaluator:
    def calculate_iou(self, box1: list, box2: list) -> float
        # 计算两个边界框的 IOU
        
    def match_boxes(self, pred_boxes: list, gt_boxes: list, ...) -> tuple
        # 匹配预测框与真实框，返回 TP/FP/FN
        
    def evaluate(self, qwen_labels_path: str, yolo_results_path: str, ...) -> dict
        # 计算 Precision/Recall/F1
```

---

## 🔌 API 接口

### Ollama API 调用格式

**请求**
```json
POST http://localhost:11434/api/generate
{
    "model": "qwen2-vl:7b",
    "prompt": "请识别图片中的对象...",
    "images": ["base64编码的图片..."],
    "stream": false
}
```

**响应**
```json
{
    "response": "{\"objects\": [{\"class_name\": \"person\", \"bbox\": [0.1, 0.2, 0.5, 0.8]}]}"
}
```

### Labelme JSON 格式

```json
{
    "version": "5.0.1",
    "flags": {},
    "shapes": [
        {
            "label": "person",
            "points": [[100, 200], [300, 400]],
            "shape_type": "rectangle"
        }
    ],
    "imagePath": "image.jpg",
    "imageHeight": 480,
    "imageWidth": 640
}
```

### YOLO TXT 格式

```
# class_id x_center y_center width height (全部归一化 0-1)
0 0.5 0.6 0.3 0.4
1 0.2 0.3 0.1 0.15
```

---

## ⚠️ 注意事项

1. **Ollama 服务**: 确保 `ollama serve` 正在运行，且模型已下载
2. **内存**: 大批量图片处理建议 16GB+ 内存
3. **GPU**: YOLOv8 训练强烈建议使用 GPU (CUDA)
4. **输出目录**: 每次运行自动创建带时间戳的新目录，避免覆盖
5. **API 超时**: Qwen3-VL 推理可能较慢，已设置 120s 超时

---

## 📊 评估指标说明

### 指标计算

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

### IOU 阈值

默认 IOU threshold = 0.5，即当预测框与真实框 IOU ≥ 0.5 时视为匹配成功。

---

## 🔧 扩展开发

### 添加新的检测模型

在 `Qwen3VLAnnotator` 中添加新模型支持：

```python
class Qwen3VLAnnotator:
    def call_qwen3vl(self, image_path: str, prompt: str) -> dict:
        if self.model_name.startswith("qwen"):
            # Qwen 系列调用方式
            return self._call_qwen(image_path, prompt)
        elif self.model_name.startswith("llama"):
            # Llama 系列调用方式
            return self._call_llama(image_path, prompt)
        # 添加更多模型...
```

### 自定义后处理

在 `convert_to_labelme()` 中添加后处理逻辑：

```python
def convert_to_labelme(self, qwen_result: dict, ...):
    # 添加过滤条件
    objects = qwen_result.get('objects', [])
    filtered = [obj for obj in objects if obj['confidence'] > 0.7]
    # ...
```

---

## 📝 版本历史

- **v1.0.0** (2026-03-20): 初始版本
  - Qwen3-VL 标注
  - 标签验证与裁剪
  - YOLOv8 训练推理
  - Precision/Recall/F1 评估

---

## ❓ 常见问题

### Q1: Ollama 连接失败

**问题**: `ConnectionError: [Errno 111] Connection refused`

**解决**:
1. 确保 Ollama 服务已启动: `ollama serve`
2. 检查端口是否正确: `ollama_url` 配置
3. 验证模型已下载: `ollama list`

### Q2: YOLO 训练报错 CUDA

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减小 `imgsz` 参数 (如 320)
2. 减小 `batch_size` (在 `train()` 中修改)
3. 使用更小的模型 `yolov8n.pt`

### Q3: 标注结果为空

**问题**: Qwen3-VL 未返回任何检测结果

**解决**:
1. 检查 Prompt 是否正确
2. 确认图片格式为 JPG
3. 查看日志中的错误信息

### Q4: GitHub 推送失败

**问题**: `remote: Support for password authentication was removed`

**解决**: 使用 Personal Access Token (PAT) 作为密码，或使用 SSH 方式推送

---

## 📞 后续支持

如有问题或建议，请提交 Issue 或 Pull Request。

---

## 📄 License

MIT License
