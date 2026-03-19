# 使用说明 - USAGE.md

## 目录
- [前置要求](#前置要求)
- [配置步骤](#配置步骤)
- [运行方式](#运行方式)
- [输出解读](#输出解读)
- [示例](#示例)

---

## 前置要求

### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.8 | 运行环境 |
| Ollama | 最新版 | 本地 LLM 服务 |
| Git | 任意版本 | 代码管理 |

### Python 依赖

```bash
torch>=1.9.0          # PyTorch 深度学习框架
ultralytics>=8.0.0    # YOLOv8 实现
opencv-python>=4.5.0 # 图像处理
Pillow>=8.0.0        # PIL 图片处理
labelme>=5.0.0       # 标注工具
numpy>=1.19.0        # 数值计算
requests>=2.25.0     # HTTP 请求
```

### 硬件要求

| 场景 | CPU | GPU | 内存 | 存储 |
|------|-----|-----|------|------|
| 测试 (少量图片) | 4核 | 可选 | 8GB | 10GB |
| 生产 (大批量) | 8核 | 推荐 NVIDIA | 16GB+ | 50GB+ |

---

## 配置步骤

### Step 1: 克隆项目

```bash
git clone https://github.com/wjjiang2014/qwen3VL_Labeling_YOLOv8_training_infer.git
cd qwen3VL_Labeling_YOLOv8_training_infer
```

### Step 2: 安装依赖

```bash
pip install -r requirements.txt
```

### Step 3: 配置 Ollama

```bash
# 启动 Ollama 服务 (终端 1)
ollama serve

# 下载 Qwen2-VL 模型 (终端 2)
ollama pull qwen2-vl:7b

# 验证模型
ollama list
```

### Step 4: 修改配置文件

编辑 `config.json`:

```json
{
    "classes": ["person", "car", "dog", "cat", "bicycle"],
    "ollama_url": "http://localhost:11434/api/generate",
    "qwen_model": "qwen2-vl:7b",
    "yolo_model": "yolov8n.pt",
    "base_dir": "./output"
}
```

**⚠️ 必填项说明：**
- `classes`: 替换为你需要检测的类别名称

**可选配置：**
- `ollama_url`: 如果 Ollama 运行在其他机器上，修改此地址
- `qwen_model`: 可改为其他已下载的模型，如 `qwen2-vl:4b`
- `yolo_model`: 可根据精度需求选择 `yolov8n/s/m/l/x.pt`
- `base_dir`: 输出目录路径

---

## 运行方式

### 基本用法

```bash
python main.py --config config.json --image-dir /path/to/images
```

### 指定训练轮数

```bash
python main.py -c config.json -i /path/to/images -e 50
```

### 完整参数

```bash
python main.py \
    --config config.json \
    --image-dir /path/to/images \
    --epochs 100
```

---

## 输出解读

### 目录结构

每次运行会自动创建带时间戳的输出目录：

```
output/
├── labelme_output_20260320_123456/     # Step 1: Qwen3-VL 原始标注
│   ├── image1.json
│   ├── image2.json
│   └── ...
│
├── verified_output_20260320_123456/    # Step 2: 验证后的标注
│   ├── image1.json
│   └── ...
│
├── crop_verify_20260320_123456/        # 裁剪验证图片 (可选查看)
│   ├── image1_person.jpg
│   ├── image1_car.jpg
│   └── ...
│
├── yolo_dataset_20260320_123456/      # Step 3: YOLO 数据集
│   ├── images/train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── labels/train/
│   │   ├── image1.txt
│   │   └── ...
│   └── data.yaml
│
├── inference_output_20260320_123456/  # Step 4-5: YOLO 推理结果
│   └── inference_results.json
│
└── runs/detect/train/                  # Step 4: YOLO 训练输出
    └── weights/
        ├── best.pt    # 最佳模型
        └── last.pt   # 最后一次模型
```

### inference_results.json 格式

```json
[
    {
        "image": "/path/to/image1.jpg",
        "detections": [
            {
                "class_name": "person",
                "confidence": 0.95,
                "bbox": [100, 200, 300, 400]
            }
        ]
    }
]
```

### 终端输出示例

```
2026-03-20 12:34:56 - INFO - ==================
2026-03-20 12:34:56 - INFO - 步骤1: Qwen3-VL 标注
2026-03-20 12:34:56 - INFO - ==================
2026-03-20 12:34:56 - INFO - 找到 100 张图片
2026-03-20 12:35:30 - INFO - 标注完成: 95/100 成功

2026-03-20 12:35:30 - INFO - ==================
2026-03-20 12:35:30 - INFO - 步骤2: 标签验证
2026-03-20 12:35:30 - INFO - ==================
2026-03-20 12:36:15 - INFO - 验证完成: 88/95 通过

2026-03-20 12:36:15 - INFO - ==================
2026-03-20 12:36:15 - INFO - 步骤3: YOLOv8 训练
2026-03-20 12:36:15 - INFO - ==================
...
训练日志...
...

2026-03-20 12:45:00 - INFO - ==================
2026-03-20 12:45:00 - INFO - 评估结果
2026-03-20 12:45:00 - INFO - ==================
2026-03-20 12:45:00 - INFO - Precision: 0.8500
2026-03-20 12:45:00 - INFO - Recall: 0.7800
2026-03-20 12:45:00 - INFO - F1: 0.8135
```

---

## 示例

### 示例 1: 简单目标检测

假设你有一个图片文件夹 `/data/cars`，需要检测汽车和行人：

1. 创建配置 `config.json`:

```json
{
    "classes": ["car", "person"],
    "ollama_url": "http://localhost:11434/api/generate",
    "qwen_model": "qwen2-vl:7b"
}
```

2. 运行:

```bash
python main.py --config config.json --image-dir /data/cars
```

### 示例 2: 自定义模型训练

使用更大的 YOLOv8 模型，训练 200 轮：

```json
{
    "classes": ["cat", "dog", "bird"],
    "yolo_model": "yolov8m.pt"
}
```

```bash
python main.py --config config.json --image-dir /data/animals --epochs 200
```

### 示例 3: 远程 Ollama

如果 Ollama 运行在另一台服务器：

```json
{
    "classes": ["object"],
    "ollama_url": "http://192.168.1.100:11434/api/generate",
    "qwen_model": "qwen2-vl:7b"
}
```

---

## 故障排查

### 常见错误

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `Connection refused` | Ollama 未启动 | 运行 `ollama serve` |
| `CUDA out of memory` | GPU 内存不足 | 减小 `imgsz` 或使用 `yolov8n.pt` |
| `No module named 'xxx'` | 依赖未安装 | `pip install -r requirements.txt` |
| `Permission denied` | 无写入权限 | 检查 `base_dir` 路径权限 |

### 调试模式

如需查看详细日志，修改代码中的日志级别：

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

---

## 后续步骤

1. **查看标注结果**: 检查 `verified_output_*` 目录
2. **训练模型**: 使用 `runs/detect/train/weights/best.pt`
3. **评估改进**: 根据 Precision/Recall 调整类别或重新标注

---

如需更多帮助，请参阅 [DESIGN.md](./DESIGN.md) 或提交 Issue。
