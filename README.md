# Qwen3VL Labeling + YOLOv8 Training Inference

使用 Qwen3-VL 进行图像标注，验证后训练 YOLOv8 模型，并评估两者差异的完整工作流。

## 功能

1. **Qwen3-VL 标注**: 使用 Qwen3-VL (Ollama API) 读取类别配置，生成 Labelme 格式标注
2. **标签验证**: 裁剪检测区域，使用 Qwen3-VL 重新验证类别是否正确
3. **YOLOv8 训练**: 使用验证后的标注数据训练 YOLOv8 模型
4. **模型推理**: 使用训练好的模型对图片进行推理
5. **评估**: 计算 Qwen3-VL 标注与 YOLO 推理结果的 Precision/Recall/F1

## 目录结构

```
.
├── main.py              # 主程序
├── config.json          # 配置文件
├── requirements.txt    # 依赖
└── README.md
```

## 配置 (config.json)

```json
{
    "classes": ["person", "car", "dog", "cat", "bicycle"],
    "ollama_url": "http://localhost:11434/api/generate",
    "qwen_model": "qwen2-vl:7b",
    "yolo_model": "yolov8n.pt",
    "base_dir": "./output"
}
```

### 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| classes | 目标检测类别列表 | 必填 |
| ollama_url | Ollama API 地址 | http://localhost:11434/api/generate |
| qwen_model | Qwen3-VL 模型名称 | qwen2-vl:7b |
| yolo_model | YOLOv8 预训练模型 | yolov8n.pt |
| base_dir | 输出目录 | ./output |

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

修改 `config.json`，设置你的类别列表。

### 3. 运行

```bash
python main.py --config config.json --image-dir /path/to/images --epochs 100
```

参数说明:
- `--config`: 配置文件路径 (必填)
- `--image-dir`: JPG 图片目录 (必填)
- `--epochs`: 训练轮数 (默认 100)

## 输出目录

每次运行会自动创建带时间戳的输出目录:

```
output/
├── labelme_output_20260320_123456/    # Qwen3-VL 原始标注
├── verified_output_20260320_123456/   # 验证后的标注
├── crop_verify_20260320_123456/      # 裁剪验证图片
├── yolo_dataset_20260320_123456/     # YOLO 数据集
├── inference_output_20260320_123456/ # YOLO 推理结果
└── runs/detect/train/                 # 训练输出
```

## 评估指标

工作流最后会输出:
- **Precision**: YOLO 检测正确的框 / YOLO 总检测数
- **Recall**: YOLO 检测正确的框 / Qwen3-VL 标注总数
- **F1**: Precision 和 Recall 的调和平均

## 环境要求

- Python 3.8+
- Ollama (运行 Qwen3-VL 模型)
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- labelme

## License

MIT
