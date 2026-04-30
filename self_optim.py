import json
import os
import re
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from qwen3_vl_infer import infer

# 获取脚本所在目录，用于日志文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置logging
log_file = os.path.join(SCRIPT_DIR, 'self_optim.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# IOU阈值，用于判断检测是否有效
IOU_THRESHOLD = 0.3

def classify_json_error(raw_output: str) -> str:
    """
    判断是否为JSON解析错误（仅在JSON解析失败时调用）
    不再区分具体错误类型，只要解析失败就返回 "json_error"

    Returns:
        str: "json_error" | "empty_output"
    """
    if not raw_output or not raw_output.strip():
        return "empty_output"
    return "json_error"

with open("self_optim_config.json", encoding='utf-8') as f:
    config = json.load(f)

with open("prompt.json", "r") as f:
    orig_prompt = json.load(f)
    det_prompt = orig_prompt["draw_box"]

# 类别和人工经验一一对应
class_name = config["class_name"]
human_experience = config.get("human_experience", "")
det_prompt = det_prompt.format(class_names=class_name)
src_path = "self-optim/data"

# 多图优化模板：聚合所有图像的目标特征
# prev_prompt: 上一版提示词，prev_recall: 上一版召回率，curr_recall: 当前召回率
self_optim_prompt_template = (
    "上一版提示词：{prev_prompt}\n"
    "当前提示词：{prompt}\n"
    "优化评价：{eval_message}\n"
    "我需要在多张图像中识别以下类别：{class_names}。\n"
    "人工经验（判断目标的重要依据）：{human_experience}\n"
    "注意：以下图像中绿色边框标注了当前提示词能够识别的目标，红色边框标注了当前提示词未能识别的目标（红框内是漏检的目标）。\n"
    "请在理解人工经验基础上，分析图像中绿色边框（已识别）与红色边框（漏检）的目标，提取其共同特征，归纳漏检原因，在保持类别泛化性的前提下优化提示词，确保不影响已识别目标。"
    "返回的提示词中需要必须包含对结果的格式要求：输出JSON数组格式：[{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"类别名\"}}, ...], 无目标时返回：[]。\n"
    "生成的提示词不能和原提示词一样，不需要任何多余的解释，只需要返回最终优化后的提示词。"
)

# JSON解析错误修复专用提示词模板
json_error_fix_template = (
    "目标：优化检测提示词使其输出纯JSON数组。\n"
    "类别：{class_names}\n"
    "人工经验：{human_experience}\n"
    "【问题提示词】{bad_prompt}\n"
    "【正常提示词】{good_prompt}\n"
    "问题提示词VL模型的输出（无法解析为JSON）：\n{raw_output}\n"
    "常见问题包括但不限于：\n问题1：输出中包含推理文字而非纯检测box的JSON数组。\n问题2：出现大量重叠或者无关目标的检测框\n"
    "错误原因：\n针对问题1： 提示词中缺少对输出格式的约束。\n针对问题2：提示词中检测对象描述不合适，条件过于抽象，可降低语义复杂度\n"
    "请对比两个提示词，分析问题原因，优化【问题提示词】使其：\n"
    "1. 只输出纯JSON数组，不含任何推理文字\n"
    "2. JSON格式：[{{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"类别\"}}]\n"
    "3. 每个目标只输出一个框\n"
    "4. 保持类别泛化性\n"
    "只返回优化后的提示词，不要解释"
)


# 针对未识别目标的二次优化模板
focus_unmatched_template = (
    "当前提示词：{prompt}\n"
    "我需要对以下类别进行目标检测：{class_names}。\n"
    "人工经验（判断目标的重要依据）：{human_experience}\n"
    "注意：以下图像中绿色边框标注了当前提示词能够识别的目标，红色边框标注了当前提示词未能识别的目标（红框内是漏检的目标）。\n"
    "我需要你根据检测结果分析，优化提示词，要求：\n"
    "1. 保持对已识别目标的识别能力（不能降低召回率）\n"
    "2. 重点关注如何识别出红色边框标注的漏检目标\n"
    "3. 适当调整特征描述，使其对漏检目标更敏感\n"
    "4. 保持对{class_names}类别的泛化性\n"
    "5. 最终只返回优化后的提示词，不要其他解释"
)

# 汇总标注框模板
boxes_summary_template = "图像{i}上有{count}个目标，坐标归一化列表：{box_list}"

def compute_iou(box1, box2):
    """
    计算两个框的IOU

    Args:
        box1, box2: [x1, y1, x2, y2] 像素坐标（两个框坐标系必须一致）

    Returns:
        float: IOU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_detections_to_ground_truth(dets, true_boxes, iou_threshold=IOU_THRESHOLD):
    """
    将检测结果与人工标注框进行匹配（两者坐标均为归一化[0,1]）

    Args:
        dets: 检测结果列表，每项为 {"bbox_2d": [x1,y1,x2,y2], "label": str}
              归一化坐标 [0,1]，基于预处理图像尺寸
        true_boxes: 人工标注列表，每项为 (label, points) 其中points为[[x1,y1],[x2,y2]]
              归一化坐标 [0,1]，基于原始图像尺寸
        iou_threshold: IOU阈值

    Returns:
        tuple: (matched_gt_indices, unmatched_gt_indices)
            matched_gt_indices: 已匹配的标注框索引列表
            unmatched_gt_indices: 未匹配的标注框索引列表
    """
    if not true_boxes:
        return [], []

    matched_gt_indices = []
    unmatched_gt_indices = []

    # true_boxes本身就是归一化坐标[0,1]，直接使用
    gt_boxes = []
    for label, points in true_boxes:
        x1_norm, y1_norm = points[0]
        x2_norm, y2_norm = points[1]
        gt_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

    # 跟踪已匹配的gt
    gt_matched = [False] * len(gt_boxes)

    for det in dets:
        det_box = det.get("bbox_2d", [])
        if len(det_box) != 4:
            continue

        # det_box已经是归一化坐标[0,1]，直接与gt_boxes计算IOU
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt_indices.append(best_gt_idx)
            gt_matched[best_gt_idx] = True

    unmatched_gt_indices = [i for i, m in enumerate(gt_matched) if not m]

    return matched_gt_indices, unmatched_gt_indices

def draw_red_boxes_on_images(all_data: list) -> list:
    """
    在所有图像上绘制红色边框标注目标区域

    Args:
        all_data: 所有图像数据列表

    Returns:
        list: 绘制了红框的图像列表（不修改原图）
    """
    images_with_boxes = []
    for img_data in all_data:
        image = img_data["image"].copy()
        boxes = img_data["true_boxes"]
        w, h = image.size

        draw = ImageDraw.Draw(image)

        # 红色边框，线宽3
        box_color = (255, 0, 0)

        for label, points in boxes:
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 坐标是归一化的，需要转换为像素坐标
            x1_pixel = int(x1 * w)
            y1_pixel = int(y1 * h)
            x2_pixel = int(x2 * w)
            y2_pixel = int(y2 * h)

            # 绘制红色边框矩形
            for t in range(3):  # 线宽3
                draw.rectangle(
                    [x1_pixel - t, y1_pixel - t, x2_pixel + t, y2_pixel + t],
                    outline=box_color
                )

        images_with_boxes.append(image)
        print(f"已为 {img_data['filename']} 绘制 {len(boxes)} 个红框")

    return images_with_boxes


def draw_match_results_on_images(all_data: list, per_image_results: list) -> list:
    """
    在图像上绘制匹配结果：绿色=已识别，红色=未识别

    Args:
        all_data: 所有图像数据列表
        per_image_results: 每张图的检测结果（包含matched/unmatched信息）

    Returns:
        list: 绘制了颜色标记的图像列表
    """
    images_with_marks = []
    green = (0, 255, 0)  # 绿色-已匹配
    red = (255, 0, 0)    # 红色-未匹配
    line_width = 3

    for img_data, result in zip(all_data, per_image_results):
        image = img_data["image"].copy()
        true_boxes = img_data["true_boxes"]
        matched_indices = result.get("matched_gt_indices", [])
        unmatched_indices = result.get("unmatched_gt_indices", [])
        w, h = image.size

        draw = ImageDraw.Draw(image)

        # 绘制已匹配的目标（绿色）
        for idx in matched_indices:
            label, points = true_boxes[idx]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1_p, y1_p = int(x1 * w), int(y1 * h)
            x2_p, y2_p = int(x2 * w), int(y2 * h)
            for t in range(line_width):
                draw.rectangle([x1_p - t, y1_p - t, x2_p + t, y2_p + t], outline=green)

        # 绘制未匹配的目标（红色）
        for idx in unmatched_indices:
            label, points = true_boxes[idx]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1_p, y1_p = int(x1 * w), int(y1 * h)
            x2_p, y2_p = int(x2 * w), int(y2 * h)
            for t in range(line_width):
                draw.rectangle([x1_p - t, y1_p - t, x2_p + t, y2_p + t], outline=red)

        matched_count = len(matched_indices)
        unmatched_count = len(unmatched_indices)
        print(f"  {img_data['filename']}: 已识别：{matched_count}, 未识别：{unmatched_count}")
        images_with_marks.append(image)
    return images_with_marks


def load_all_images_data(src_path: str) -> list:
    """
    加载src_path下所有图像及其标注

    Returns:
        list: [{"image": PIL.Image, "filename": str, "true_boxes": [(label, points), ...]}, ...]
    """
    all_data = []
    load_summary = []
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
            # true_boxes: [(label, points), ...]，label保留用于记录是哪个目标
            true_boxes = [(shape['label'], shape["points"]) for shape in shapes]

            all_data.append({
                "image": image,
                "filename": file,
                "true_boxes": true_boxes,
                "path": img_file
            })
            load_summary.append(f"{file}: {len(true_boxes)}个目标")

    logger.info(f"加载图像完成，共{len(all_data)}张图像，详情: {load_summary}")
    return all_data


def evaluate_prompt_on_images(prompt: str, all_data: list, use_red_box: bool = True) -> tuple:
    """
    在所有图像上评估提示词，返回详细检测结果

    Returns:
        tuple: (total_match_count, total_true_count, per_image_results, error_type, error_raw_output)
            total_match_count: 总匹配数（有效检测数）
            total_true_count: 总真实目标数
            per_image_results: 每张图的详细结果
            error_type: "json_error" | "empty_output"
            error_raw_output: 错误时的VL模型原始输出（如果有的话）
    """
    total_match_count = 0
    total_true_count = 0
    per_image_results = []
    error_type = None  # None | "reasoning_text" | "unclosed_json"
    error_raw_output = None  # 保存错误时的原始输出

    for img_data in all_data:
        image = img_data["image"]
        filename = img_data["filename"]
        true_boxes = img_data["true_boxes"]
        true_count = len(true_boxes)
        total_true_count += true_count
        dets, _, raw_output = infer(prompt, [image], show=False, crop=False)

        # 如果解析失败，存储原始输出用于后续优化
        if dets is None:
            error_type = classify_json_error(raw_output)
            error_raw_output = raw_output
            logger.warning(f"JSON解析错误: 模型输出无法被解析为JSON")
            logger.warning(f"原始输出内容 ({len(raw_output)} 字符):\n{raw_output[:1500]}")
            if len(raw_output) > 1500:
                print(f"...(省略 {len(raw_output)-1500} 字符)")
            dets = []
            matched_gt = []
            unmatched_gt = list(range(true_count))
        else:
            # JSON解析成功后，基于IoU确认识别效果
            matched_gt, unmatched_gt = match_detections_to_ground_truth(dets, true_boxes)

        match_count = len(matched_gt)
        total_match_count += match_count

        per_image_results.append({
            "filename": filename,
            "true_count": true_count,
            "match_count": match_count,
            "det_count": len(dets) if dets else 0,
            "matched_gt_indices": matched_gt,
            "unmatched_gt_indices": unmatched_gt,
            "true_boxes": true_boxes,
            "dets": dets
        })

        # 打印详细信息
        recall = match_count / true_count if true_count > 0 else 0
        print(f"  {filename}: 真实{true_count}, 检测{len(dets) if dets else 0}, 匹配{match_count}, 召回率{recall:.2%}")

    overall_recall = total_match_count / total_true_count if total_true_count > 0 else 0
    print(f"  总体: 真实{total_true_count}, 匹配{total_match_count}, 总体召回率{overall_recall:.2%}")

    return total_match_count, total_true_count, per_image_results, error_type, error_raw_output


def self_optimize(all_data: list, max_iter: int = 5) -> tuple:
    """
    多图共同优化提示词

    Args:
        all_data: 所有图像数据列表
        max_iter: 最大迭代次数

    Returns:
        tuple: (最优提示词, 所有提示词及检测结果列表)
    """
    human_exp_str = human_experience
    # 详细记录每次迭代的结果，用于追溯
    # 每次评估后 append: {"prompt": str, "match_count": int, "results": list, "error_type": str|None}
    iter_prompts_detail = []

    # ========== 接口1: evaluate_prompt_on_images - 评估初始提示词 ==========
    logger.info("评估初始提示词")
    best_match_count, total_true_count, best_results, initial_error_type, initial_error_raw_output = evaluate_prompt_on_images(det_prompt, all_data)

    best_prompt = det_prompt
    best_match_count_current = best_match_count
    best_results_current = best_results

    # 记录初始评估结果
    iter_prompts_detail.append({
        "prompt": det_prompt,
        "match_count": best_match_count,
        "results": best_results,
        "error_type": initial_error_type,
        "error_raw_output": initial_error_raw_output
    })

    # 当前待评估的提示词和结果（初始为det_prompt的评估结果）
    current_prompt = det_prompt
    current_results = best_results
    current_match_count = best_match_count
    last_error_type = initial_error_type  # 记录上一个错误的类型
    last_error_raw_output = initial_error_raw_output  # 记录上一个错误的原始输出

    for i in range(max_iter):
        # ========== 第一步：检查是否所有目标都被检测到 ==========
        all_detected = all(res["match_count"] >= res["true_count"] for res in current_results)
        if all_detected and current_match_count > 0:
            logger.info("所有目标均已检测到，停止迭代，并保存最终检测结果...")
            final_marked_images = draw_match_results_on_images(all_data, current_results)
            final_save_dir = "tmp/marked"
            os.makedirs(final_save_dir, exist_ok=True)
            for img, img_data in zip(final_marked_images, all_data):
                save_path = os.path.join(final_save_dir, f"final_{img_data['filename']}")
                img.save(save_path)
            logger.info(f"最终检测结果已保存到 {final_save_dir}")
            break
        print("\n" + f"第 {i+1} 次迭代".center(60, "="))

        # ========== 第二步：根据是否有JSON解析错误选择优化策略 ==========
        if last_error_type == "json_error":
            # JSON解析错误：使用专用错误修复模板
            # 找到历史迭代中最后一个没有JSON解析错误的提示词
            good_entry = None
            for entry in reversed(iter_prompts_detail):
                if entry.get("error_type") != "json_error":
                    good_entry = entry
                    break
            if good_entry is None:
                logger.warning("警告：找不到正常提示词，使用初始提示词")
                good_entry = iter_prompts_detail[0]

            good_prompt = good_entry["prompt"]
            bad_prompt = current_prompt  # 导致JSON解析错误的提示词

            logger.warning(f"检测到JSON解析错误，使用错误修复模板优化提示词")
            prompt_template = json_error_fix_template
            results_for_draw = current_results
            raw_output_for_template = last_error_raw_output if last_error_raw_output else "（无原始输出）"
        else:
            # 正常情况：使用标准优化模板
            prompt_to_optimize = current_prompt
            results_for_draw = current_results
            prompt_template = self_optim_prompt_template
            raw_output_for_template = None

            # 获取上一版提示词和召回率
            if len(iter_prompts_detail) >= 2:
                prev_entry = iter_prompts_detail[-2]
                prev_prompt = prev_entry["prompt"]
                prev_match = prev_entry["match_count"]
                prev_recall = prev_match / total_true_count if total_true_count > 0 else 0
            else:
                # 首次优化，没有上一版
                prev_prompt = "（无上一版）"
                prev_recall = 0

            curr_recall = current_match_count / total_true_count if total_true_count > 0 else 0

            # 根据召回率变化生成动态评价
            if prev_recall == 0:
                eval_message = "提示词缺少对对象核心特征的描述，导致模型难以准确定位。"
            elif curr_recall > prev_recall:
                eval_message = f"召回率从{prev_recall:.2%}提升到{curr_recall:.2%}，优化方向正确，但仍有提升空间。"
            elif curr_recall == prev_recall:
                eval_message = "召回率未变化，需要尝试新的优化策略。"
            else:
                eval_message = f"召回率从{prev_recall:.2%}下降到{curr_recall:.2%}，说明当前优化方向不正确，需要调整策略。"

        marked_images = draw_match_results_on_images(all_data, results_for_draw)

        # 临时保存marked_images到tmp/marked目录
        marked_save_dir = "tmp/marked"
        os.makedirs(marked_save_dir, exist_ok=True)
        for img, img_data in zip(marked_images, all_data):
            save_path = os.path.join(marked_save_dir, f"iter{i+1}_{img_data['filename']}")
            img.save(save_path)

        # ========== 接口2: infer - 基于绿红框图像生成新提示词 ==========
        # 基于选择的提示词生成优化后的新提示词
        format_kwargs = {
            "prompt": prompt_to_optimize,
            "class_names": class_name,
            "human_experience": human_exp_str
        }
        if last_error_type == "json_error":
            format_kwargs["good_prompt"] = good_prompt
            format_kwargs["bad_prompt"] = bad_prompt
            format_kwargs["raw_output"] = raw_output_for_template[:2000] if raw_output_for_template else ""
        else:
            format_kwargs["prev_prompt"] = prev_prompt
            format_kwargs["eval_message"] = eval_message

        prompt_for_infer = prompt_template.format(**format_kwargs)
        new_prompt, _ = infer(prompt_for_infer, marked_images, isdets=False)

        prev_recall_str = f"{current_match_count}/{total_true_count} ({current_match_count/total_true_count:.1%})" if current_match_count > 0 else "无前序数据"
        logger.info(f"[优化前] 召回率: {prev_recall_str} | 提示词: {prev_prompt}")
        logger.info(f"[优化后] 提示词: {new_prompt}")

        current_match_count, _, current_results, error_type, error_raw_output = evaluate_prompt_on_images(new_prompt, all_data)
        current_recall = current_match_count / total_true_count if total_true_count > 0 else 0

        result_lines = [f"  {r['filename']}: 真实{r['true_count']}, 匹配{r['match_count']}, 检测{r['det_count']}" for r in current_results]
        logger.info(f"[验证新提示词结果]\n" + "\n".join(result_lines))
        logger.info(f"[总体] 召回率: {current_match_count}/{total_true_count} ({current_recall:.1%})")

        # 记录本次评估结果
        iter_prompts_detail.append({
            "prompt": new_prompt,
            "match_count": current_match_count,
            "results": current_results,
            "error_type": error_type,
            "error_raw_output": error_raw_output
        })

        # 更新错误类型记录
        last_error_type = error_type
        last_error_raw_output = error_raw_output

        # 更新当前最佳结果
        if current_match_count > best_match_count_current:
            best_match_count_current = current_match_count
            best_prompt = new_prompt
            best_results_current = current_results
            logger.info(f"更新最优提示词，匹配数: {current_match_count}/{total_true_count}, 召回率: {current_recall:.2%}")
        else:
            logger.info(f"新提示词召回率: {current_match_count}/{total_true_count} ({current_recall:.2%}) (未超过当前最优)")

        # 将新提示词设为下一次迭代的当前提示词
        current_prompt = new_prompt

    # 打印所有提示词及检测结果
    logger.info("所有提示词列表:")
    for idx, p in enumerate(iter_prompts_detail):
        recall_info = f"召回率:{p['match_count']/total_true_count:.2%}" if total_true_count > 0 else "召回率:0%"
        err_info = f" [{p.get('error_type')}]" if p.get("error_type") else ""
        logger.info(f"  [{idx}] {recall_info}{err_info}: {p['prompt']}")

    # 第二阶段：针对未识别目标的二次优化
    if best_match_count_current < total_true_count:
        logger.info("第二阶段：针对未识别目标进行二次优化...")

        # 绘制绿红标记图像
        marked_images = draw_match_results_on_images(all_data, best_results_current)

        # 构建检测结果分析字符串
        detection_analysis_lines = []
        for res in best_results_current:
            matched = res["match_count"]
            unmatched = len(res["unmatched_gt_indices"])
            detection_analysis_lines.append(
                f"图像{res['filename']}: 已识别{matched}个, 未识别{unmatched}个"
            )
        detection_analysis = "\n".join(detection_analysis_lines)

        # ========== 接口4: infer - 二次优化（针对未识别目标） ==========
        # 使用focus_unmatched_template进行二次优化
        prompt_for_refine = focus_unmatched_template.format(
            prompt=best_prompt,
            class_names=class_name,
            human_experience=human_exp_str,
            detection_analysis=detection_analysis
        )
        logger.info("=" * 80)
        logger.info("[接口调用] infer - 二次优化（针对未识别目标）")
        logger.info(f"[输入] template: focus_unmatched_template")
        logger.info(f"[输入] best_prompt: {best_prompt[:200]}...")
        logger.info(f"[输入] detection_analysis: {detection_analysis[:200]}...")
        logger.info(f"[输入] marked_images: {len(marked_images)} 张")
        logger.info(f"[输入] isdets: False")

        refined_prompt, _ = infer(prompt_for_refine, marked_images, isdets=False)

        logger.info(f"[输出] refined_prompt: {refined_prompt[:200] if refined_prompt else 'None'}...")
        print(f"二次优化后的提示词: {refined_prompt}")

        # ========== 接口5: evaluate_prompt_on_images - 验证二次优化后的提示词 ==========
        print("验证二次优化后的提示词...")
        logger.info("=" * 80)
        logger.info("[接口调用] evaluate_prompt_on_images - 验证二次优化提示词")
        logger.info(f"[输入] refined_prompt: {refined_prompt[:200]}...")
        logger.info(f"[输入] all_data: {len(all_data)} 张图像")
        refined_match_count, _, refined_results, _, _ = evaluate_prompt_on_images(refined_prompt, all_data)
        logger.info(f"[输出] refined_match_count: {refined_match_count}")
        logger.info(f"[输出] refined_results: 每张图检测结果（见下方）")
        for r in refined_results:
            logger.info(f"  - {r['filename']}: true={r['true_count']}, match={r['match_count']}, det={r['det_count']}")
        refined_recall = refined_match_count / total_true_count if total_true_count > 0 else 0

        # 记录二次优化结果
        iter_prompts_detail.append({
            "prompt": refined_prompt,
            "match_count": refined_match_count,
            "results": refined_results,
            "error_type": None
        })

        # 检查召回率是否下降
        if refined_match_count < best_match_count_current:
            print(f"⚠️ 二次优化后召回率下降: {best_match_count_current/total_true_count:.2%} -> {refined_recall:.2%}")
            print("二次优化后的提示词被拒绝")
        elif refined_match_count >= best_match_count_current:
            best_prompt = refined_prompt
            best_match_count_current = refined_match_count
            best_results_current = refined_results
            print(f"二次优化后的提示词效果更佳，更新最优提示词")

    print(f"\n{'='*60}")
    print(f"最优提示词（匹配数: {best_match_count_current}/{total_true_count}, 召回率: {best_match_count_current/total_true_count:.2%}）: {best_prompt}")
    return best_prompt, iter_prompts_detail


if __name__ == "__main__":
    # 加载所有图像数据
    all_data = load_all_images_data(src_path)

    if not all_data:
        logger.warning("未找到任何图像数据，退出")
        exit(1)

    # 执行多图共同优化
    best_prompt, all_prompts = self_optimize(all_data, max_iter=5)
    logger.info(f"\n最终最优提示词: {best_prompt}")

    # 保存最优提示词到prompt.json
    prompt_file = os.path.join(SCRIPT_DIR, "prompt.json")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    prompt_data["optim_det"] = best_prompt
    with open(prompt_file, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=4)
    logger.info(f"最优提示词已保存到 prompt.json['optim_det']")
