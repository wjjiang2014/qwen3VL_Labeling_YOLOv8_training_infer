"""
测试区分两种JSON解析错误：
1. 模型输出了大量中间推理过程文字（不是JSON格式）
2. 大量误检导致输出未封闭（部分JSON但不完整）
"""
import json
import re

# 模拟两种错误情况
error_case_1 = """根据图像内容分析，图中存在一个符合您描述的可疑物品：

- **物品位置**：位于图像左侧，靠近站台边缘，停放在一个白色立柱旁。
- **物品特征**：一个黑色的拉杆行李箱，箱体完整，拉杆处于竖立状态，箱体上无任何明显标识或异常。

**坐标**：[203, 609, 285, 847]
"""

error_case_2 = """[{"bbox_2d": [203, 609, 285, 847], "label": "可疑行李箱"}, {"bbox_2d": [100, 200, 300, 400], "label": "另一个目标"}, {"bbox_2d": [50, 100, 150, 200], "label": "还有"""

error_case_3 = """[{"bbox_2d": [203, 609, 285, 847], "label": "可疑行李箱"}"""  # 未封闭

# 模拟用户遇到的真实错误案例 - 被截断的JSON（经过 ```json 处理后的）
error_case_real = """[{"bbox_2d": [609, 793, 666, 895], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [753, 812, 795, 915], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [876, 548, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600], "label": "无人看管、放置于公共场所、周围无明显人员看护的包裹或行李"},
        {"bbox_2d": [872, 557, 913, 600"""  # 被截断


def classify_json_error(raw_output: str) -> str:
    """
    分类JSON解析错误的类型（仅在JSON解析失败时调用）

    Returns:
        str: "reasoning_text" | "unclosed_json"
    """
    if not raw_output or not raw_output.strip():
        return "empty_output"

    stripped = raw_output.strip()

    # 检查是否以 [ 或 { 开头（应该是JSON）
    starts_with_bracket = stripped.startswith('[') or stripped.startswith('{')

    # 计算括号匹配情况
    bracket_count = stripped.count('[') - stripped.count(']')
    brace_count = stripped.count('{') - stripped.count('}')

    print(f"  stripped[:50]: {stripped[:50]}...")
    print(f"  starts_with_bracket: {starts_with_bracket}")
    print(f"  bracket_count: {bracket_count}, brace_count: {brace_count}")

    # 如果以 [ 或 { 开头，说明本应是JSON，只是被截断了
    if starts_with_bracket:
        # 括号不匹配 = 未封闭JSON
        if bracket_count != 0 or brace_count != 0:
            return "unclosed_json"
        # 括号匹配但仍然解析失败，可能是其他问题
        return "unclosed_json"

    # 不以 [ 或 { 开头，才是真正的推理过程文字
    # 检查是否包含明显的自然语言描述
    has_chinese_text = bool(re.search(r'[\u4e00-\u9fff].*[:：]', stripped))
    has_markdown = bool(re.search(r'\*\*|```|###|----', stripped))
    has_bullet_points = bool(re.search(r'^[-*•]|\n\s*[-*•]', stripped, re.MULTILINE))

    print(f"  has_chinese_text: {has_chinese_text}, has_markdown: {has_markdown}, has_bullet_points: {has_bullet_points}")

    # 有大量自然语言描述 = 推理过程文字
    if has_chinese_text or has_markdown or has_bullet_points:
        return "reasoning_text"

    # 其他未知错误当作推理过程文字处理
    return "reasoning_text"


if __name__ == "__main__":
    print("=== 测试 classify_json_error ===\n")

    test_cases = [
        ("推理过程文字", error_case_1, "reasoning_text"),
        ("未封闭JSON", error_case_2, "unclosed_json"),
        ("未封闭JSON(简单)", error_case_3, "unclosed_json"),
        ("真实错误案例(被截断JSON)", error_case_real, "unclosed_json"),
    ]

    for name, text, expected in test_cases:
        print(f"测试: {name}")
        print(f"  文本长度: {len(text)} 字符")
        print(f"  预期: {expected}")

        try:
            result = json.loads(text)
            print(f"  JSON解析成功（不应该发生）: {result}")
        except json.JSONDecodeError as e:
            error_type = classify_json_error(text)
            status = "✓" if error_type == expected else "✗"
            print(f"  {status} 分类结果: {error_type} (预期: {expected})")
            print(f"  解析错误: {e.msg} at pos {e.pos}")
        print()