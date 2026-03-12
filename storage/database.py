"""
database.py
===========
CSV 数据存储模块
"""

import json
import pandas as pd
import os
import time

# ==========================================
# 路径与字段定义
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 定义两个数据库文件名
FILE_INITIAL = "initial_case.csv"   # 原始判例库
FILE_ADJUSTED = "adjusted_case.csv" # 修正后的数据
FILE_MANUAL = "knowledge_manual.txt" # 行业权威手册

PATH_INITIAL = os.path.join(DATA_DIR, FILE_INITIAL)
PATH_ADJUSTED = os.path.join(DATA_DIR, FILE_ADJUSTED)
PATH_MANUAL = os.path.join(DATA_DIR, FILE_MANUAL)

# 定义统一的字段格式
COLUMNS_SCHEMA = [
    "name", "type", "input_review", "input_context", "expert_summary", "timestamp",
    "score_优雅性", "reason_优雅性",
    "score_辨识度", "reason_辨识度",
    "score_协调性", "reason_协调性",
    "score_饱和度", "reason_饱和度",
    "score_持久性", "reason_持久性",
    "score_苦涩度", "reason_苦涩度"
]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# ==========================================
# 核心功能函数
# ==========================================

def _init_csv_if_not_exists(file_path):
    """初始化指定的 CSV 文件"""
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=COLUMNS_SCHEMA)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')


def load_all_cases():
    """
    读取逻辑：同时读取【原始库】和【修正库】，合并后供 AI 参考 (RAG)

    Returns:
        pd.DataFrame: 合并后的判例数据
    """
    _init_csv_if_not_exists(PATH_INITIAL)
    _init_csv_if_not_exists(PATH_ADJUSTED)

    dfs = []
    for p in [PATH_INITIAL, PATH_ADJUSTED]:
        try:
            # 尝试读取，如果文件损坏或为空则跳过
            d = pd.read_csv(p, encoding='utf-8-sig')
            # 简单的列校验，防止空文件报错
            if not d.empty:
                dfs.append(d)
        except Exception:
            pass

    if dfs:
        # 合并两个数据源，reset_index 防止索引冲突
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=COLUMNS_SCHEMA)


def flatten_case_data(nested_data):
    """
    将嵌套字典拍平为符合 Schema 的格式

    Args:
        nested_data: 嵌套结构的判例数据

    Returns:
        dict: 扁平化的单行数据
    """
    # 基础字段
    flat_row = {
        "name": nested_data.get("name", "未命名"),
        "type": nested_data.get("type", "通用"),
        "input_review": nested_data.get("input_review", ""),
        "input_context": nested_data.get("input_context", ""),
        "expert_summary": nested_data.get("expert_summary", ""),
        "timestamp": nested_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    }

    # 分数与理由 (动态提取，防止 Key 报错)
    scores = nested_data.get("scores", {})
    reasons = nested_data.get("reasons", {})

    factors = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
    for f in factors:
        flat_row[f"score_{f}"] = scores.get(f, 0.0)
        flat_row[f"reason_{f}"] = reasons.get(f, "")

    return flat_row


def insert_case(case_data_dict, target="adjusted"):
    """
    插入数据

    Args:
        case_data_dict: 判例数据字典
        target: "initial" (原始库) 或 "adjusted" (修正库)

    Returns:
        tuple: (success: bool, message: str)
    """
    # 1. 确定目标文件
    if target == "initial":
        target_path = PATH_INITIAL
    else:
        target_path = PATH_ADJUSTED

    try:
        # 2. 初始化文件确保表头存在
        _init_csv_if_not_exists(target_path)

        # 3. 数据扁平化处理
        flat_row = flatten_case_data(case_data_dict)

        # 4. 读取旧数据并追加
        try:
            current_df = pd.read_csv(target_path, encoding='utf-8-sig')
        except:
            current_df = pd.DataFrame(columns=COLUMNS_SCHEMA)

        new_row_df = pd.DataFrame([flat_row])

        # 严格按照 COLUMNS_SCHEMA 排序
        new_row_df = new_row_df.reindex(columns=COLUMNS_SCHEMA).fillna("")

        updated_df = pd.concat([current_df, new_row_df], ignore_index=True)

        # 5. 写入
        updated_df.to_csv(target_path, index=False, encoding='utf-8-sig')

        print(f"✅ 数据已保存至 {target}: {target_path}")
        return True, f"保存成功 (库: {target})"

    except Exception as e:
        return False, f"保存失败: {str(e)}"


# ==========================================
# JSON 兼容性（可选保留）
# ==========================================

KB_FILE = os.path.join(DATA_DIR, "tea_knowledge_base.json")


def load_json_kb():
    """加载 JSON 知识库"""
    if not os.path.exists(KB_FILE):
        return []
    try:
        with open(KB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_json_kb(record):
    """保存记录到 JSON 知识库"""
    data = load_json_kb()
    data.append(record)
    with open(KB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)