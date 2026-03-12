"""
vector_store.py
===============
向量索引和嵌入存储模块
"""

import json
import numpy as np
import os

import dashscope
from dashscope import TextEmbedding

from .database import DATA_DIR


# ==========================================
# 向量存储路径
# ==========================================

PATH_VECTORS = os.path.join(DATA_DIR, "vectors.npy")
PATH_VECTOR_META = os.path.join(DATA_DIR, "vectors_meta.json")


# ==========================================
# 嵌入函数
# ==========================================

def get_embedding(text):
    """
    调用阿里云生成 Embedding (单条)

    Args:
        text: 输入文本

    Returns:
        list: 嵌入向量，失败返回 None
    """
    try:
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            input=text
        )
        if resp.status_code == 200:
            return resp.output.embeddings[0].embedding
        else:
            print(f"Embedding API Error: {resp}")
            return None
    except Exception as e:
        print(f"Embedding Exception: {e}")
        return None


# ==========================================
# 向量索引管理
# ==========================================

def refresh_vector_index(all_cases):
    """
    全量/增量刷新向量库
    逻辑：遍历所有案例 -> 检查是否有向量 -> 没有则计算 -> 保存

    Args:
        all_cases: pd.DataFrame，所有案例数据

    Returns:
        bool: 是否成功
    """
    # 1. 加载旧数据
    if os.path.exists(PATH_VECTORS) and os.path.exists(PATH_VECTOR_META):
        try:
            vectors = np.load(PATH_VECTORS)
            with open(PATH_VECTOR_META, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except:
            vectors = np.array([])
            meta = []
    else:
        vectors = None
        meta = []

    # 2. 找出新数据 (简单的去重逻辑)
    existing_reviews = set([m.get('input_review', '') for m in meta])

    new_vectors = []
    new_meta = []
    dirty = False

    # 遍历传入的 DataFrame
    for _, row in all_cases.iterrows():
        txt = str(row.get('input_review', ''))
        # 如果文本太短或已存在，跳过
        if len(txt) < 2 or txt in existing_reviews:
            continue

        # 计算新向量
        emb = get_embedding(txt)
        if emb:
            new_vectors.append(emb)
            # 记录元数据
            r_dict = row.to_dict()
            new_meta.append(r_dict)
            dirty = True

    # 3. 合并保存
    if dirty:
        if vectors is not None and len(vectors) > 0:
            final_vectors = np.vstack([vectors, np.array(new_vectors)])
            final_meta = meta + new_meta
        else:
            final_vectors = np.array(new_vectors)
            final_meta = new_meta

        np.save(PATH_VECTORS, final_vectors)
        with open(PATH_VECTOR_META, 'w', encoding='utf-8') as f:
            json.dump(final_meta, f, ensure_ascii=False)
        print(f"向量库已更新，当前共 {len(final_meta)} 条。")

    return True


def load_vector_store():
    """
    读取向量库

    Returns:
        tuple: (vectors: np.ndarray, meta: list)
    """
    if os.path.exists(PATH_VECTORS) and os.path.exists(PATH_VECTOR_META):
        try:
            vectors = np.load(PATH_VECTORS)
            with open(PATH_VECTOR_META, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return vectors, meta
        except:
            return None, []
    return None, []