"""
ai_services.py
===============
AI 服务封装模块
"""

import numpy as np
from http import HTTPStatus
import dashscope
from dashscope import TextEmbedding
from openai import OpenAI
import streamlit as st


class AliyunEmbedder:
    """阿里云文本嵌入服务"""

    def __init__(self, api_key: str):
        """
        初始化嵌入器

        Args:
            api_key: 阿里云 API Key
        """
        self.api_key = api_key
        dashscope.api_key = api_key

    def encode(self, texts: list) -> np.ndarray:
        """
        编码文本列表为向量

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: 向量数组
        """
        if not texts:
            return np.array([])

        vectors = []
        for text in texts:
            vec = self._encode_single(text)
            if vec is not None:
                vectors.append(vec)

        return np.array(vectors) if vectors else np.array([])

    def embed_texts(self, texts: list) -> list:
        """
        批量编码文本（兼容方法名）

        Args:
            texts: 文本列表

        Returns:
            list: 向量列表
        """
        vectors = self.encode(texts)
        return vectors.tolist() if len(vectors) > 0 else []

    def _encode_single(self, text: str) -> list:
        """编码单个文本"""
        try:
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v1,
                input=text
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
        return None


def llm_normalize_user_input(user_input: str, client: OpenAI) -> str:
    """
    使用 LLM 标准化用户输入

    Args:
        user_input: 用户输入文本
        client: OpenAI 客户端

    Returns:
        str: 标准化后的文本
    """
    if not user_input or not user_input.strip():
        return user_input

    prompt = f"""请将以下茶评描述标准化为更连贯、完整的文本，保留所有关键信息：
原文：{user_input}

要求：
1. 保持原意不变
2. 使语言更连贯流畅
3. 保留所有感官描述词汇
4. 不要添加原文没有的信息

标准化后的文本："""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的茶评文本标准化助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] LLM normalization failed: {e}")
        return user_input