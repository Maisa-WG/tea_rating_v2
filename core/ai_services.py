"""
ai_services.py
===============
AI 服务封装模块（GraphRAG 增强版）

新增：
- init_graphrag_graph(): 初始化 GraphRAG 图谱的辅助函数
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
        self.api_key = api_key
        dashscope.api_key = api_key

    def encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([])
        vectors = []
        for text in texts:
            vec = self._encode_single(text)
            if vec is not None:
                vectors.append(vec)
        return np.array(vectors) if vectors else np.array([])

    def embed_texts(self, texts: list) -> list:
        vectors = self.encode(texts)
        return vectors.tolist() if len(vectors) > 0 else []

    def _encode_single(self, text: str) -> list:
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
    """使用 LLM 标准化用户输入"""
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


def init_graphrag_graph(kb_data=None, llm_client=None, model="deepseek-chat", cache_dir=""):
    """
    初始化 GraphRAG 茶评感官图谱

    Args:
        kb_data: 知识库文本列表（可选，用于 LLM 扩充图谱）
        llm_client: OpenAI 兼容客户端（可选，用于 LLM 抽取和社区摘要）
        model: LLM 模型名称
        cache_dir: 图谱缓存目录

    Returns:
        TeaSensoryGraph 实例
    """
    try:
        from retrieval.graphrag_retriever import get_or_create_graph

        # 准备知识库文本
        kb_texts = []
        if kb_data:
            if isinstance(kb_data, list):
                for item in kb_data[:20]:
                    if isinstance(item, str):
                        kb_texts.append(item)
                    elif isinstance(item, dict):
                        kb_texts.append(item.get("text", ""))

        graph = get_or_create_graph(
            kb_texts=kb_texts if kb_texts else None,
            llm_client=llm_client,
            model=model,
            cache_dir=cache_dir
        )

        print(f"[GraphRAG] 图谱初始化成功: "
              f"{graph.graph.number_of_nodes()} 节点, "
              f"{graph.graph.number_of_edges()} 边, "
              f"{len(graph.communities)} 社区")

        return graph

    except Exception as e:
        print(f"[ERROR] GraphRAG 初始化失败: {e}")
        import traceback
        traceback.print_exc()

        # 退化：返回仅种子图的图谱
        try:
            from retrieval.graphrag_retriever import TeaSensoryGraph
            graph = TeaSensoryGraph()
            graph.build_seed_graph()
            # 填充默认社区摘要
            for stage_name in graph.communities:
                if not graph.communities[stage_name].summary:
                    graph.communities[stage_name].summary = graph._default_community_summary(stage_name)
            print(f"[GraphRAG] 降级为种子图: {graph.graph.number_of_nodes()} 节点")
            return graph
        except Exception as e2:
            print(f"[ERROR] 种子图也失败了: {e2}")
            return None
