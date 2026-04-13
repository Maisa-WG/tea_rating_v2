"""
scoring.py
===========
核心评分逻辑模块
"""

import json
import re
import time
import logging
import numpy as np
from typing import Tuple, List, Dict, Any

import streamlit as st
from openai import OpenAI

from config.constants import FACTORS
from core.resource_manager import ResourceManager, DEFAULT_EMBEDDING_DIM

logger = logging.getLogger(__name__)



def _ensure_2d_float_array(vectors: Any) -> np.ndarray:
    """把 embedding 结果统一转为二维 float32 numpy。"""
    arr = np.array(vectors, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr



def run_scoring(
    user_input: str,
    kb: Tuple,
    basic_cases: List[Dict],
    supp_cases: Tuple,
    prompt_config: Dict,
    embedder,
    client: OpenAI,
    model_id: str,
    r_num: int = 3,
    c_num: int = 5
) -> Tuple[Dict, str, str, str, str]:
    """
    执行评分逻辑。

    Args:
        user_input: 用户输入的茶评描述
        kb: 知识库 (index, data)
        basic_cases: 基础判例列表
        supp_cases: 进阶判例 (index, data)
        prompt_config: 提示词配置
        embedder: 嵌入器
        client: OpenAI 客户端
        model_id: 模型 ID
        r_num: 参考知识库条目数量
        c_num: 参考进阶判例条目数量

    Returns:
        Tuple: (scores, kb_history, case_history, sys_prompt, user_prompt)
    """
    kb_idx, kb_data = kb
    supp_idx, supp_data = supp_cases

    sys_prompt = prompt_config.get('system_template', '')
    user_tpl = prompt_config.get('user_template', '')

    query_vec = None

    # 1. 从知识库检索相关内容
    kb_context = ""
    kb_history = ""
    if kb_idx is not None and kb_data and len(kb_data) > 0:
        try:
            query_vec = _ensure_2d_float_array(embedder.encode([user_input]))
            if query_vec.size > 0 and getattr(kb_idx, 'd', query_vec.shape[1]) == query_vec.shape[1]:
                _, indices = kb_idx.search(query_vec, min(r_num, len(kb_data)))
                kb_chunks = [kb_data[i] for i in indices[0] if 0 <= int(i) < len(kb_data)]
                kb_context = "\n\n".join(kb_chunks)
                kb_history = f"参考了 {len(kb_chunks)} 条知识库内容"
            elif query_vec.size > 0:
                logger.warning(
                    f"知识库索引维度不匹配：索引 {getattr(kb_idx, 'd', 'unknown')}，查询 {query_vec.shape[1]}"
                )
                kb_history = "知识库索引维度不匹配，已跳过本次知识检索"
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            kb_history = "知识库检索失败"

    # 2. 从进阶判例检索相似案例
    case_context = ""
    case_history = ""
    if supp_data and len(supp_data) > 0:
        try:
            if query_vec is None or query_vec.size == 0:
                query_vec = _ensure_2d_float_array(embedder.encode([user_input]))

            if query_vec.size > 0:
                idx_invalid = (
                    supp_idx is None
                    or getattr(supp_idx, 'd', DEFAULT_EMBEDDING_DIM) != query_vec.shape[1]
                    or getattr(supp_idx, 'ntotal', 0) != len(supp_data)
                )

                if idx_invalid:
                    logger.warning("进阶判例索引缺失 / 维度不匹配 / 数量不匹配，正在按缓存向量重建...")
                    supp_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
                    st.session_state.supp_cases = (supp_idx, supp_data)
                    logger.info("✅ 进阶判例索引已按缓存向量重建")

                _, indices = supp_idx.search(query_vec, min(c_num, len(supp_data)))
                similar_cases = [supp_data[i] for i in indices[0] if 0 <= int(i) < len(supp_data)]
                case_context = _format_cases_for_prompt(similar_cases)
                case_history = f"参考了 {len(similar_cases)} 条相似判例"
        except Exception as e:
            logger.error(f"判例检索失败: {e}")
            case_history = "判例检索失败"

    # 3. 格式化基础判例
    basic_context = _format_basic_cases(basic_cases)

    # 4. 构建完整提示词
    user_prompt = user_tpl.format(
        product_desc=user_input,
        context_text=kb_context,
        basic_case_text=basic_context,
        case_text=case_context
    )

    # 保存提示词到 session state 供查看
    try:
        if hasattr(st, 'session_state'):
            st.session_state.last_llm_sys_prompt = sys_prompt
            st.session_state.last_llm_user_prompt = user_prompt
    except Exception:
        pass

    # 5. 调用 LLM（带超时和重试机制）
    max_retries = 2
    timeout = 60

    # 判断是否为 DeepSeek-reasoner（不支持 temperature / response_format / system role）
    is_reasoner = (model_id == "deepseek-reasoner")

    for attempt in range(max_retries):
        try:
            if is_reasoner:
                # DeepSeek-reasoner: 不支持 temperature、response_format，
                # 且 system 消息需要放在 user 消息中
                combined_prompt = (
                    f"[系统指令]\n{sys_prompt}\n\n"
                    f"[用户输入]\n{user_prompt}\n\n"
                    "请严格以 JSON 格式返回结果，不要包含任何其他内容。"
                )
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": combined_prompt}
                    ],
                    max_tokens=4000,
                    timeout=timeout,
                )
            else:
                # Qwen-8B / DeepSeek-chat: 标准 OpenAI 兼容调用
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=timeout,
                    response_format={"type": "json_object"}
                )

            content = response.choices[0].message.content

            try:
                st.session_state.last_llm_raw_output = content
            except Exception:
                pass
            
            logger.warning(f"[LLM RAW OUTPUT] {repr(content)}")
            
            scores = _parse_llm_response(content)
            
            if scores is None:
                logger.error(f"[LLM PARSE FAILED] raw={repr(content)}")
            
            return scores, kb_history, case_history, sys_prompt, user_prompt


        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt == max_retries - 1 or "timeout" not in error_msg.lower():
                st.error(f"评分失败: {error_msg}")
                if "timeout" in error_msg.lower():
                    st.warning("💡 模型响应时间过长，建议：\n1. 检查网络连接\n2. 减少输入内容\n3. 联系管理员检查模型服务状态")
                return None, "", "", sys_prompt, user_prompt

            logger.info("由于超时，正在重试...")
            continue



def _format_cases_for_prompt(cases: List[Dict]) -> str:
    """格式化判例用于提示词。"""
    if not cases:
        return "暂无相似判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')
        scores = case.get('scores', {})
        score_str = ", ".join([f"{k}:{v.get('score', 0)}" for k, v in scores.items()])
        parts.append(f"[判例{i}] {text}\n得分: {score_str}")

    return "\n\n".join(parts)



def _format_basic_cases(cases: List[Dict]) -> str:
    """格式化基础判例。"""
    if not cases:
        return "暂无基础判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')
        scores = case.get('scores', {})
        score_str = ", ".join([f"{k}:{v.get('score', 0)}" for k, v in scores.items()])
        parts.append(f"[基础判例{i}] {text}\n得分: {score_str}")

    return "\n\n".join(parts)



def _parse_llm_response(content: str) -> Dict:
    """解析 LLM 响应。"""
    if content is None:
        return None

    if not isinstance(content, str):
        try:
            content = str(content)
        except Exception:
            return None

    text = content.strip()

    # 去掉 markdown 代码块
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 去掉 think 标签
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 先尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试抽取最外层 JSON
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return None

