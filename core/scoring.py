"""
scoring.py
===========
核心评分逻辑模块（GraphRAG 集成版）

主要变更：
1. 集成 GraphRAG 管线：感官词提取 → 图扩展 → 因子映射 → 社区摘要
2. 在 user_prompt 中注入 GraphRAG 上下文（5个标记段）
3. 返回 debug_payload，供 UI 展示图检索调试信息
4. 保持原有 FAISS 检索逻辑不变
"""

import json
import faiss
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

from config.constants import FACTORS


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
    c_num: int = 5,
    graphrag_graph=None,
    graphrag_llm_client=None,
    graphrag_model: str = "deepseek-chat",
    designed_shape: Optional[Dict[str, float]] = None
) -> Tuple[Dict, str, str, str, str]:
    """
    执行评分逻辑（GraphRAG 增强版）

    Args:
        user_input: 用户输入的茶评描述
        kb: 知识库 (index, data)
        basic_cases: 基础判例列表
        supp_cases: 进阶判例 (index, data)
        prompt_config: 提示词配置
        embedder: 嵌入器
        client: OpenAI 客户端（主评分模型）
        model_id: 模型 ID
        r_num: 参考知识库条目数量
        c_num: 参考进阶判例条目数量
        graphrag_graph: TeaSensoryGraph 实例（可选）
        graphrag_llm_client: 用于 GraphRAG 抽取的 LLM 客户端
        graphrag_model: GraphRAG 抽取使用的模型
        designed_shape: 设计形状（理想分数）

    Returns:
        Tuple: (scores, kb_history, case_history, sys_prompt, user_prompt)
              其中 scores 字典增加了 "graphrag_debug" 键
    """
    kb_idx, kb_data = kb
    supp_idx, supp_data = supp_cases

    sys_prompt = prompt_config.get('system_template', '')
    user_tpl = prompt_config.get('user_template', '')

    # ========================================
    # Phase A: 原有 FAISS 向量检索（保持不变）
    # ========================================

    # 1. 从知识库检索相关内容
    kb_context = ""
    kb_history = ""
    if kb_data and len(kb_data) > 0:
        try:
            query_vec = embedder.encode([user_input])
            if len(query_vec) > 0:
                D, I = kb_idx.search(query_vec, min(r_num, len(kb_data)))
                kb_chunks = [kb_data[i] for i in I[0] if i < len(kb_data)]
                kb_context = "\n\n".join(kb_chunks)
                kb_history = f"参考了 {len(kb_chunks)} 条知识库内容"
        except Exception as e:
            print(f"[ERROR] KB search failed: {e}")

    # 2. 从进阶判例检索相似案例
    case_context = ""
    case_history = ""
    if supp_data and len(supp_data) > 0:
        try:
            query_vec = embedder.encode([user_input])

            if not isinstance(query_vec, np.ndarray):
                query_vec = np.array(query_vec)
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)

            if query_vec.shape[1] != supp_idx.d:
                print(f"[WARNING] 判例索引维度不匹配：索引 {supp_idx.d}，查询 {query_vec.shape[1]}")

                if 'rebuilt_supp_idx' in st.session_state:
                    print(f"[INFO] 使用缓存的新索引（维度: {st.session_state.rebuilt_supp_idx.d}）")
                    new_idx = st.session_state.rebuilt_supp_idx
                    D, I = new_idx.search(query_vec, min(c_num, len(supp_data)))
                    similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                    case_context = _format_cases_for_prompt(similar_cases)
                    case_history = f"参考了 {len(similar_cases)} 条相似判例（缓存索引）"
                else:
                    print(f"[INFO] 重新编码判例数据...")
                    start_time = time.time()
                    batch_size = 8
                    all_embeddings = []

                    for i in range(0, len(supp_data), batch_size):
                        batch_texts = [item["text"] for item in supp_data[i:i+batch_size]]
                        max_retries = 3
                        batch_embeddings = None
                        for retry in range(max_retries):
                            try:
                                batch_embeddings = embedder.encode(batch_texts)
                                break
                            except Exception as e:
                                if retry < max_retries - 1:
                                    time.sleep(2 ** retry)
                                else:
                                    batch_embeddings = np.zeros((len(batch_texts), query_vec.shape[1]))
                        if batch_embeddings is not None:
                            all_embeddings.append(batch_embeddings)
                        time.sleep(0.5)

                    all_embeddings = np.vstack(all_embeddings)
                    if not isinstance(all_embeddings, np.ndarray):
                        all_embeddings = np.array(all_embeddings)
                    if len(all_embeddings.shape) == 1:
                        all_embeddings = all_embeddings.reshape(1, -1)

                    new_idx = faiss.IndexFlatL2(all_embeddings.shape[1])
                    new_idx.add(all_embeddings.astype('float32'))
                    st.session_state.supp_cases = (new_idx, supp_data)
                    st.session_state.rebuilt_supp_idx = new_idx

                    try:
                        from config.settings import PATHS
                        from core.resource_manager import ResourceManager
                        ResourceManager.save(new_idx, supp_data, PATHS.supp_case_index, PATHS.supp_case_data, is_json=True)
                    except Exception as e:
                        print(f"[WARNING] 索引保存失败: {e}")

                    D, I = new_idx.search(query_vec, min(c_num, len(supp_data)))
                    similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                    case_context = _format_cases_for_prompt(similar_cases)
                    case_history = f"参考了 {len(similar_cases)} 条相似判例（索引已重建）"

            elif len(query_vec) > 0:
                D, I = supp_idx.search(query_vec, min(c_num, len(supp_data)))
                similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                case_context = _format_cases_for_prompt(similar_cases)
                case_history = f"参考了 {len(similar_cases)} 条相似判例"
        except Exception as e:
            print(f"[ERROR] Case search failed: {e}")
            case_history = "判例检索失败"

    # 3. 格式化基础判例
    basic_context = _format_basic_cases(basic_cases)

    # ========================================
    # Phase B: GraphRAG 管线（新增）
    # ========================================

    graphrag_context = ""
    graphrag_debug = {}

    if graphrag_graph is not None:
        try:
            print("[GraphRAG] 开始执行 GraphRAG 管线...")

            # 运行完整管线
            graphrag_debug = graphrag_graph.run_graphrag_pipeline(
                review_text=user_input,
                llm_client=graphrag_llm_client,
                model=graphrag_model,
                designed_shape=designed_shape
            )

            # 格式化为 Prompt 段落
            graphrag_context = graphrag_graph.format_graphrag_context_for_prompt(graphrag_debug)

            print(f"[GraphRAG] 管线完成: 提取 {len(graphrag_debug.get('extracted_terms', []))} 个感官词, "
                  f"overlap={graphrag_debug.get('overlap_score', 0)}")

        except Exception as e:
            print(f"[ERROR] GraphRAG pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            graphrag_context = "（GraphRAG 检索失败，请依赖常规知识库评分）"
            graphrag_debug = {"error": str(e)}

    # ========================================
    # Phase C: 构建完整 Prompt
    # ========================================

    # 在 user_prompt 中注入 GraphRAG 上下文
    if graphrag_context:
        user_prompt = user_tpl.format(
            product_desc=user_input,
            context_text=kb_context,
            basic_case_text=basic_context,
            case_text=case_context
        )
        # 在 user_prompt 末尾追加 GraphRAG 上下文
        user_prompt += f"\n\n====================\n【GraphRAG 感官图谱分析结果】\n====================\n{graphrag_context}"
    else:
        user_prompt = user_tpl.format(
            product_desc=user_input,
            context_text=kb_context,
            basic_case_text=basic_context,
            case_text=case_context
        )

    # 回写 debug_payload 中的 prompt
    graphrag_debug["final_system_prompt"] = sys_prompt
    graphrag_debug["final_user_prompt"] = user_prompt

    # 保存提示词到 session state 供查看
    try:
        if hasattr(st, 'session_state'):
            st.session_state.last_llm_sys_prompt = sys_prompt
            st.session_state.last_llm_user_prompt = user_prompt
            st.session_state.last_graphrag_debug = graphrag_debug
    except:
        pass

    # ========================================
    # Phase D: 调用 LLM 评分
    # ========================================

    max_retries = 2
    timeout = 120

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=timeout
            )

            content = response.choices[0].message.content
            scores = _parse_llm_response(content)

            # 将 graphrag_debug 附加到 scores 中
            if scores and isinstance(scores, dict):
                scores["graphrag_debug"] = graphrag_debug

            return scores, kb_history, case_history, sys_prompt, user_prompt

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] LLM call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")

            if attempt == max_retries - 1 or "timeout" not in error_msg.lower():
                st.error(f"评分失败: {error_msg}")
                if "timeout" in error_msg.lower():
                    st.warning("💡 模型响应时间过长，建议：\n1. 检查网络连接\n2. 减少输入内容\n3. 联系管理员检查模型服务状态")
                return None, "", "", sys_prompt, user_prompt

            print(f"[INFO] Retrying due to timeout...")
            continue


def _format_cases_for_prompt(cases: List[Dict]) -> str:
    """格式化判例用于提示词"""
    if not cases:
        return "暂无相似判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')[:200]
        scores = case.get('scores', {})
        score_str = ", ".join([f"{k}:{v.get('score', 0)}" for k, v in scores.items()])
        parts.append(f"[判例{i}] {text}...\n得分: {score_str}")

    return "\n\n".join(parts)


def _format_basic_cases(cases: List[Dict]) -> str:
    """格式化基础判例"""
    if not cases:
        return "暂无基础判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')[:200]
        parts.append(f"[基础判例{i}] {text}...")

    return "\n\n".join(parts)


def _parse_llm_response(content: str) -> Dict:
    """解析 LLM 响应"""
    try:
        return json.loads(content)
    except:
        try:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
    return None
