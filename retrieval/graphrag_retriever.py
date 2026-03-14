"""
graphrag_retriever.py
======================
GraphRAG 茶评感官知识图谱检索模块（演示版）

核心功能：
1. 构图：六因子节点 + LLM 从知识库抽取感官词节点 + 关联边
2. 社区发现与摘要（Community Summaries）
3. 在线查询：提取评论感官词 → 图中 1-hop 扩展 → 因子映射
4. 形状 overlap：设计形状 vs 感知形状
5. debug_payload 全链路透传

Author: GraphRAG Tea Demo
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any, Set

try:
    import networkx as nx
except ImportError:
    nx = None

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

FACTORS = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]

# 因子 → 三段归属（用于社区分组）
FACTOR_STAGE = {
    "优雅性": "前段·香", "辨识度": "前段·香",
    "协调性": "中段·味", "饱和度": "中段·味",
    "持久性": "后段·韵", "苦涩度": "后段·韵",
}

# 三段社区定义
STAGE_COMMUNITIES = {
    "前段·香": {
        "id": "community_aroma",
        "factors": ["优雅性", "辨识度"],
        "summary": "",
    },
    "中段·味": {
        "id": "community_taste",
        "factors": ["协调性", "饱和度"],
        "summary": "",
    },
    "后段·韵": {
        "id": "community_aftertaste",
        "factors": ["持久性", "苦涩度"],
        "summary": "",
    },
}


@dataclass
class SensoryTerm:
    """感官描述词节点"""
    term: str
    factor: str           # 所属因子
    polarity: str         # "正向" / "负向" / "中性"
    weight: float = 1.0   # 关联强度 0-1
    source: str = ""      # 来源：kb / llm / seed


@dataclass
class GraphEdge:
    """图中的边"""
    source: str
    target: str
    relation: str   # "正向指标" / "负向指标" / "属于社区"
    weight: float = 1.0


@dataclass
class CommunityInfo:
    """社区信息"""
    community_id: str
    name: str
    factors: List[str]
    terms: List[str]
    summary: str


# ---------------------------------------------------------------------------
# LLM 感官词抽取器（从知识库 / 从评论）
# ---------------------------------------------------------------------------

# 用于从知识库批量抽取感官词的 Prompt
_KB_EXTRACT_PROMPT = """你是茶学感官分析专家。请从以下茶学知识库文本中，抽取与"罗马测评法六因子"相关的感官描述词/短语。

六因子定义：
- 优雅性：香气的愉悦感（正向如：清雅、幽香、花香；负向如：闷、杂、刺鼻）
- 辨识度：香气的可识别性（正向如：兰花香、蜜香、果香；负向如：平淡、糊味）
- 协调性：茶汤融合度（正向如：协调、圆润、平衡；负向如：割裂、突兀）
- 饱和度：茶汤浓厚度（正向如：饱满、浓厚、稠滑；负向如：寡淡、水薄）
- 持久性：余韵持续（正向如：回甘持久、喉韵长；负向如：散得快、余味短）
- 苦涩度：苦涩舒适度（正向/高分如：不苦不涩、微苦即化；负向/低分如：苦涩重、锁喉）

知识库文本：
{kb_text}

请严格输出 JSON 数组，每个元素格式如下（不要输出其他内容）：
[
  {{"term": "清雅", "factor": "优雅性", "polarity": "正向", "weight": 0.9}},
  {{"term": "锁喉", "factor": "苦涩度", "polarity": "负向", "weight": 0.95}},
  ...
]
"""

# 用于从用户评论中实时抽取感官词的 Prompt
_REVIEW_EXTRACT_PROMPT = """你是茶学感官分析专家。请从以下茶评文本中，提取所有感官描述词/短语，并判断每个词对应的六因子归属。

六因子：优雅性、辨识度、协调性、饱和度、持久性、苦涩度

茶评文本：
"{review_text}"

请严格输出 JSON 数组（不要输出其他任何内容，不要 Markdown）：
[
  {{"term": "清甜", "factor": "优雅性", "polarity": "正向"}},
  {{"term": "花香", "factor": "辨识度", "polarity": "正向"}},
  ...
]

注意：
1. 只提取文本中实际出现的词，不要编造
2. 一个词可能同时关联多个因子，请分别列出
3. polarity 只能是 "正向" / "负向" / "中性"
"""

# 用于生成社区摘要的 Prompt
_COMMUNITY_SUMMARY_PROMPT = """你是茶学感官分析专家。以下是"{stage_name}"维度下的感官描述词集合，这些词都与"{factor_names}"因子相关。

感官词列表：
{terms_text}

请用一段话（50-80字）总结消费者通常如何用这些词描述该维度的体验。
要求：
- 体现正向词与负向词的对比
- 语言简洁专业
- 不要分点，直接输出一段话
"""


def _safe_parse_json_array(text: str) -> List[Dict]:
    """安全解析 JSON 数组，兼容 Markdown 包裹"""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # 尝试提取 [...]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return []


def _safe_parse_text(text: str) -> str:
    """安全提取纯文本"""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 种子感官词表（兜底 + 演示保障）
# ---------------------------------------------------------------------------

_SEED_TERMS: List[Dict] = [
    # ---- 优雅性 ----
    {"term": "清雅", "factor": "优雅性", "polarity": "正向", "weight": 0.9},
    {"term": "幽香", "factor": "优雅性", "polarity": "正向", "weight": 0.85},
    {"term": "花香", "factor": "优雅性", "polarity": "正向", "weight": 0.8},
    {"term": "清甜", "factor": "优雅性", "polarity": "正向", "weight": 0.8},
    {"term": "高级", "factor": "优雅性", "polarity": "正向", "weight": 0.85},
    {"term": "干净", "factor": "优雅性", "polarity": "正向", "weight": 0.75},
    {"term": "杂味", "factor": "优雅性", "polarity": "负向", "weight": 0.9},
    {"term": "闷", "factor": "优雅性", "polarity": "负向", "weight": 0.85},
    {"term": "刺鼻", "factor": "优雅性", "polarity": "负向", "weight": 0.9},
    {"term": "霉味", "factor": "优雅性", "polarity": "负向", "weight": 0.95},
    {"term": "柔和", "factor": "优雅性", "polarity": "正向", "weight": 0.75},
    {"term": "愉悦", "factor": "优雅性", "polarity": "正向", "weight": 0.85},

    # ---- 辨识度 ----
    {"term": "兰花香", "factor": "辨识度", "polarity": "正向", "weight": 0.9},
    {"term": "蜜香", "factor": "辨识度", "polarity": "正向", "weight": 0.85},
    {"term": "果香", "factor": "辨识度", "polarity": "正向", "weight": 0.85},
    {"term": "桂花香", "factor": "辨识度", "polarity": "正向", "weight": 0.9},
    {"term": "松烟香", "factor": "辨识度", "polarity": "正向", "weight": 0.85},
    {"term": "木质香", "factor": "辨识度", "polarity": "正向", "weight": 0.8},
    {"term": "桂圆干香", "factor": "辨识度", "polarity": "正向", "weight": 0.85},
    {"term": "野花蜜甜", "factor": "辨识度", "polarity": "正向", "weight": 0.8},
    {"term": "陈香", "factor": "辨识度", "polarity": "正向", "weight": 0.75},
    {"term": "毫香", "factor": "辨识度", "polarity": "正向", "weight": 0.8},
    {"term": "平淡", "factor": "辨识度", "polarity": "负向", "weight": 0.8},
    {"term": "糊味", "factor": "辨识度", "polarity": "负向", "weight": 0.85},
    {"term": "香精感", "factor": "辨识度", "polarity": "负向", "weight": 0.9},

    # ---- 协调性 ----
    {"term": "协调", "factor": "协调性", "polarity": "正向", "weight": 0.9},
    {"term": "平衡", "factor": "协调性", "polarity": "正向", "weight": 0.9},
    {"term": "圆润", "factor": "协调性", "polarity": "正向", "weight": 0.85},
    {"term": "融合", "factor": "协调性", "polarity": "正向", "weight": 0.85},
    {"term": "顺口", "factor": "协调性", "polarity": "正向", "weight": 0.8},
    {"term": "割裂", "factor": "协调性", "polarity": "负向", "weight": 0.9},
    {"term": "突兀", "factor": "协调性", "polarity": "负向", "weight": 0.85},
    {"term": "失衡", "factor": "协调性", "polarity": "负向", "weight": 0.9},
    {"term": "分离", "factor": "协调性", "polarity": "负向", "weight": 0.85},

    # ---- 饱和度 ----
    {"term": "饱满", "factor": "饱和度", "polarity": "正向", "weight": 0.9},
    {"term": "浓厚", "factor": "饱和度", "polarity": "正向", "weight": 0.9},
    {"term": "醇厚", "factor": "饱和度", "polarity": "正向", "weight": 0.9},
    {"term": "稠滑", "factor": "饱和度", "polarity": "正向", "weight": 0.85},
    {"term": "丰富", "factor": "饱和度", "polarity": "正向", "weight": 0.8},
    {"term": "顺滑", "factor": "饱和度", "polarity": "正向", "weight": 0.8},
    {"term": "寡淡", "factor": "饱和度", "polarity": "负向", "weight": 0.9},
    {"term": "水薄", "factor": "饱和度", "polarity": "负向", "weight": 0.9},
    {"term": "空", "factor": "饱和度", "polarity": "负向", "weight": 0.85},

    # ---- 持久性 ----
    {"term": "回甘持久", "factor": "持久性", "polarity": "正向", "weight": 0.9},
    {"term": "喉韵", "factor": "持久性", "polarity": "正向", "weight": 0.85},
    {"term": "余香", "factor": "持久性", "polarity": "正向", "weight": 0.85},
    {"term": "生津", "factor": "持久性", "polarity": "正向", "weight": 0.8},
    {"term": "回甘", "factor": "持久性", "polarity": "正向", "weight": 0.85},
    {"term": "留香", "factor": "持久性", "polarity": "正向", "weight": 0.8},
    {"term": "余韵", "factor": "持久性", "polarity": "正向", "weight": 0.8},
    {"term": "散得快", "factor": "持久性", "polarity": "负向", "weight": 0.85},
    {"term": "余味短", "factor": "持久性", "polarity": "负向", "weight": 0.85},

    # ---- 苦涩度（高分=不苦不涩=正向）----
    {"term": "不苦不涩", "factor": "苦涩度", "polarity": "正向", "weight": 0.95},
    {"term": "微苦即化", "factor": "苦涩度", "polarity": "正向", "weight": 0.85},
    {"term": "涩感轻", "factor": "苦涩度", "polarity": "正向", "weight": 0.8},
    {"term": "苦涩重", "factor": "苦涩度", "polarity": "负向", "weight": 0.95},
    {"term": "锁喉", "factor": "苦涩度", "polarity": "负向", "weight": 0.95},
    {"term": "拉扯", "factor": "苦涩度", "polarity": "负向", "weight": 0.9},
    {"term": "收敛", "factor": "苦涩度", "polarity": "负向", "weight": 0.85},
    {"term": "涩感", "factor": "苦涩度", "polarity": "负向", "weight": 0.8},
    {"term": "粘腻", "factor": "苦涩度", "polarity": "负向", "weight": 0.8},
]


# ---------------------------------------------------------------------------
# 核心类：TeaSensoryGraph（茶评感官知识图谱）
# ---------------------------------------------------------------------------

class TeaSensoryGraph:
    """
    茶评六因子感官知识图谱

    图结构：
    - 因子节点（6个）：优雅性、辨识度、协调性、饱和度、持久性、苦涩度
    - 感官词节点（N个）：清雅、花香、醇厚、锁喉 ...
    - 边：感官词 --[正向指标/负向指标]--> 因子
    - 社区：前段·香、中段·味、后段·韵（各包含2个因子及其关联词）
    """

    def __init__(self):
        if nx is None:
            raise ImportError("请安装 networkx: pip install networkx")
        self.graph = nx.DiGraph()
        self.terms: Dict[str, SensoryTerm] = {}         # term_text -> SensoryTerm
        self.communities: Dict[str, CommunityInfo] = {}  # community_id -> CommunityInfo
        self._initialized = False

    # ============================
    # 1. 构图
    # ============================

    def build_seed_graph(self) -> None:
        """用种子词表构建基础图（保障演示可用）"""
        # 添加因子节点
        for f in FACTORS:
            self.graph.add_node(f, node_type="factor", stage=FACTOR_STAGE[f])

        # 添加种子感官词节点和边
        for item in _SEED_TERMS:
            self._add_sensory_term(
                term=item["term"],
                factor=item["factor"],
                polarity=item["polarity"],
                weight=item.get("weight", 1.0),
                source="seed"
            )

        # 构建社区
        self._build_communities()
        self._initialized = True
        print(f"[GraphRAG] 种子图构建完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")

    def build_from_kb_via_llm(self, kb_texts: List[str], llm_client, model: str = "deepseek-chat") -> int:
        """
        通过 LLM 从知识库文本中抽取感官词并扩充图谱

        Args:
            kb_texts: 知识库文本片段列表
            llm_client: OpenAI 兼容客户端
            model: 模型名称

        Returns:
            新增感官词数量
        """
        if not self._initialized:
            self.build_seed_graph()

        added_count = 0
        # 合并短文本，减少 LLM 调用次数
        merged_texts = []
        buf = ""
        for t in kb_texts:
            if len(buf) + len(t) > 2000:
                if buf:
                    merged_texts.append(buf)
                buf = t
            else:
                buf += "\n" + t
        if buf:
            merged_texts.append(buf)

        for i, chunk in enumerate(merged_texts[:10]):  # 最多处理10个块
            try:
                prompt = _KB_EXTRACT_PROMPT.format(kb_text=chunk[:3000])
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是茶学感官分析专家，只输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                    timeout=30
                )
                content = response.choices[0].message.content
                items = _safe_parse_json_array(content)

                for item in items:
                    term = item.get("term", "").strip()
                    factor = item.get("factor", "").strip()
                    polarity = item.get("polarity", "中性").strip()
                    weight = float(item.get("weight", 0.8))

                    if term and factor in FACTORS and term not in self.terms:
                        self._add_sensory_term(term, factor, polarity, weight, source="llm_kb")
                        added_count += 1

                print(f"[GraphRAG] KB chunk {i+1}/{len(merged_texts)}: 抽取 {len(items)} 个词")

            except Exception as e:
                print(f"[GraphRAG] KB extraction failed for chunk {i+1}: {e}")
                continue

        # 重建社区
        self._build_communities()
        print(f"[GraphRAG] LLM 扩充完成: 新增 {added_count} 个感官词, 图谱共 {self.graph.number_of_nodes()} 节点")
        return added_count

    def generate_community_summaries_via_llm(self, llm_client, model: str = "deepseek-chat") -> None:
        """通过 LLM 生成社区摘要"""
        for stage_name, comm_info in self.communities.items():
            if comm_info.summary:  # 已有则跳过
                continue
            try:
                terms_text = ", ".join(comm_info.terms[:30])
                factor_names = "、".join(comm_info.factors)
                prompt = _COMMUNITY_SUMMARY_PROMPT.format(
                    stage_name=stage_name,
                    factor_names=factor_names,
                    terms_text=terms_text
                )
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是茶学感官分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    timeout=20
                )
                summary = _safe_parse_text(response.choices[0].message.content)
                comm_info.summary = summary
                print(f"[GraphRAG] 社区摘要生成: {stage_name}")

            except Exception as e:
                print(f"[GraphRAG] 社区摘要生成失败 ({stage_name}): {e}")
                # 使用默认摘要
                comm_info.summary = self._default_community_summary(stage_name)

        # 对没有 LLM 摘要的社区使用默认值
        for stage_name, comm_info in self.communities.items():
            if not comm_info.summary:
                comm_info.summary = self._default_community_summary(stage_name)

    def _add_sensory_term(self, term: str, factor: str, polarity: str, weight: float, source: str) -> None:
        """添加一个感官词节点和边"""
        if factor not in FACTORS:
            return
        st = SensoryTerm(term=term, factor=factor, polarity=polarity, weight=weight, source=source)
        self.terms[term] = st

        self.graph.add_node(term, node_type="sensory_term", factor=factor, polarity=polarity, source=source)

        relation = "正向指标" if polarity == "正向" else ("负向指标" if polarity == "负向" else "中性指标")
        self.graph.add_edge(term, factor, relation=relation, weight=weight, polarity=polarity)

    def _build_communities(self) -> None:
        """基于三段分组构建社区"""
        self.communities.clear()
        for stage_name, stage_def in STAGE_COMMUNITIES.items():
            factors = stage_def["factors"]
            # 收集该社区下的所有感官词
            terms_in_community = []
            for t_name, t_obj in self.terms.items():
                if t_obj.factor in factors:
                    terms_in_community.append(t_name)

            self.communities[stage_name] = CommunityInfo(
                community_id=stage_def["id"],
                name=stage_name,
                factors=factors,
                terms=sorted(terms_in_community),
                summary=stage_def.get("summary", "")
            )

    def _default_community_summary(self, stage_name: str) -> str:
        """生成默认社区摘要"""
        comm = self.communities.get(stage_name)
        if not comm:
            return ""
        pos_terms = [t for t in comm.terms if self.terms.get(t, SensoryTerm("", "", "")).polarity == "正向"][:5]
        neg_terms = [t for t in comm.terms if self.terms.get(t, SensoryTerm("", "", "")).polarity == "负向"][:5]
        factors_str = "、".join(comm.factors)

        summary = f"{stage_name}社区覆盖{factors_str}两个因子。"
        if pos_terms:
            summary += f"消费者常用"{'"、"'.join(pos_terms)}"等正向词汇描述良好体验"
        if neg_terms:
            summary += f"，用"{'"、"'.join(neg_terms)}"等词汇描述缺陷"
        summary += "。"
        return summary

    # ============================
    # 2. 在线查询
    # ============================

    def extract_terms_from_review(self, review_text: str, llm_client=None, model: str = "deepseek-chat") -> List[Dict]:
        """
        从用户评论中提取感官词

        优先使用 LLM 提取，失败则退化为图谱词表匹配

        Returns:
            [{"term": "清甜", "factor": "优雅性", "polarity": "正向", "matched_in_graph": True}, ...]
        """
        extracted = []

        # 方式 1: LLM 提取
        if llm_client:
            try:
                prompt = _REVIEW_EXTRACT_PROMPT.format(review_text=review_text[:1000])
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是茶学感官分析专家，只输出JSON数组。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=20
                )
                content = response.choices[0].message.content
                items = _safe_parse_json_array(content)

                for item in items:
                    term = item.get("term", "").strip()
                    factor = item.get("factor", "").strip()
                    polarity = item.get("polarity", "中性").strip()
                    if term and factor in FACTORS:
                        in_graph = term in self.terms or term in self.graph
                        extracted.append({
                            "term": term,
                            "factor": factor,
                            "polarity": polarity,
                            "matched_in_graph": in_graph,
                            "source": "llm_extract"
                        })

                if extracted:
                    print(f"[GraphRAG] LLM 从评论抽取 {len(extracted)} 个感官词")
                    return extracted

            except Exception as e:
                print(f"[GraphRAG] LLM 评论抽取失败: {e}, 降级为词表匹配")

        # 方式 2: 词表匹配（兜底）
        for term_text, term_obj in self.terms.items():
            if term_text in review_text:
                extracted.append({
                    "term": term_text,
                    "factor": term_obj.factor,
                    "polarity": term_obj.polarity,
                    "matched_in_graph": True,
                    "source": "graph_match"
                })

        print(f"[GraphRAG] 词表匹配从评论中找到 {len(extracted)} 个感官词")
        return extracted

    def expand_1hop(self, matched_terms: List[Dict]) -> Dict[str, Any]:
        """
        对匹配到的感官词进行 1-hop 邻域扩展

        Returns:
            {
                "graph_hits": [...],           # 命中的图节点
                "neighborhood": [...],         # 1-hop 扩展结果
                "factor_hit_count": {...},     # 每个因子被命中的次数
                "expanded_terms": [...]        # 扩展发现的额外感官词
            }
        """
        graph_hits = []
        neighborhood = []
        factor_hit_count = {f: 0 for f in FACTORS}
        expanded_terms = []
        visited = set()

        for item in matched_terms:
            term = item["term"]
            factor = item["factor"]

            # 记录直接命中
            graph_hits.append({
                "node": term,
                "type": "sensory_term",
                "direct_factor": factor,
                "polarity": item.get("polarity", "中性")
            })
            factor_hit_count[factor] = factor_hit_count.get(factor, 0) + 1
            visited.add(term)

            # 1-hop: 从该词出发的所有邻居
            if self.graph.has_node(term):
                # 出边：感官词 -> 因子
                for _, neighbor, data in self.graph.out_edges(term, data=True):
                    edge_info = {
                        "from": term,
                        "to": neighbor,
                        "relation": data.get("relation", ""),
                        "weight": data.get("weight", 1.0)
                    }
                    neighborhood.append(edge_info)

                # 入边：查看是否有其他词也指向相同因子（同社区扩展）
                if self.graph.has_node(factor):
                    for predecessor, _, data in self.graph.in_edges(factor, data=True):
                        if predecessor not in visited and predecessor != term:
                            visited.add(predecessor)
                            expanded_terms.append({
                                "term": predecessor,
                                "factor": factor,
                                "relation": data.get("relation", ""),
                                "via": f"同因子扩展({factor})"
                            })

        # 限制扩展词数量
        expanded_terms = expanded_terms[:15]

        return {
            "graph_hits": graph_hits,
            "neighborhood": neighborhood,
            "factor_hit_count": factor_hit_count,
            "expanded_terms": expanded_terms
        }

    def map_to_factors(self, matched_terms: List[Dict]) -> Dict[str, Dict]:
        """
        将提取的感官词映射为六因子感知值

        Returns:
            {
                "优雅性": {"score_hint": 7.0, "positive_terms": [...], "negative_terms": [...], "evidence_count": 3},
                ...
            }
        """
        factor_mapping = {}
        for f in FACTORS:
            factor_mapping[f] = {
                "positive_terms": [],
                "negative_terms": [],
                "neutral_terms": [],
                "evidence_count": 0,
                "score_hint": 5.0  # 默认中性
            }

        for item in matched_terms:
            factor = item["factor"]
            if factor not in factor_mapping:
                continue
            term = item["term"]
            polarity = item.get("polarity", "中性")
            fm = factor_mapping[factor]

            if polarity == "正向":
                fm["positive_terms"].append(term)
            elif polarity == "负向":
                fm["negative_terms"].append(term)
            else:
                fm["neutral_terms"].append(term)
            fm["evidence_count"] += 1

        # 计算 score_hint（简单启发式）
        for f, fm in factor_mapping.items():
            pos = len(fm["positive_terms"])
            neg = len(fm["negative_terms"])
            total = pos + neg + len(fm["neutral_terms"])

            if total == 0:
                fm["score_hint"] = 4.0  # 无证据 → 保守分
            else:
                # 正向越多分越高，负向越多分越低
                ratio = (pos - neg) / total  # [-1, 1]
                fm["score_hint"] = round(5.0 + ratio * 4.0, 1)  # [1.0, 9.0]
                fm["score_hint"] = max(1.0, min(9.0, fm["score_hint"]))

        return factor_mapping

    def get_community_summaries(self, matched_terms: List[Dict]) -> List[Dict]:
        """
        获取被命中因子所在社区的摘要

        Returns:
            [{"community_id": "...", "name": "...", "summary": "...", "factors": [...]}, ...]
        """
        hit_factors = set(item["factor"] for item in matched_terms)
        summaries = []
        seen = set()

        for stage_name, comm in self.communities.items():
            if any(f in hit_factors for f in comm.factors):
                if comm.community_id not in seen:
                    seen.add(comm.community_id)
                    summaries.append({
                        "community_id": comm.community_id,
                        "name": comm.name,
                        "summary": comm.summary or self._default_community_summary(stage_name),
                        "factors": comm.factors,
                        "terms_count": len(comm.terms)
                    })

        return summaries

    # ============================
    # 3. 形状 Overlap
    # ============================

    def compute_shape_overlap(
        self,
        perceived_mapping: Dict[str, Dict],
        designed_shape: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        计算感知形状与设计形状的 overlap

        Args:
            perceived_mapping: map_to_factors 的输出
            designed_shape: 设计形状（理想分数），默认全6分

        Returns:
            {
                "perceived_shape": {"优雅性": 7.0, ...},
                "designed_shape": {"优雅性": 6.0, ...},
                "overlap_score": 0.85,
                "factor_gaps": {"优雅性": +1.0, ...}
            }
        """
        if designed_shape is None:
            designed_shape = {f: 6.0 for f in FACTORS}

        perceived_shape = {}
        for f in FACTORS:
            perceived_shape[f] = perceived_mapping.get(f, {}).get("score_hint", 4.0)

        # Cosine similarity
        p_vec = [perceived_shape[f] for f in FACTORS]
        d_vec = [designed_shape[f] for f in FACTORS]

        dot = sum(a * b for a, b in zip(p_vec, d_vec))
        norm_p = math.sqrt(sum(a * a for a in p_vec)) or 1e-10
        norm_d = math.sqrt(sum(a * a for a in d_vec)) or 1e-10
        cosine_sim = dot / (norm_p * norm_d)

        # 各因子差距
        factor_gaps = {}
        for f in FACTORS:
            factor_gaps[f] = round(perceived_shape[f] - designed_shape[f], 1)

        return {
            "perceived_shape": perceived_shape,
            "designed_shape": designed_shape,
            "overlap_score": round(cosine_sim, 4),
            "factor_gaps": factor_gaps
        }

    # ============================
    # 4. 主入口：完整 GraphRAG Pipeline
    # ============================

    def run_graphrag_pipeline(
        self,
        review_text: str,
        llm_client=None,
        model: str = "deepseek-chat",
        designed_shape: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的 GraphRAG 检索管线

        Returns:
            debug_payload 包含所有中间结果
        """
        if not self._initialized:
            self.build_seed_graph()

        # Step 1: 提取感官词
        extracted_terms = self.extract_terms_from_review(review_text, llm_client, model)

        # Step 2: 1-hop 邻域扩展
        expansion = self.expand_1hop(extracted_terms)

        # Step 3: 因子映射
        factor_mapping = self.map_to_factors(extracted_terms)

        # Step 4: 社区摘要
        community_summaries = self.get_community_summaries(extracted_terms)

        # Step 5: 形状 Overlap
        overlap = self.compute_shape_overlap(factor_mapping, designed_shape)

        # 组装 debug_payload
        debug_payload = {
            "extracted_terms": extracted_terms,
            "graph_hits": expansion["graph_hits"],
            "neighborhood_expansion": expansion["neighborhood"],
            "expanded_terms": expansion["expanded_terms"],
            "factor_hit_count": expansion["factor_hit_count"],
            "community_summaries": community_summaries,
            "factor_mapping": {
                f: {
                    "score_hint": fm["score_hint"],
                    "positive_terms": fm["positive_terms"],
                    "negative_terms": fm["negative_terms"],
                    "evidence_count": fm["evidence_count"]
                } for f, fm in factor_mapping.items()
            },
            "overlap_score": overlap["overlap_score"],
            "perceived_shape": overlap["perceived_shape"],
            "designed_shape": overlap["designed_shape"],
            "factor_gaps": overlap["factor_gaps"],
            "graph_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "factor_nodes": len(FACTORS),
                "sensory_term_nodes": len(self.terms),
                "communities": len(self.communities)
            }
        }

        return debug_payload

    # ============================
    # 5. Prompt 组装辅助
    # ============================

    def format_graphrag_context_for_prompt(self, debug_payload: Dict) -> str:
        """
        将 GraphRAG 结果格式化为 Prompt 段落

        输出包含五个标记段：
        【Extracted Sensory Terms】
        【1-hop Graph Expansion】
        【Community Summaries】
        【Factor Mapping Result】
        【Perceived Shape / Designed Shape Overlap】
        """
        parts = []

        # --- Extracted Sensory Terms ---
        terms = debug_payload.get("extracted_terms", [])
        terms_str = ""
        if terms:
            for t in terms:
                flag = "✓图谱命中" if t.get("matched_in_graph") else "○新发现"
                terms_str += f"  - {t['term']} → {t['factor']}（{t['polarity']}）[{flag}]\n"
        else:
            terms_str = "  （未提取到感官描述词）\n"
        parts.append(f"【Extracted Sensory Terms】\n{terms_str}")

        # --- 1-hop Graph Expansion ---
        hits = debug_payload.get("graph_hits", [])
        neighborhood = debug_payload.get("neighborhood_expansion", [])
        expanded = debug_payload.get("expanded_terms", [])
        expansion_str = ""
        if hits:
            expansion_str += "直接命中节点：\n"
            for h in hits[:10]:
                expansion_str += f"  - {h['node']} ({h['type']}) → {h['direct_factor']} [{h['polarity']}]\n"
        if neighborhood:
            expansion_str += "1-hop 边扩展：\n"
            for e in neighborhood[:10]:
                expansion_str += f"  - {e['from']} --[{e['relation']}]--> {e['to']} (权重:{e['weight']})\n"
        if expanded:
            expansion_str += "同因子扩展词：\n"
            for ex in expanded[:8]:
                expansion_str += f"  - {ex['term']} → {ex['factor']} ({ex['via']})\n"
        if not expansion_str:
            expansion_str = "  （无图扩展结果）\n"
        parts.append(f"【1-hop Graph Expansion】\n{expansion_str}")

        # --- Community Summaries ---
        comms = debug_payload.get("community_summaries", [])
        comm_str = ""
        if comms:
            for c in comms:
                comm_str += f"  [{c['name']}] (因子: {'、'.join(c['factors'])}, 词汇量: {c['terms_count']})\n"
                comm_str += f"    摘要: {c['summary']}\n"
        else:
            comm_str = "  （未命中任何社区）\n"
        parts.append(f"【Community Summaries】\n{comm_str}")

        # --- Factor Mapping Result ---
        fm = debug_payload.get("factor_mapping", {})
        fm_str = ""
        for f in FACTORS:
            info = fm.get(f, {})
            hint = info.get("score_hint", 4.0)
            pos = info.get("positive_terms", [])
            neg = info.get("negative_terms", [])
            count = info.get("evidence_count", 0)
            fm_str += f"  {f}: 预估={hint}, 证据数={count}"
            if pos:
                fm_str += f", 正向=[{'、'.join(pos)}]"
            if neg:
                fm_str += f", 负向=[{'、'.join(neg)}]"
            fm_str += "\n"
        parts.append(f"【Factor Mapping Result】\n{fm_str}")

        # --- Perceived Shape / Designed Shape Overlap ---
        perceived = debug_payload.get("perceived_shape", {})
        designed = debug_payload.get("designed_shape", {})
        overlap = debug_payload.get("overlap_score", 0.0)
        gaps = debug_payload.get("factor_gaps", {})
        shape_str = f"  Overlap Score (cosine): {overlap}\n"
        shape_str += "  感知形状 vs 设计形状:\n"
        for f in FACTORS:
            p_val = perceived.get(f, 4.0)
            d_val = designed.get(f, 6.0)
            gap = gaps.get(f, 0.0)
            arrow = "↑" if gap > 0 else ("↓" if gap < 0 else "=")
            shape_str += f"    {f}: 感知={p_val} / 设计={d_val} ({arrow}{abs(gap)})\n"
        parts.append(f"【Perceived Shape / Designed Shape Overlap】\n{shape_str}")

        return "\n".join(parts)

    # ============================
    # 6. 持久化
    # ============================

    def save(self, out_dir: str) -> None:
        """保存图谱到文件"""
        os.makedirs(out_dir, exist_ok=True)

        # 保存图的边
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})
        with open(os.path.join(out_dir, "graph_edges.json"), "w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)

        # 保存节点
        nodes = []
        for n, data in self.graph.nodes(data=True):
            nodes.append({"id": n, **data})
        with open(os.path.join(out_dir, "graph_nodes.json"), "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)

        # 保存社区
        comms = {}
        for name, c in self.communities.items():
            comms[name] = asdict(c)
        with open(os.path.join(out_dir, "communities.json"), "w", encoding="utf-8") as f:
            json.dump(comms, f, ensure_ascii=False, indent=2)

        # 保存感官词表
        terms = {t: asdict(obj) for t, obj in self.terms.items()}
        with open(os.path.join(out_dir, "sensory_terms.json"), "w", encoding="utf-8") as f:
            json.dump(terms, f, ensure_ascii=False, indent=2)

        print(f"[GraphRAG] 图谱已保存到 {out_dir}")

    def load(self, in_dir: str) -> bool:
        """从文件加载图谱"""
        try:
            edges_path = os.path.join(in_dir, "graph_edges.json")
            nodes_path = os.path.join(in_dir, "graph_nodes.json")
            comms_path = os.path.join(in_dir, "communities.json")
            terms_path = os.path.join(in_dir, "sensory_terms.json")

            if not os.path.exists(edges_path):
                return False

            self.graph = nx.DiGraph()

            # 加载节点
            if os.path.exists(nodes_path):
                with open(nodes_path, "r", encoding="utf-8") as f:
                    nodes = json.load(f)
                for n in nodes:
                    nid = n.pop("id")
                    self.graph.add_node(nid, **n)

            # 加载边
            with open(edges_path, "r", encoding="utf-8") as f:
                edges = json.load(f)
            for e in edges:
                src = e.pop("source")
                tgt = e.pop("target")
                self.graph.add_edge(src, tgt, **e)

            # 加载感官词
            if os.path.exists(terms_path):
                with open(terms_path, "r", encoding="utf-8") as f:
                    terms_data = json.load(f)
                for t, obj in terms_data.items():
                    self.terms[t] = SensoryTerm(**obj)

            # 加载社区
            if os.path.exists(comms_path):
                with open(comms_path, "r", encoding="utf-8") as f:
                    comms_data = json.load(f)
                for name, c in comms_data.items():
                    self.communities[name] = CommunityInfo(**c)

            self._initialized = True
            print(f"[GraphRAG] 图谱加载成功: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
            return True

        except Exception as e:
            print(f"[GraphRAG] 图谱加载失败: {e}")
            return False


# ---------------------------------------------------------------------------
# 便捷入口函数
# ---------------------------------------------------------------------------

def get_or_create_graph(
    kb_texts: Optional[List[str]] = None,
    llm_client=None,
    model: str = "deepseek-chat",
    cache_dir: str = "",
) -> TeaSensoryGraph:
    """
    获取或创建茶评感官图谱（带缓存）
    """
    graph = TeaSensoryGraph()

    # 尝试从缓存加载
    if cache_dir and graph.load(cache_dir):
        return graph

    # 构建种子图
    graph.build_seed_graph()

    # LLM 扩充
    if kb_texts and llm_client:
        try:
            graph.build_from_kb_via_llm(kb_texts, llm_client, model)
        except Exception as e:
            print(f"[GraphRAG] LLM 扩充失败，使用种子图: {e}")

    # 生成社区摘要
    if llm_client:
        try:
            graph.generate_community_summaries_via_llm(llm_client, model)
        except Exception as e:
            print(f"[GraphRAG] 社区摘要生成失败，使用默认: {e}")
            # 确保有默认摘要
            for stage_name in graph.communities:
                if not graph.communities[stage_name].summary:
                    graph.communities[stage_name].summary = graph._default_community_summary(stage_name)
    else:
        # 无 LLM 时使用默认摘要
        for stage_name in graph.communities:
            graph.communities[stage_name].summary = graph._default_community_summary(stage_name)

    # 保存缓存
    if cache_dir:
        try:
            graph.save(cache_dir)
        except Exception as e:
            print(f"[GraphRAG] 缓存保存失败: {e}")

    return graph
