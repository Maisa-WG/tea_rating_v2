"""
GraphRAG-enhanced retrieval module (static knowledge base)

This module is designed to be plugged into your existing FAISS-based RAG pipeline.
It adds a lightweight GraphRAG layer on top of vector retrieval:

1) Offline indexing:
   - Chunk documents
   - Extract entity-relation triples (LLM-assisted or rule-based)
   - Build a graph (nodes=entities, edges=relations) and attach chunk ids
   - Discover communities and generate community summaries (LLM-assisted)
   - Persist: graph.jsonl, communities.json, chunk_meta.jsonl

2) Online retrieval:
   - Vector retrieve Top-K seed chunks (FAISS)
   - Expand candidates by graph neighborhood + community membership
   - Optional rerank (hybrid score = vector_sim + graph_score)
   - Return: seed chunks + expanded chunks + community summaries

NOTE:
- For Chinese domain KBs, entity/relation extraction is best done via an LLM batch job.
- This file includes a default extractor interface; you can bind it to deepseek-chat.

Author: (generated)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Any, Set

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str = ""
    tags: Dict[str, str] = None


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    chunk_id: str


@dataclass
class Community:
    community_id: str
    nodes: List[str]
    chunk_ids: List[str]
    summary: str = ""


# -----------------------------
# LLM-assisted extractor (pluggable)
# -----------------------------

class EntityRelationExtractor:
    """
    Pluggable extractor.
    Bind `extract_triples(text)` to:
      - an LLM call (recommended), or
      - a rule-based method (baseline)
    Output triples should be concise and consistent across the corpus.
    """
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        raise NotImplementedError


class RuleBasedExtractor(EntityRelationExtractor):
    """
    Very lightweight baseline: extract pseudo-entities by delimiter heuristics.
    For real use, replace with LLM-assisted extraction.
    """
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        # Heuristic: treat terms inside 《》 or “” as entities, relate by "关联"
        ents: List[str] = []
        for left, right in [("《", "》"), ("“", "”"), ("\"", "\"")]:
            start = 0
            while True:
                i = text.find(left, start)
                if i == -1:
                    break
                j = text.find(right, i + 1)
                if j == -1:
                    break
                ent = text[i + 1 : j].strip()
                if ent:
                    ents.append(ent)
                start = j + 1
        # Build naive triples among entities
        triples = []
        for i in range(len(ents) - 1):
            triples.append((ents[i], "关联", ents[i + 1]))
        return triples


# -----------------------------
# GraphRAG Indexer (offline)
# -----------------------------

class GraphRAGIndexer:
    def __init__(self, extractor: Optional[EntityRelationExtractor] = None):
        if nx is None:
            raise ImportError("networkx is required for GraphRAGIndexer. Please `pip install networkx`.")
        self.extractor = extractor or RuleBasedExtractor()
        self.graph = nx.MultiDiGraph()
        self.chunk_meta: Dict[str, Dict[str, Any]] = {}
        self.triples: List[Triple] = []
        self.communities: List[Community] = []

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self.chunk_meta[c.chunk_id] = {"source": c.source, "tags": c.tags or {}}
            triples = self.extractor.extract_triples(c.text)
            for h, r, t in triples:
                self._add_triple(h, r, t, c.chunk_id)

    def _add_triple(self, head: str, relation: str, tail: str, chunk_id: str) -> None:
        head = head.strip()
        tail = tail.strip()
        relation = relation.strip() or "相关"
        if not head or not tail:
            return
        self.graph.add_node(head)
        self.graph.add_node(tail)
        self.graph.add_edge(head, tail, relation=relation, chunk_id=chunk_id)
        self.triples.append(Triple(head=head, relation=relation, tail=tail, chunk_id=chunk_id))

    def build_communities(self, min_size: int = 5) -> List[Community]:
        """
        Community discovery on the undirected projection (greedy modularity).
        For large graphs you may switch to Louvain.
        """
        if self.graph.number_of_nodes() == 0:
            self.communities = []
            return self.communities

        undirected = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            undirected.add_edge(u, v)

        from networkx.algorithms.community import greedy_modularity_communities

        comms = list(greedy_modularity_communities(undirected))
        communities: List[Community] = []
        for idx, nodes in enumerate(comms):
            nodes_list = sorted(list(nodes))
            if len(nodes_list) < min_size:
                continue
            chunk_ids: Set[str] = set()
            for u in nodes_list:
                for _, _, data in self.graph.out_edges(u, data=True):
                    chunk_ids.add(str(data.get("chunk_id", "")))
                for _, _, data in self.graph.in_edges(u, data=True):
                    chunk_ids.add(str(data.get("chunk_id", "")))
            communities.append(Community(
                community_id=f"c{idx}",
                nodes=nodes_list,
                chunk_ids=sorted([cid for cid in chunk_ids if cid]),
                summary=""
            ))
        self.communities = communities
        return communities

    def attach_community_summaries(self, summaries: Dict[str, str]) -> None:
        """
        Provide summaries generated by an external batch job (LLM).
        summaries: {community_id: summary_text}
        """
        for c in self.communities:
            if c.community_id in summaries:
                c.summary = summaries[c.community_id]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # graph edges
        graph_path = os.path.join(out_dir, "graph_edges.jsonl")
        with open(graph_path, "w", encoding="utf-8") as f:
            for tr in self.triples:
                f.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")

        # node list
        nodes_path = os.path.join(out_dir, "graph_nodes.json")
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.graph.nodes())), f, ensure_ascii=False, indent=2)

        # communities
        comm_path = os.path.join(out_dir, "communities.json")
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.communities], f, ensure_ascii=False, indent=2)

        # chunk meta
        meta_path = os.path.join(out_dir, "chunk_meta.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for cid, meta in self.chunk_meta.items():
                row = {"chunk_id": cid, **meta}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# GraphRAG Retriever (online)
# -----------------------------

class GraphRAGRetriever:
    """
    Online hybrid retriever.

    Inputs:
      - `vector_hits`: List[(chunk_id, score)] from FAISS (higher is better)
      - `chunk_text_map`: Dict[chunk_id -> text]
      - graph artifacts: out_dir from GraphRAGIndexer.save()
    """
    def __init__(self, artifact_dir: str):
        if nx is None:
            raise ImportError("networkx is required for GraphRAGRetriever. Please `pip install networkx`.")
        self.artifact_dir = artifact_dir
        self.graph = nx.MultiDiGraph()
        self.communities: Dict[str, Community] = {}
        self.node2communities: Dict[str, Set[str]] = {}
        self.chunk2nodes: Dict[str, Set[str]] = {}
        self._load()

    def _load(self) -> None:
        edges_path = os.path.join(self.artifact_dir, "graph_edges.jsonl")
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                tr = json.loads(line)
                h, r, t, cid = tr["head"], tr["relation"], tr["tail"], tr["chunk_id"]
                self.graph.add_node(h)
                self.graph.add_node(t)
                self.graph.add_edge(h, t, relation=r, chunk_id=cid)
                self.chunk2nodes.setdefault(cid, set()).update([h, t])

        comm_path = os.path.join(self.artifact_dir, "communities.json")
        if os.path.exists(comm_path):
            comms = json.load(open(comm_path, "r", encoding="utf-8"))
            for c in comms:
                com = Community(**c)
                self.communities[com.community_id] = com
                for n in com.nodes:
                    self.node2communities.setdefault(n, set()).add(com.community_id)

    def expand(self,
               vector_hits: List[Tuple[str, float]],
               chunk_text_map: Dict[str, str],
               top_seed: int = 5,
               hop: int = 1,
               max_expand: int = 12,
               w_vec: float = 0.7,
               w_graph: float = 0.3) -> Dict[str, Any]:
        """
        Return hybrid context:
          - seeds: top vector chunks
          - expanded: graph-augmented chunks
          - community_summaries: matched community summaries
        """
        seeds = vector_hits[:top_seed]
        seed_chunk_ids = [cid for cid, _ in seeds]

        # Collect seed nodes
        seed_nodes: Set[str] = set()
        for cid in seed_chunk_ids:
            seed_nodes |= self.chunk2nodes.get(cid, set())

        # Neighborhood expansion
        cand_chunks: Set[str] = set(seed_chunk_ids)
        frontier = set(seed_nodes)
        visited_nodes = set(seed_nodes)
        for _ in range(max(1, hop)):
            next_frontier = set()
            for n in frontier:
                # outgoing and incoming neighbors
                for _, v, data in self.graph.out_edges(n, data=True):
                    next_frontier.add(v)
                    cand_chunks.add(str(data.get("chunk_id", "")))
                for u, _, data in self.graph.in_edges(n, data=True):
                    next_frontier.add(u)
                    cand_chunks.add(str(data.get("chunk_id", "")))
            next_frontier -= visited_nodes
            visited_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        cand_chunks = {c for c in cand_chunks if c}
        # Limit expansion size (keep seeds + best graph-score)
        # Graph-score: count of shared nodes with seed neighborhood
        scored: List[Tuple[str, float]] = []
        for cid in cand_chunks:
            nodes = self.chunk2nodes.get(cid, set())
            gscore = len(nodes & visited_nodes) / max(1, len(nodes))
            scored.append((cid, gscore))

        # Normalize vector scores among candidates
        vec_score_map = {cid: s for cid, s in vector_hits}
        vec_vals = [vec_score_map.get(cid, 0.0) for cid, _ in scored]
        vmin, vmax = (min(vec_vals), max(vec_vals)) if vec_vals else (0.0, 1.0)
        def norm(x: float) -> float:
            if vmax <= vmin:
                return 0.0
            return (x - vmin) / (vmax - vmin)

        merged: List[Tuple[str, float, float, float]] = []
        for cid, gscore in scored:
            v = norm(vec_score_map.get(cid, 0.0))
            final = w_vec * v + w_graph * gscore
            merged.append((cid, final, v, gscore))

        merged.sort(key=lambda x: x[1], reverse=True)

        # keep seeds first, then top expanded
        final_chunks: List[str] = []
        seen = set()
        for cid, _, _, _ in merged:
            if cid in seed_chunk_ids and cid not in seen:
                final_chunks.append(cid); seen.add(cid)
        for cid, _, _, _ in merged:
            if cid not in seen:
                final_chunks.append(cid); seen.add(cid)
            if len(final_chunks) >= top_seed + max_expand:
                break

        # Communities touched by visited_nodes
        comm_ids: Set[str] = set()
        for n in visited_nodes:
            comm_ids |= self.node2communities.get(n, set())
        comm_summaries = []
        for cid in sorted(list(comm_ids)):
            com = self.communities.get(cid)
            if com and com.summary:
                comm_summaries.append({"community_id": cid, "summary": com.summary})

        return {
            "seed_chunks": [{"chunk_id": cid, "score": float(s), "text": chunk_text_map.get(cid, "")} for cid, s in seeds],
            "expanded_chunks": [{"chunk_id": cid, "text": chunk_text_map.get(cid, "")} for cid in final_chunks if cid in chunk_text_map],
            "community_summaries": comm_summaries,
            "debug": {
                "seed_nodes": sorted(list(seed_nodes))[:50],
                "visited_nodes_size": len(visited_nodes),
                "candidate_chunks_size": len(cand_chunks),
                "communities_hit": sorted(list(comm_ids))[:50],
            }
        }


# -----------------------------
# Integration snippet (how to plug into your app)
# -----------------------------

def integrate_with_existing_rag(
    faiss_hits: List[Tuple[str, float]],
    chunk_text_map: Dict[str, str],
    graphrag_artifact_dir: str,
) -> str:
    """
    Example helper: return a single context string for prompt assembly.
    """
    retriever = GraphRAGRetriever(graphrag_artifact_dir)
    pack = retriever.expand(faiss_hits, chunk_text_map)
    parts: List[str] = []

    # Community summaries first (global background)
    for s in pack["community_summaries"][:3]:
        parts.append(f"[CommunitySummary:{s['community_id']}]\n{s['summary']}")

    # Then chunks (seeds + expansions)
    for item in pack["expanded_chunks"]:
        cid = item["chunk_id"]
        txt = item["text"].strip()
        if not txt:
            continue
        parts.append(f"[Chunk:{cid}]\n{txt}")

    return "\n\n".join(parts)
