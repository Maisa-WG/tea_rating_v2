"""
resource_manager.py
====================
资源与数据管理模块
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple

import streamlit as st
import faiss

from config.settings import PATHS


class ResourceManager:
    """负责外部文件加载、数据持久化、微调数据管理"""

    # ---------- 通用文件读写 ----------
    @staticmethod
    def load_external_text(path: Path, fallback: str = "") -> str:
        """读取外部文本文件"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback

    @staticmethod
    def load_external_json(path: Path, fallback: Any = None) -> Any:
        """读取外部 JSON 文件"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback if fallback is not None else []

    @staticmethod
    def save_json(data: Any, path: Path):
        """保存数据为 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- FAISS 索引 + 数据 ----------
    @staticmethod
    def save(index: Any, data: Any, idx_path: Path, data_path: Path, is_json: bool = False):
        """保存 FAISS 索引和对应的数据文件"""
        if index:
            faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb", encoding="utf-8" if is_json else None) as f:
            if is_json:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                pickle.dump(data, f)

    @staticmethod
    def load(idx_path: Path, data_path: Path, is_json: bool = False) -> Tuple[Any, List]:
        """加载 FAISS 索引和对应的数据文件"""
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb", encoding="utf-8" if is_json else None) as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except:
                pass
        return faiss.IndexFlatL2(1024), []

    # ---------- 微调数据管理 ----------
    @staticmethod
    def _read_existing_finetune_texts() -> set:
        """读取已有微调数据中的判例文本集合，用于去重"""
        existing = set()
        if PATHS.training_file.exists():
            try:
                with open(PATHS.training_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            user_msg = entry.get("messages", [{}])[1].get("content", "")
                            # 提取【待评分产品】和【参考标准】之间的文本
                            start = user_msg.find("【待评分产品】")
                            end = user_msg.find("【参考标准")
                            if start >= 0 and end > start:
                                text = user_msg[start + len("【待评分产品】"):end].strip()
                                if text:
                                    existing.add(text)
                        except:
                            continue
            except:
                pass
        return existing

    @staticmethod
    def append_cases_to_finetune(cases: List[Dict], sys_prompt: str, user_tpl: str) -> Tuple[int, int]:
        """
        将判例追加到微调数据文件（自动去重）。

        Args:
            cases: 判例列表
            sys_prompt: 系统提示词
            user_tpl: 用户提示词模板

        Returns:
            Tuple[int, int]: (新增条数, 跳过的重复条数)
        """
        existing_texts = ResourceManager._read_existing_finetune_texts()
        added, skipped = 0, 0
        try:
            with open(PATHS.training_file, "a", encoding="utf-8") as f:
                for c in cases:
                    case_text = c.get("text", "").strip()
                    if case_text in existing_texts:
                        skipped += 1
                        continue
                    scores = c.get("scores", {})
                    master_comment = c.get("master_comment", "（人工校准）")
                    user_content = (user_tpl
                                    .replace("{product_desc}", case_text)
                                    .replace("{context_text}", "")
                                    .replace("{basic_case_text}", "")
                                    .replace("{case_text}", ""))
                    assistant_content = json.dumps(
                        {"master_comment": master_comment, "scores": scores},
                        ensure_ascii=False
                    )
                    entry = {
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    existing_texts.add(case_text)
                    added += 1
            return added, skipped
        except Exception as e:
            print(f"[ERROR] Finetune append failed: {e}")
            return 0, 0

    @staticmethod
    def save_ft_status(job_id, status, fine_tuned_model=None):
        """保存微调任务状态"""
        data = {"job_id": job_id, "status": status, "timestamp": time.time()}
        if fine_tuned_model:
            data["fine_tuned_model"] = fine_tuned_model
        with open(PATHS.ft_status, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_ft_status():
        """加载微调任务状态"""
        if PATHS.ft_status.exists():
            try:
                return json.load(open(PATHS.ft_status, 'r'))
            except:
                pass
        return None

    # ---------- RAG 文件列表管理 ----------
    @staticmethod
    def save_kb_files(file_list: List[str]):
        """保存知识库文件列表"""
        with open(PATHS.kb_files, "w", encoding="utf-8") as f:
            json.dump(file_list, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_kb_files() -> List[str]:
        """加载知识库文件列表"""
        if PATHS.kb_files.exists():
            try:
                with open(PATHS.kb_files, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return []

    # ---------- 茶评示例管理 ----------
    @staticmethod
    def load_tea_examples():
        """加载茶评示例列表，如果文件不存在或数据无效则返回 None（使用默认值）"""
        if PATHS.tea_examples_file.exists():
            try:
                with open(PATHS.tea_examples_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 验证数据格式：必须是列表，且每个元素都包含 title 和 text
                    if isinstance(data, list) and all(
                        isinstance(item, dict)
                        and 'title' in item
                        and 'text' in item
                        for item in data
                    ):
                        return data
                    else:
                        st.warning("⚠️ 茶评示例数据格式异常，将使用默认值")
                        return None
            except Exception as e:
                st.error(f"加载茶评示例失败: {e}")
        return None

    @staticmethod
    def save_tea_examples(examples: List[Dict]):
        """保存茶评示例列表到文件"""
        try:
            with open(PATHS.tea_examples_file, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"保存茶评示例失败: {e}")
            return False