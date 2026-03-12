"""
storage 模块
包含数据存储和向量索引功能
"""

from .database import (
    load_all_cases,
    insert_case,
    load_json_kb,
    save_json_kb,
    flatten_case_data,
    COLUMNS_SCHEMA,
    PATH_INITIAL,
    PATH_ADJUSTED,
    PATH_MANUAL
)
from .vector_store import (
    get_embedding,
    refresh_vector_index,
    load_vector_store
)

__all__ = [
    'load_all_cases',
    'insert_case',
    'load_json_kb',
    'save_json_kb',
    'flatten_case_data',
    'get_embedding',
    'refresh_vector_index',
    'load_vector_store',
    'COLUMNS_SCHEMA',
    'PATH_INITIAL',
    'PATH_ADJUSTED',
    'PATH_MANUAL'
]