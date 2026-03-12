"""
配置模块
包含页面配置、CSS样式、路径配置和常量定义
"""

from .settings import apply_page_config, PATHS
from .constants import (
    TEA_EXAMPLES,
    FACTORS,
    FACTOR_ROWS,
    DEFAULT_USER_TEMPLATE,
    REFINE_SYSTEM_PROMPT
)

__all__ = [
    'apply_page_config',
    'PATHS',
    'TEA_EXAMPLES',
    'FACTORS',
    'FACTOR_ROWS',
    'DEFAULT_USER_TEMPLATE',
    'REFINE_SYSTEM_PROMPT'
]