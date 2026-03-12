"""
settings.py
===========
页面配置、CSS样式和路径定义
中国风茶学主题视觉系统
"""

import streamlit as st
from pathlib import Path


# ==========================================
# 页面配置
# ==========================================

def apply_page_config():
    """应用页面配置"""
    st.set_page_config(
        page_title="茶品六因子 AI 评分器",
        page_icon="🍵",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# ==========================================
# 中国传统色彩体系 (重建版)
# ==========================================

# 品牌主色系
COLORS = {
    # --- 核心品牌色 ---
    "primary": "#4A5D53",        # 黛绿 - 品牌主色
    "primary_light": "#6B7D70",  # 黛绿浅
    "primary_dark": "#3A4A40",   # 黛绿深

    # --- 辅助色 ---
    "secondary": "#8B5A2B",      # 茶色
    "accent": "#B8860B",          # 茶金 - 点缀

    # --- 中性色 ---
    "white": "#FFFFFF",
    "bg_page": "#FAFAF8",        # 暖白 - 纸张感
    "bg_card": "#FFFFFF",
    "bg_sidebar": "#F5F9F5",     # 月白 - 侧边栏
    "border": "#E8E8E8",
    "border_light": "#F0F0F0",

    # --- 文字色 ---
    "text_primary": "#2D2D2D",   # 墨色
    "text_secondary": "#5A5A5A",  # 深灰
    "text_muted": "#8A8A8A",     # 中灰
    "text_light": "#AAAAAA",     # 浅灰

    # --- 功能色 ---
    "success": "#6BAA4A",        # 柳绿 - 调整
    "warning": "#E6A23C",        # 琥珀 - 调整
    "danger": "#D9445C",         # 朱砂 - 调整
    "info": "#5B9BD5",           # 天青 - 调整
}

# 六因子专属色（产品系列色）
FACTOR_COLORS = {
    "优雅性": {"hex": "#C9B037", "name": "缃色", "series": "芽绿", "desc": "桂花明黄"},
    "辨识度": {"hex": "#6B8E5A", "name": "竹青", "series": "芽绿", "desc": "新竹绿"},
    "协调性": {"hex": "#A67B5B", "name": "檀色", "series": "姜黄", "desc": "紫檀木"},
    "饱和度": {"hex": "#8B6914", "name": "赭石", "series": "姜黄", "desc": "丹霞"},
    "持久性": {"hex": "#5B7C99", "name": "黛蓝", "series": "云蓝", "desc": "远山青"},
    "苦涩度": {"hex": "#6B4A3A", "name": "紫檀", "series": "胭脂", "desc": "古木深"},
}

# 评分区间色
SCORE_COLORS = {
    (9, 10): {"hex": "#C9B037", "name": "缃色", "bg": "#FDF6E3"},
    (7, 8): {"hex": "#6BAA4A", "name": "柳绿", "bg": "#EDF5EB"},
    (5, 6): {"hex": "#E6A23C", "name": "琥珀", "bg": "#FDF6ED"},
    (3, 4): {"hex": "#8B6914", "name": "赭石", "bg": "#F5F0E8"},
    (1, 2): {"hex": "#D9445C", "name": "朱砂", "bg": "#FDF0F2"},
}


def get_factor_color(factor: str) -> str:
    """获取因子颜色"""
    return FACTOR_COLORS.get(factor, {}).get("hex", COLORS["primary"])


def get_score_color(score: int) -> tuple:
    """获取评分颜色 (hex, bg)"""
    for (low, high), info in SCORE_COLORS.items():
        if low <= score <= high:
            return info["hex"], info["bg"]
    return COLORS["text_muted"], COLORS["bg_card"]


# ==========================================
# 全局 CSS 样式系统
# ==========================================

CSS_STYLES = f"""
<style>
/* ==================== 字体与基础 ==================== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;600&display=swap');

* {{
    box-sizing: border-box;
}}

:root {{
    /* 核心品牌色 */
    --color-primary: {COLORS['primary']};
    --color-primary-light: {COLORS['primary_light']};
    --color-primary-dark: {COLORS['primary_dark']};
    --color-secondary: {COLORS['secondary']};
    --color-accent: {COLORS['accent']};

    /* 中性色 */
    --color-white: {COLORS['white']};
    --color-bg-page: {COLORS['bg_page']};
    --color-bg-card: {COLORS['bg_card']};
    --color-bg-sidebar: {COLORS['bg_sidebar']};
    --color-border: {COLORS['border']};
    --color-border-light: {COLORS['border_light']};

    /* 文字色 */
    --color-text-primary: {COLORS['text_primary']};
    --color-text-secondary: {COLORS['text_secondary']};
    --color-text-muted: {COLORS['text_muted']};
    --color-text-light: {COLORS['text_light']};

    /* 功能色 */
    --color-success: {COLORS['success']};
    --color-warning: {COLORS['warning']};
    --color-danger: {COLORS['danger']};
    --color-info: {COLORS['info']};

    /* 六因子色 */
    --factor-grace: {FACTOR_COLORS['优雅性']['hex']};
    --factor-distinct: {FACTOR_COLORS['辨识度']['hex']};
    --factor-harmony: {FACTOR_COLORS['协调性']['hex']};
    --factor-saturation: {FACTOR_COLORS['饱和度']['hex']};
    --factor-endurance: {FACTOR_COLORS['持久性']['hex']};
    --factor-bitterness: {FACTOR_COLORS['苦涩度']['hex']};

    /* 字体 */
    --font-serif: "Noto Serif SC", "KaiTi", "楷体", serif;
    --font-sans: "Noto Sans SC", -apple-system, sans-serif;
}}

/* ==================== 页面基础 ==================== */
.stApp {{
    background-color: {COLORS['bg_page']};
    font-family: var(--font-sans);
    color: var(--color-text-primary);
}}

/* ==================== Hero 区域 ==================== */
.hero-section {{
    background: linear-gradient(180deg, {COLORS['bg_page']} 0%, {COLORS['bg_sidebar']} 100%);
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid {COLORS['border_light']};
    position: relative;
}}

.hero-section::after {{
    content: "";
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 1px;
    background: linear-gradient(90deg, transparent, {COLORS['border']}, transparent);
}}

.hero-title {{
    font-family: var(--font-serif);
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--color-primary);
    text-align: center;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
}}

.hero-subtitle {{
    font-family: var(--font-serif);
    font-size: 1rem;
    color: var(--color-text-secondary);
    text-align: center;
    font-style: italic;
    margin-bottom: 0.75rem;
}}

.slogan {{
    font-family: var(--font-serif);
    font-size: 1.1rem;
    color: var(--color-text-secondary);
    text-align: center !important;
    font-style: italic;
    margin-bottom: 0.75rem;
    padding: 0.5rem 1rem;
    display: block;
    margin-left: auto;
    margin-right: auto;
}}

.hero-meta {{
    text-align: center;
    font-size: 0.8rem;
    color: var(--color-text-muted);
}}

/* ==================== 侧边栏 ==================== */
[data-testid="stSidebar"] {{
    background-color: {COLORS['bg_sidebar']};
    border-right: 1px solid {COLORS['border_light']};
    font-size: 0.9rem;
}}

.sidebar-brand {{
    padding: 1rem;
    text-align: center;
    border-bottom: 1px solid {COLORS['border_light']};
    margin-bottom: 1rem;
}}

.sidebar-brand-icon {{
    font-size: 2rem;
    margin-bottom: 0.25rem;
}}

.sidebar-brand-title {{
    font-family: var(--font-serif);
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-primary);
    letter-spacing: 0.1em;
}}

.sidebar-section {{
    padding: 0.5rem 1rem 0.75rem 1rem;
    border-bottom: 1px solid {COLORS['border_light']};
}}

.sidebar-section-title {{
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}}

/* Metric 卡片 */
.metric-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 0.6rem;
    text-align: center;
    transition: all 0.2s ease;
}}

.metric-card:hover {{
    border-color: var(--color-primary);
    box-shadow: 0 2px 8px rgba(74, 93, 83, 0.1);
}}

.metric-value {{
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--color-primary);
}}

.metric-label {{
    font-size: 0.7rem;
    color: var(--color-text-muted);
    margin-top: 0.25rem;
}}

/* ==================== Tab 导航 ==================== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {COLORS['border_light']};
    padding-bottom: 0;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border: none;
    border-radius: 6px 6px 0 0;
    padding: 0.6rem 1.2rem;
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: {COLORS['bg_sidebar']};
    color: var(--color-primary);
}}

.stTabs [aria-selected="true"] {{
    background: {COLORS['primary']};
    color: {COLORS['white']};
    font-weight: 600;
}}

/* ==================== 按钮 ==================== */
.stButton > button {{
    font-family: var(--font-sans);
    font-weight: 500;
    border-radius: 6px;
    transition: all 0.2s ease;
}}

.stButton > button[kind="primary"] {{
    background: {COLORS['primary']};
    color: {COLORS['white']};
    border: none;
    padding: 0.5rem 1.25rem;
}}

.stButton > button[kind="primary"]:hover {{
    background: {COLORS['primary_dark']};
    box-shadow: 0 2px 8px rgba(74, 93, 83, 0.3);
}}

.stButton > button[kind="secondary"] {{
    background: {COLORS['white']};
    color: var(--color-primary);
    border: 1px solid {COLORS['primary']};
}}

.stButton > button[kind="secondary"]:hover {{
    background: var(--color-primary);
    color: {COLORS['white']};
}}

/* ==================== 输入框 ==================== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    font-family: var(--font-sans);
    transition: all 0.2s ease;
}}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: var(--color-primary);
    box-shadow: 0 0 0 2px rgba(74, 93, 83, 0.1);
}}

/* ==================== 卡片容器 ==================== */
.content-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}}

.content-card:hover {{
    border-color: {COLORS['border']};
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}}

.card-title {{
    font-family: var(--font-serif);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {COLORS['border_light']};
}}

/* ==================== 六因子卡片 ==================== */
.factor-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}}

.factor-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}}

.factor-card-grace {{
    border-left: 4px solid {FACTOR_COLORS['优雅性']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #FAF8F0 100%);
}}

.factor-card-distinct {{
    border-left: 4px solid {FACTOR_COLORS['辨识度']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #F5F8F3 100%);
}}

.factor-card-harmony {{
    border-left: 4px solid {FACTOR_COLORS['协调性']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #F8F5F2 100%);
}}

.factor-card-saturation {{
    border-left: 4px solid {FACTOR_COLORS['饱和度']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #F6F4EF 100%);
}}

.factor-card-endurance {{
    border-left: 4px solid {FACTOR_COLORS['持久性']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #F3F5F8 100%);
}}

.factor-card-bitterness {{
    border-left: 4px solid {FACTOR_COLORS['苦涩度']['hex']};
    background: linear-gradient(135deg, {COLORS['white']} 0%, #F4F2F0 100%);
}}

.factor-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}}

.factor-name {{
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 1rem;
}}

.factor-score {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 3rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}}

.factor-comment {{
    font-size: 0.85rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin-bottom: 0.5rem;
}}

.factor-suggestion {{
    font-size: 0.8rem;
    color: var(--color-text-muted);
    font-style: italic;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.02);
    border-radius: 4px;
    border-left: 2px solid {COLORS['border']};
}}

/* ==================== 宗师总评 ==================== */
.master-comment {{
    background: linear-gradient(135deg, #FDFCFB 0%, #F8F6F2 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    font-family: var(--font-serif);
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--color-text-primary);
    position: relative;
    padding-left: 3rem;
}}

.master-comment::before {{
    content: '"';
    position: absolute;
    left: 1rem;
    top: 0.5rem;
    font-size: 3rem;
    color: {FACTOR_COLORS['优雅性']['hex']};
    opacity: 0.4;
    font-family: serif;
    line-height: 1;
}}

.master-comment-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    font-family: var(--font-sans);
}}

/* ==================== 标签/徽章 ==================== */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}}

.badge-primary {{
    background: {COLORS['primary']};
    color: {COLORS['white']};
}}

.badge-outline {{
    background: transparent;
    border: 1px solid {COLORS['border']};
    color: var(--color-text-secondary);
}}

.badge-success {{
    background: #EDF5EB;
    color: {COLORS['success']};
}}

.badge-warning {{
    background: #FDF6ED;
    color: {COLORS['warning']};
}}

.badge-danger {{
    background: #FDF0F2;
    color: {COLORS['danger']};
}}

/* ==================== 提示框 ==================== */
.stSuccess {{
    background: #EDF5EB;
    border-left: 4px solid {COLORS['success']};
    border-radius: 6px;
}}

.stWarning {{
    background: #FDF6ED;
    border-left: 4px solid {COLORS['warning']};
    border-radius: 6px;
}}

.stError {{
    background: #FDF0F2;
    border-left: 4px solid {COLORS['danger']};
    border-radius: 6px;
}}

.stInfo {{
    background: #EDF2F7;
    border-left: 4px solid {COLORS['info']};
    border-radius: 6px;
}}

/* ==================== 展开面板 ==================== */
.streamlit-expanderHeader {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-weight: 500;
    color: var(--color-primary);
}}

.streamlit-expanderHeader:hover {{
    background: {COLORS['bg_sidebar']};
}}

/* ==================== 分隔线 ==================== */
hr {{
    border: none;
    border-top: 1px solid {COLORS['border_light']};
    margin: 1.5rem 0;
}}

.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {COLORS['border']}, transparent);
    margin: 1rem 0;
}}

/* ==================== 表格 ==================== */
.dataframe {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    overflow: hidden;
}}

/* ==================== 滚动条 ==================== */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS['bg_page']};
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS['border']};
    border-radius: 3px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--color-text-muted);
}}

/* ==================== 空状态 ==================== */
.empty-state {{
    text-align: center;
    padding: 3rem 1rem;
    color: var(--color-text-muted);
}}

.empty-state-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}}

.empty-state-text {{
    font-size: 0.95rem;
}}

/* ==================== 响应式 ==================== */
@media (max-width: 768px) {{
    .hero-title {{
        font-size: 1.6rem;
    }}

    .hero-subtitle {{
        font-size: 0.9rem;
    }}

    .content-card {{
        padding: 1rem;
    }}
}}
</style>
"""


def apply_css_styles():
    """应用全局CSS样式"""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


# ==========================================
# 知识库配置
# ==========================================

# 知识库模式：local（纯本地）、github（云端）、hybrid（混合）
KB_MODE = "local"

# 向量化批处理大小（优化性能）
KB_EMBEDDING_BATCH_SIZE = 50  # 从 10 增加到 50，减少 API 调用次数


# ==========================================
# 路径配置类
# ==========================================

class PathConfig:
    """路径管理类 —— 集中定义所有文件路径"""

    # 外部资源文件
    SRC_SYS_PROMPT = Path("sys_p.txt")

    # 运行时数据目录
    DATA_DIR = Path("./tea_data")
    RAG_DIR = Path("./tea_data/RAG")
    BACKUP_DIR = Path("./tea_backup")

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAG_DIR.mkdir(exist_ok=True)
        self.BACKUP_DIR.mkdir(exist_ok=True)
        self.GRAPHRAG_DIR = self.DATA_DIR / "graphrag_artifacts"
        self.GRAPHRAG_DIR.mkdir(exist_ok=True)

        # --- 知识库（RAG） ---
        self.kb_index = self.DATA_DIR / "kb.index"
        self.kb_chunks = self.DATA_DIR / "kb_chunks.pkl"
        self.kb_files = self.DATA_DIR / "kb_files.json"
        self.kb_metadata = self.DATA_DIR / "kb_metadata.json"  # 文件元数据缓存

        # --- 判例库（基础 + 进阶） ---
        self.basic_case_data = self.DATA_DIR / "basic_case.json"
        self.supp_case_index = self.DATA_DIR / "supp_cases.index"
        self.supp_case_data = self.DATA_DIR / "supplementary_case.json"

        # --- 微调与 Prompt ---
        self.training_file = self.DATA_DIR / "deepseek_finetune.jsonl"
        self.ft_status = self.DATA_DIR / "ft_status.json"
        self.prompt_config_file = self.DATA_DIR / "prompts.json"
        self.tea_examples_file = self.DATA_DIR / "tea_examples.json"

        # --- 模板与默认配置（位于 tea_backup） ---
        self.template_file = self.BACKUP_DIR / "template.xlsx"
        self.default_prompts = self.BACKUP_DIR / "default_prompts.json"


# 全局路径实例
PATHS = PathConfig()
