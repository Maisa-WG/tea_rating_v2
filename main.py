"""
main.py
========
茶饮六因子AI评分器 Pro - 主入口

模块化重构后的主程序
"""

import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import time
import pickle

# ==========================================
# 导入模块
# ==========================================

# 配置模块
from config.settings import apply_page_config, apply_css_styles, PATHS
from config.constants import DEFAULT_USER_TEMPLATE

# 核心模块
from core.resource_manager import ResourceManager
from core import bootstrap_cases

# UI 模块
from ui.sidebar import render_sidebar
from ui.dialogs import (
    show_prompt_dialog,
    show_tea_examples_dialog,
    manage_tea_examples_dialog,
    edit_tea_example_dialog
)
from ui.tab1_interactive import render_tab1
from ui.tab2_batch import render_tab2
from ui.tab3_knowledge import render_tab3
from ui.tab4_cases import render_tab4
from ui.tab5_finetune import render_tab5
from ui.tab6_prompts import render_tab6


# ==========================================
# 页面配置
# ==========================================

apply_page_config()
apply_css_styles()


# ==========================================
# Session 初始化
# ==========================================

if 'loaded' not in st.session_state:
    print("\n" + "=" * 70)
    print("[INFO] ========== 茶饮六因子AI评分器 - 系统初始化 ==========")
    print("=" * 70)

    # 1. 加载知识库缓存
    print("[INFO] 步骤 1/4: 加载知识库缓存...")
    kb_idx, kb_data = ResourceManager.load(PATHS.kb_index, PATHS.kb_chunks)
    st.session_state.kb = (kb_idx, kb_data)
    st.session_state.kb_files = ResourceManager.load_kb_files()
    print(f"[INFO]   → 知识库: {len(kb_data)} 个片段")

    # 2. 加载判例库（基础 + 进阶）
    print("[INFO] 步骤 2/4: 加载判例库...")
    st.session_state.basic_cases = ResourceManager.load_external_json(PATHS.basic_case_data, fallback=[])
    supp_idx, supp_data = ResourceManager.load(PATHS.supp_case_index, PATHS.supp_case_data, is_json=True)
    st.session_state.supp_cases = (supp_idx, supp_data)
    print(f"[INFO]   → 基础判例: {len(st.session_state.basic_cases)} 条")
    print(f"[INFO]   → 进阶判例: {len(supp_data)} 条")

    # 3. RAG 延迟加载标记
    print("[INFO] 步骤 3/4: 检查 RAG 状态...")
    if len(kb_data) == 0:
        st.session_state.rag_loading_needed = True
        st.session_state.rag_loading_status = "pending"
        print("[INFO]   ⚠️ 本地知识库为空，将从 GitHub 加载")
    else:
        st.session_state.rag_loading_needed = False
        st.session_state.rag_loading_status = "complete"
        print(f"[INFO]   ✅ 已加载 {len(kb_data)} 个知识片段")

    # 4. 加载 Prompt 配置
    print("[INFO] 步骤 4/4: 加载 Prompt 配置...")
    if PATHS.prompt_config_file.exists():
        try:
            with open(PATHS.prompt_config_file, 'r', encoding='utf-8') as f:
                st.session_state.prompt_config = json.load(f)
        except:
            st.session_state.prompt_config = {
                "system_template": ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, ""),
                "user_template": DEFAULT_USER_TEMPLATE
            }
    else:
        st.session_state.prompt_config = {
            "system_template": ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, ""),
            "user_template": DEFAULT_USER_TEMPLATE
        }
    print("[INFO]   ✅ Prompt 配置加载完成")

    # 5. 加载茶评示例
    print("[INFO] 步骤 5/5: 加载茶评示例...")
    tea_examples = ResourceManager.load_tea_examples()
    if tea_examples is None or len(tea_examples) == 0:
        # 文件不存在或内容为空，使用默认值
        st.session_state.tea_examples = TEA_EXAMPLES[:]
        print("[INFO]   ✅ 使用默认茶评示例")
    else:
        st.session_state.tea_examples = tea_examples
        print(f"[INFO]   ✅ 已加载 {len(tea_examples)} 个茶评示例")

    st.session_state.loaded = True
    print("=" * 70)
    print("[INFO] ========== 系统初始化完成 ==========")
    print("=" * 70 + "\n")


# ==========================================
# 渲染侧边栏
# ==========================================

embedder, client, client_d, model_id = render_sidebar()

# 初始化判例
bootstrap_cases(embedder)


# ==========================================
# Hero 区域
# ==========================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🍵 茶品六因子 AI 评分器 Pro</div>
    <div class="slogan">"一片叶子落入水中，改变了水的味道..."</div>
    <div class="hero-meta">推理服务开放时间：9:00~20:00</div>
</div>
""", unsafe_allow_html=True)

# Tab 定义 (更简洁的标签)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "交互评分", "批量评分", "知识库",
    "判例库", "模型微调", "Prompt配置"
])


# ==========================================
# 渲染各 Tab
# ==========================================

with tab1:
    render_tab1(embedder, client, client_d, model_id)

with tab2:
    render_tab2(embedder, client, client_d, model_id)

with tab3:
    import os
    aliyun_key = os.getenv("ALIYUN_API_KEY") or st.secrets.get("ALIYUN_API_KEY", "")
    render_tab3(aliyun_key)

with tab4:
    render_tab4(embedder)

with tab5:
    render_tab5()

with tab6:
    render_tab6()


# ==========================================
# 弹窗处理
# ==========================================
# 使用 if-elif 链确保同一脚本运行中最多打开一个弹窗

if st.session_state.get('show_prompt_dialog'):
    show_prompt_dialog()
    st.session_state.show_prompt_dialog = False

elif st.session_state.get('show_tea_examples'):
    show_tea_examples_dialog()
    # 不立即清除状态，保持弹窗可重入（编辑后可返回）
    if st.session_state.get('editing_tea_example_idx') is None:
        st.session_state.show_tea_examples = False

elif st.session_state.get('manage_tea_examples'):
    manage_tea_examples_dialog()
    st.session_state.manage_tea_examples = False

elif st.session_state.get('editing_tea_example_idx') is not None:
    idx = st.session_state.editing_tea_example_idx
    edit_tea_example_dialog(idx)
    st.session_state.editing_tea_example_idx = None

elif st.session_state.get('show_basic_cases'):
    from ui.dialogs import show_basic_cases_dialog
    show_basic_cases_dialog(embedder)
    st.session_state.show_basic_cases = False

elif st.session_state.get('show_supp_cases'):
    from ui.dialogs import show_supp_cases_dialog
    show_supp_cases_dialog(embedder)
    st.session_state.show_supp_cases = False

elif st.session_state.get('editing_basic_idx') is not None:
    from ui.dialogs import edit_basic_case_dialog
    edit_basic_case_dialog(st.session_state.editing_basic_idx)
    # 关闭时清除状态

elif st.session_state.get('editing_supp_idx') is not None:
    from ui.dialogs import edit_supp_case_dialog
    edit_supp_case_dialog(st.session_state.editing_supp_idx, embedder)
    # 关闭时清除状态