"""
tab5_finetune.py
=================
模型微调 Tab - 升级版
"""

import streamlit as st
import requests


def render_tab5():
    """渲染模型微调 Tab - 升级版"""
    with st.container():
        MANAGER_URL = "http://117.50.138.123:8001"

        # 标题区域
        st.markdown("""
        <div style="padding: 15px; background: linear-gradient(135deg, #FDF6ED 0%, #FAF0D8 100%);
                    border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #B8860B;">
            <div style="color: #8B5A2B; font-size: 1.2em; font-weight: 600; margin-bottom: 5px;">🚀 模型微调 (LoRA)</div>
            <div style="color: #666; font-size: 0.9em;">使用判例库数据训练专用模型，提升评分准确性</div>
        </div>
        """, unsafe_allow_html=True)

        # 服务器状态
        server_status = _check_server_status(MANAGER_URL)
        _display_server_status(server_status)

        st.markdown("---")

        # 当前微调数据统计
        _display_data_statistics()

        st.markdown("---")

        # 三个功能区域
        ft_c1, ft_c2, ft_c3 = st.columns(3)

        with ft_c1:
            _render_manual_data_section()

        with ft_c2:
            _render_auto_fill_section()

        with ft_c3:
            _render_training_section(MANAGER_URL, server_status)


def _check_server_status(manager_url: str) -> str:
    """检查服务器状态"""
    try:
        resp = requests.get(f"{manager_url}/status", timeout=2)
        if resp.status_code == 200:
            status_data = resp.json()
            return "idle" if status_data.get("vllm_status") == "running" else "training"
        else:
            return "error"
    except:
        return "offline"


def _display_server_status(status: str):
    """显示服务器状态 - 升级版"""
    st.markdown("##### 🔮 服务器状态")

    if status == "idle":
        st.markdown("""
        <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; border-left: 4px solid #6BAA4A;">
            <span style="color: #2D4A1C; font-weight: 600;">🟢 服务器就绪</span>
            <span style="color: #666; font-size: 0.9em;">（正在进行推理服务）</span>
        </div>
        """, unsafe_allow_html=True)

    elif status == "training":
        st.markdown("""
        <div style="padding: 12px; background: #FDF6ED; border-radius: 8px; border-left: 4px solid #E6A23C;">
            <span style="color: #8B5A2B; font-weight: 600;">🟠 正在微调训练中...</span>
            <span style="color: #666; font-size: 0.9em;">（推理服务暂停）</span>
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️ **注意：** 此时无法进行评分交互，请耐心等待训练完成。")

    elif status == "offline":
        st.markdown("""
        <div style="padding: 12px; background: #FDF0F2; border-radius: 8px; border-left: 4px solid #D9445C;">
            <span style="color: #D9445C; font-weight: 600;">🔴 无法连接到 GPU 服务器</span>
            <span style="color: #666; font-size: 0.9em;">（请联系管理员）</span>
        </div>
        """, unsafe_allow_html=True)


def _display_data_statistics():
    """显示数据统计 - 升级版"""
    from config.settings import PATHS

    if PATHS.training_file.exists():
        with open(PATHS.training_file, "r", encoding="utf-8") as f:
            data_count = len(f.readlines())
    else:
        data_count = 0

    basic_count = len(st.session_state.basic_cases)
    supp_count = len(st.session_state.supp_cases[1])

    st.markdown("##### 📊 微调数据统计")

    st.markdown(f"""
    <div style="display: flex; gap: 15px; margin: 10px 0;">
        <div style="flex: 1; padding: 12px; background: white; border: 1px solid #E8E8E8; border-radius: 8px; text-align: center;">
            <div style="color: #999; font-size: 0.8rem; margin-bottom: 4px;">微调数据</div>
            <div style="color: #B8860B; font-size: 1.5rem; font-weight: 600;">{data_count}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: white; border: 1px solid #E8E8E8; border-radius: 8px; text-align: center;">
            <div style="color: #999; font-size: 0.8rem; margin-bottom: 4px;">基础判例</div>
            <div style="color: #6BAA4A; font-size: 1.5rem; font-weight: 600;">{basic_count}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: white; border: 1px solid #E8E8E8; border-radius: 8px; text-align: center;">
            <div style="color: #999; font-size: 0.8rem; margin-bottom: 4px;">进阶判例</div>
            <div style="color: #5B9BD5; font-size: 1.5rem; font-weight: 600;">{supp_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_manual_data_section():
    """渲染手动准备数据区域 - 升级版"""
    st.markdown("""
    <div style="padding: 12px; background: #F5F9F5; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4A5D53;">
        <div style="color: #4A5D53; font-weight: 600; margin-bottom: 4px;">📄 手动准备数据</div>
        <div style="color: #666; font-size: 0.85rem;">请按照模板格式填写微调训练数据</div>
    </div>
    """, unsafe_allow_html=True)

    ft_file = st.file_uploader(
        "上传已填写的微调数据文件",
        type=['xlsx', 'xls'],
        key="ft_data_upload",
        help="支持 XLSX、XLS 格式，单文件最大 200MB",
        label_visibility="visible"
    )

    # 下载模板按钮
    template_path = r"D:\jinghan_research\tea_rating_v2-main\tea_backup\template.xlsx"
    import os
    if os.path.exists(template_path):
        with open(template_path, 'rb') as f:
            template_data = f.read()

        st.download_button(
            "⬇️ 下载微调数据模板",
            template_data,
            "微调数据模板.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_ft_template",
            use_container_width=True
        )

    if ft_file:
        st.success(f"✅ 已选择：{ft_file.name}")

        if st.button("📤 导入微调数据", type="primary", key="ft_import", use_container_width=True):
            st.info("⚠️ 导入功能待完整实现")


def _render_auto_fill_section():
    """渲染自动填充数据区域 - 升级版"""
    st.markdown("""
    <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #6BAA4A;">
        <div style="color: #2D4A1C; font-weight: 600; margin-bottom: 4px;">🤖 自动填充数据</div>
        <div style="color: #666; font-size: 0.85rem;">从判例库自动提取数据并追加到微调文件</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("💡 自动跳过重复数据")

    # 单列布局，两个按钮垂直排列
    if st.button("➕ 基础判例 → 微调数据", use_container_width=True, key="ft_add_basic"):
        st.info("⚠️ 功能待完整实现")

    if st.button("➕ 进阶判例 → 微调数据", use_container_width=True, key="ft_add_supp"):
        st.info("⚠️ 功能待完整实现")


def _render_training_section(manager_url: str, server_status: str):
    """渲染启动训练区域 - 升级版"""
    from config.settings import PATHS

    st.markdown("""
    <div style="padding: 12px; background: #FDF6ED; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #B8860B;">
        <div style="color: #8B5A2B; font-weight: 600; margin-bottom: 4px;">🔥 启动训练</div>
        <div style="color: #666; font-size: 0.85rem;">将数据上传至 GPU 服务器并开始训练</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⏱️ 训练期间推理服务将中断约 2-5 分钟")

    data_count = 0
    if PATHS.training_file.exists():
        with open(PATHS.training_file, "r", encoding="utf-8") as f:
            data_count = len(f.readlines())

    # 数据状态
    if data_count > 0:
        st.markdown(f"""
        <div style="padding: 8px 12px; background: #EDF5EB; border-radius: 6px; margin: 10px 0;">
            <span style="color: #4A5D53; font-weight: 600;">✅ 准备就绪：</span>
            <span style="color: #666;">{data_count} 条训练数据</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 8px 12px; background: #FDF0F2; border-radius: 6px; margin: 10px 0;">
            <span style="color: #D9445C; font-weight: 600;">⚠️ 暂无训练数据</span>
        </div>
        """, unsafe_allow_html=True)

    btn_disabled = (server_status != "idle") or (data_count == 0)

    # 自定义按钮颜色
    st.markdown("""
    <style>
    button[kind="primary"] {
        background-color: #4A5D53 !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #4A5D53DD !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🔥 开始微调", type="primary", disabled=btn_disabled, use_container_width=True, key="start_ft"):
        st.info("⚠️ 训练功能待完整实现")
