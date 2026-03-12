"""
tab2_batch.py
==============
批量评分 Tab - 简化版
"""

import streamlit as st


def render_tab2(embedder, client, client_d, model_id):
    """渲染批量评分 Tab - 简化版"""
    with st.container():
        # 参数设置区域
        c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
        r_n = c2.number_input("参考知识库条目数量", 1, 20, 3, key="rb")
        c_n = c4.number_input("参考进阶判例条目数量", 1, 20, 5, key="cb")

        # 文件上传区域
        f = st.file_uploader(
            "选择文件",
            type=['txt', 'docx'],
            help="上传包含茶评描述的文件",
            key="batch_uploader"
        )

        # 上传提示
        if f:
            st.success(f"✅ 已选择文件：{f.name}")

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

        # 批量评分按钮
        if st.button("批量评分", type="primary", width='stretch', disabled=not f):
            if f:
                st.info("⚠️ 批量评分功能待完整实现")
