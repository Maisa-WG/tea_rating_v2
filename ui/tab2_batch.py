"""
tab2_batch.py
==============
批量评分 Tab - 完整实现版
"""

import streamlit as st
import time
import logging
from typing import List, Dict, Tuple
from io import BytesIO

from config.constants import FACTORS
from config.settings import get_factor_color, FACTOR_COLORS
from utils.helpers import parse_batch_file, create_word_report

logger = logging.getLogger(__name__)


def render_tab2(embedder, client, client_d, model_id):
    """渲染批量评分 Tab - 完整版"""

    with st.container():
        st.info("💡 批量评分功能：上传包含多个茶评的文件，系统将自动分割并逐个评分，最后生成 Word 报告。")

        # 参数设置区域
        c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
        r_num = c2.number_input("参考知识库条目数量", 2, 5, 3, key="rb")
        c_num = c4.number_input("参考进阶判例条目数量", 2, 10, 5, key="cb")

        # 文件上传区域
        f = st.file_uploader(
            "选择文件",
            type=['txt', 'docx', 'pdf'],
            help="支持格式：.txt, .docx, .pdf。文件中可包含多个茶评描述。",
            key="batch_uploader"
        )

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
        if st.button("🚀 批量评分", type="primary", width='stretch', disabled=not f):
            if f:
                _handle_batch_scoring(f, embedder, client, client_d, model_id, r_num, c_num)

        # 显示历史结果
        if 'batch_results' in st.session_state and st.session_state.batch_results:
            _display_batch_results()


def _handle_batch_scoring(uploaded_file, embedder, client, client_d, model_id, r_num, c_num):
    """处理批量评分逻辑"""

    # 1. 解析文件并分割茶评
    with st.spinner("📄 正在解析文件..."):
        reviews = parse_batch_file(uploaded_file)

        if not reviews:
            st.error("❌ 无法从文件中提取茶评内容，请检查文件格式。")
            return

        st.success(f"✅ 成功提取 {len(reviews)} 条茶评")

    # 显示提取的茶评预览
    with st.expander("📋 查看提取的茶评条目", expanded=True):
        for i, review in enumerate(reviews, 1):
            st.text(f"{i}. {review[:100]}{'...' if len(review) > 100 else ''}")

    # 2. 自动开始批量评分
    _batch_score_reviews(reviews, embedder, client, client_d, model_id, r_num, c_num)


def _batch_score_reviews(reviews: List[str], embedder, client, client_d, model_id, r_num, c_num):
    """执行批量评分"""

    # 获取必要的数据
    kb = st.session_state.kb
    basic_cases = st.session_state.basic_cases
    supp_cases = st.session_state.supp_cases
    prompt_config = st.session_state.prompt_config

    logger.info(f"开始批量评分，共 {len(reviews)} 条茶评")

    # 检查必要参数
    if not embedder or not client:
        st.error("❌ 系统未正确初始化，请检查侧边栏配置")
        return

    if not kb or not kb[1]:
        st.warning("⚠️ 知识库为空，评分效果可能受影响")

    if not supp_cases or not supp_cases[1]:
        st.warning("⚠️ 进阶判例库为空，评分效果可能受影响")

    # 初始化结果列表
    results = []
    failed_items = []

    # 创建进度条
    progress_bar = st.progress(0, text="准备开始评分...")
    status_text = st.empty()

    # 导入评分函数
    from core.ai_services import llm_normalize_user_input
    from core.scoring import run_scoring

    total_reviews = len(reviews)

    for i, review in enumerate(reviews):
        # 更新进度
        progress = (i) / total_reviews
        progress_bar.progress(progress, text=f"正在评分第 {i+1}/{total_reviews} 条...")

        try:
            # 文本预处理（可选）
            with st.spinner(f"🍵 正在处理第 {i+1}/{total_reviews} 条..."):
                review_clean = review  # 暂时不使用 LLM 预处理，直接使用原文
                logger.info(f"第 {i+1} 条准备评分，原文长度: {len(review)}")

                # 调用评分逻辑
                scores, kb_h, case_h, sent_sys_p, sent_user_p = run_scoring(
                    user_input=review_clean,
                    kb=kb,
                    basic_cases=basic_cases,
                    supp_cases=supp_cases,
                    prompt_config=prompt_config,
                    embedder=embedder,
                    client=client,
                    model_id=model_id,
                    r_num=r_num,
                    c_num=c_num
                )

                if scores is None:
                    failed_items.append({
                        'id': i + 1,
                        'text': review,
                        'error': '评分失败'
                    })
                    status_text.warning(f"⚠️ 第 {i+1} 条评分失败")
                else:
                    # 生成宗师总评 - 处理嵌套的 scores 结构
                    actual_scores = scores
                    if isinstance(scores, dict) and 'scores' in scores:
                        actual_scores = scores['scores']
                    elif isinstance(scores, dict) and 'master_comment' in scores:
                        actual_scores = scores

                    try:
                        master_comment = scores.get("master_comment", "")
                    except Exception as e:
                        logger.error(f"第 {i+1} 条生成宗师总评失败: {e}")
                        master_comment = "暂无总评"

                    results.append({
                        'id': i + 1,
                        'text': review,
                        'text_clean': review_clean,
                        'scores': scores,
                        'master_comment': master_comment,
                        'kb_history': kb_h,
                        'case_history': case_h
                    })

                    status_text.success(f"✅ 第 {i+1} 条评分完成")

            # 性能优化：缩短等待时间
            time.sleep(0.1)

        except Exception as e:
            failed_items.append({
                'id': i + 1,
                'text': review,
                'error': str(e)
            })
            status_text.error(f"❌ 第 {i+1} 条评分出错: {e}")

    # 完成进度
    progress_bar.progress(1.0, text="评分完成！")

    # 保存结果到 session_state
    st.session_state.batch_results = results
    st.session_state.batch_failed = failed_items

    # 显示总结
    st.markdown("---")
    st.markdown("### 📊 批量评分完成")

    col1, col2, col3 = st.columns(3)
    col1.metric("总数", str(total_reviews))
    col2.metric("成功", str(len(results)), delta_color="normal")
    col3.metric("失败", str(len(failed_items)), delta_color="inverse")

    if failed_items:
        with st.expander("⚠️ 查看失败的条目"):
            for item in failed_items:
                st.text(f"条目 {item['id']}: {item['error']}")
                st.text(f"内容: {item['text'][:100]}...")
                st.markdown("---")


def _display_batch_results():
    """显示批量评分结果"""

    results = st.session_state.batch_results
    failed_items = st.session_state.get('batch_failed', [])

    st.markdown("---")
    st.markdown("### 📋 评分结果详情")

    # 标签页：成功结果 / 失败结果
    tab1, tab2, tab3 = st.tabs(["✅ 成功结果", "❌ 失败结果", "📥 导出报告"])

    with tab1:
        if not results:
            st.info("没有成功的评分结果")
        else:
            for i, item in enumerate(results):
                with st.expander(f"条目 {item['id']}", expanded=i == 0):
                    # 原文
                    st.text(f"**原文**: {item['text']}")

                    # 宗师总评
                    st.info(f"📝 **宗师总评**: {item['master_comment']}")

                    # 六因子评分
                    scores_data = item['scores']

                    # 检查嵌套结构
                    if 'scores' in scores_data:
                        s = scores_data['scores']
                    else:
                        s = scores_data

                    # 显示六因子
                    cols = st.columns(3)
                    for j, factor in enumerate(FACTORS):
                        if factor in s:
                            with cols[j % 3]:
                                factor_color = get_factor_color(factor)
                                score = s[factor]['score']
                                comment = s[factor]['comment']
                                suggestion = s[factor].get('suggestion', '')

                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin: 5px 0;">
                                        <div style="color: {factor_color}; font-weight: 600;">📌 {factor}</div>
                                        <div style="display: flex; align-items: baseline; margin: 5px 0;">
                                            <span style="background-color: {factor_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9rem;">
                                                {score}/9
                                            </span>
                                        </div>
                                        <div style="font-size: 0.9rem; margin: 5px 0;">{comment}</div>
                                        <div style="font-size: 0.85rem; color: #666;">💡 {suggestion}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # 检索历史
                    st.caption(f"🔍 {item['kb_history']} | {item['case_history']}")

    with tab2:
        if not failed_items:
            st.info("没有失败的条目")
        else:
            for item in failed_items:
                st.error(f"**条目 {item['id']}**: {item['error']}")
                st.text(f"内容: {item['text']}")
                st.markdown("---")

    with tab3:
        if results:
            st.markdown("### 📥 导出 Word 报告")

            st.info("💡 点击下方按钮生成并下载完整的 Word 格式评分报告")

            if st.button("📄 生成 Word 报告", type="primary"):
                with st.spinner("正在生成报告..."):
                    try:
                        # 准备报告数据
                        report_data = []
                        for item in results:
                            report_data.append({
                                'id': item['id'],
                                'text': item['text'],
                                'scores': item['scores'],
                                'master_comment': item['master_comment']
                            })

                        # 生成 Word 文档
                        doc_buffer = create_word_report(report_data)

                        # 提供下载
                        st.download_button(
                            label="📥 下载评分报告",
                            data=doc_buffer,
                            file_name=f"茶评批量评分报告_{time.strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                        st.success("✅ 报告生成成功！")

                    except Exception as e:
                        st.error(f"❌ 报告生成失败: {e}")
        else:
            st.warning("⚠️ 没有可导出的结果")

