"""
tab1_interactive.py
====================
交互评分 Tab - 升级版
"""

import streamlit as st

from config.constants import FACTORS
from config.settings import get_factor_color, get_score_color, FACTOR_COLORS
from utils.visualization import plot_flavor_shape
import re


# 智能截断辅助函数
def _truncate_chinese(text, max_length=100):
    """智能截断中文字符串，保留完整句子

    Args:
        text: 要截断的文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    # 找到最后一个句号、问号或感叹号
    for i in range(max_length, 0, -1):
        if text[i] in '。！？、，；：':
            return text[:i+1]

    # 如果没有标点，就在max_length处截断
    return text[:max_length] + "..."


# 因子对应的 CSS 类名映射
FACTOR_CARD_CLASS = {
    "优雅性": "factor-card-grace",
    "辨识度": "factor-card-distinct",
    "协调性": "factor-card-harmony",
    "饱和度": "factor-card-saturation",
    "持久性": "factor-card-endurance",
    "苦涩度": "factor-card-bitterness",
}


def render_tab1(embedder, client, client_d, model_id):
    """渲染交互评分 Tab"""
    with st.container():
        st.info("💡 将参考知识库与判例库进行评分。确认结果可一键更新判例库。")

        # 参数设置区域
        c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
        r_num = c2.number_input("参考知识库条目数量", 1, 20, 3, key="r1")
        c_num = c4.number_input("参考进阶判例条目数量", 1, 20, 5, key="c1")

        # 用户输入区域
        if 'current_user_input' not in st.session_state:
            st.session_state.current_user_input = ""

        user_input = st.text_area(
            "请输入茶评描述",
            value=st.session_state.current_user_input,
            height=120,
            key="ui",
            placeholder="例：这款桂花乌龙干茶清甜带花香，热闻像刚蒸好的桂花糕..."
        )
        st.session_state.current_user_input = user_input

        # Session state 初始化
        if 'last_scores' not in st.session_state:
            st.session_state.last_scores = None
            st.session_state.last_master_comment = ""
        if 'last_llm_sys_prompt' not in st.session_state:
            st.session_state.last_llm_sys_prompt = ""
        if 'last_llm_user_prompt' not in st.session_state:
            st.session_state.last_llm_user_prompt = ""
        if 'score_version' not in st.session_state:
            st.session_state.score_version = 0

        # 评分按钮 - 与输入框宽度一致
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

        # 评分按钮 - 与输入框宽度一致
        if st.button("开始评分", type="primary", use_container_width=True):
            if not user_input:
                st.warning("⚠️ 请输入茶评描述")
            else:
                _handle_scoring(user_input, embedder, client, client_d, model_id, r_num, c_num)

        # 评分结果展示
        if st.session_state.last_scores:
            st.markdown("---")
            _render_scoring_results(user_input, embedder)


def _handle_scoring(user_input, embedder, client, client_d, model_id, r_num, c_num):
    """处理评分逻辑"""
    with st.spinner(f"🍵 正在使用 {model_id} 品鉴..."):
        # 调用核心评分逻辑
        from core.ai_services import llm_normalize_user_input
        from core.scoring import run_scoring

        user_input_clean = llm_normalize_user_input(user_input, client_d)

        # 获取必要的数据
        kb = st.session_state.kb
        basic_cases = st.session_state.basic_cases
        supp_cases = st.session_state.supp_cases
        prompt_config = st.session_state.prompt_config

        # 调用 run_scoring 函数
        scores, kb_h, case_h, sent_sys_p, sent_user_p = run_scoring(
            user_input=user_input_clean,
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
            st.error("❌ 评分失败，请检查配置")
            return

        # 保存评分结果
        st.session_state.last_scores = {
            "scores": scores,
            "kb_history": kb_h,
            "case_history": case_h,
            "sys_prompt": sent_sys_p,
            "user_prompt": sent_user_p
        }

        # 同时保存到独立的 session_state 变量，供弹窗使用
        st.session_state.last_llm_sys_prompt = sent_sys_p
        st.session_state.last_llm_user_prompt = sent_user_p

        # 生成宗师总评
        master_comment = _generate_master_comment(scores, user_input_clean)
        st.session_state.last_master_comment = master_comment


def _render_scoring_results(user_input, embedder):
    """渲染评分结果 - 升级版"""
    s = st.session_state.last_scores["scores"]
    mc = st.session_state.last_master_comment

    # 宗师总评区域
    st.markdown('<div class="master-comment-label">宗师总评</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="master-comment">
        {mc}
    </div>
    ''', unsafe_allow_html=True)

    # 风味形态图 + 六因子卡片
    left_col, right_col = st.columns([30, 70])

    with left_col:
        st.markdown("##### 📊 风味形态")
        st.pyplot(plot_flavor_shape(st.session_state.last_scores), use_container_width=True)

    with right_col:
        st.markdown("##### 🏷️ 六因子评分")

        cols = st.columns(2)
        for i, f in enumerate(FACTORS):
            if f in s:
                d = s[f]
                factor_info = FACTOR_COLORS.get(f, {})
                factor_color = factor_info.get("hex", "#4A5D53")
                factor_name_cn = factor_info.get("name", f)
                score_hex, score_bg = get_score_color(d['score'])
                card_class = FACTOR_CARD_CLASS.get(f, "factor-card")

                with cols[i % 2]:
                    # 使用产品卡片风格
                    st.markdown(
                        f'''<div class="{card_class}">
                            <div class="factor-header">
                                <span class="factor-name">
                                    <span style="color: {factor_color};">{f}</span>
                                    <span style="font-size: 0.75rem; color: #999; margin-left: 4px;">{factor_name_cn}</span>
                                </span>
                                <span class="factor-score" style="background-color: {score_hex}; color: white;">
                                    {d['score']}/9
                                </span>
                            </div>
                            <div class="factor-comment">{d['comment']}</div>
                            <div class="factor-suggestion">💡 {d.get('suggestion', '')}</div>
                        </div>''',
                        unsafe_allow_html=True
                    )

    # 校准与修正区域
    st.markdown("---")
    _render_calibration_ui(user_input, embedder, s, mc)


def _render_calibration_ui(user_input, embedder, s, mc):
    """渲染校准与修正 UI - 升级版"""
    st.markdown("##### 🛠️ 评分校准与修正")

    v = st.session_state.score_version

    # 校准总评
    cal_master = st.text_area(
        "📝 校准总评",
        mc,
        key=f"cal_master_{v}",
        height=80,
        placeholder="请输入校准后的总评..."
    )

    cal_scores = {}

    # 分项调整区域
    st.markdown("##### 🍃 分项调整")

    active_factors = [f for f in FACTORS if f in s]
    grid_cols = st.columns(3)

    for i, f in enumerate(active_factors):
        factor_color = get_factor_color(f)
        with grid_cols[i % 3]:
            with st.container(border=True):
                t_col, s_col = st.columns([1, 1])

                with t_col:
                    st.markdown(
                        f"<div style='padding-top: 5px; color: {factor_color}; font-weight: 600;'>📌 {f}</div>",
                        unsafe_allow_html=True
                    )

                with s_col:
                    new_score = st.number_input(
                        "分数",
                        0, 9,
                        int(s[f]['score']),
                        1,
                        key=f"s_{f}_{v}",
                        label_visibility="collapsed"
                    )

                cal_scores[f] = {
                    "score": new_score,
                    "comment": st.text_area(
                        "评语",
                        s[f]['comment'],
                        key=f"c_{f}_{v}",
                        height=70,
                        placeholder="评语",
                        label_visibility="collapsed"
                    ),
                    "suggestion": st.text_area(
                        "建议",
                        s[f].get('suggestion', ''),
                        key=f"sg_{f}_{v}",
                        height=60,
                        placeholder="建议",
                        label_visibility="collapsed"
                    )
                }

    # 保存按钮
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("💾 保存校准评分", type="primary"):
            _save_calibrated_score(user_input, cal_scores, cal_master, embedder)

    with col2:
        if st.button("🔄 重置校准"):
            st.session_state.score_version += 1
            st.rerun()


def _save_calibrated_score(user_input, cal_scores, cal_master, embedder):
    """保存校准后的评分"""
    import time

    nc = {
        "text": user_input,
        "scores": cal_scores,
        "master_comment": cal_master,
        "created_at": time.strftime("%Y-%m-%d")
    }

    # 保存到进阶判例
    supp_idx, supp_data = st.session_state.supp_cases
    supp_data.append(nc)
    supp_idx.add(embedder.encode([user_input]))
    st.session_state.supp_cases = (supp_idx, supp_data)

    from config.settings import PATHS
    from core.resource_manager import ResourceManager

    ResourceManager.save(
        supp_idx,
        supp_data,
        PATHS.supp_case_index,
        PATHS.supp_case_data,
        is_json=True
    )

    st.success("✅ 校准已保存到进阶判例")
    st.session_state.score_version += 1
    time.sleep(0.5)


def _generate_master_comment(scores, user_input):
    """
    生成宗师总评

    Args:
        scores: 评分结果字典
        user_input: 用户输入的茶评

    Returns:
        str: 宗师总评文本
    """
    if not scores:
        return "暂无评分结果"

    # 提取分数信息
    factor_scores = []
    total_score = 0
    count = 0

    for factor, data in scores.items():
        if isinstance(data, dict) and 'score' in data:
            score = data['score']
            factor_scores.append(f"{factor.split(' ')[0]}{score}分")
            total_score += score
            count += 1

    avg_score = total_score / count if count > 0 else 0

    # 生成总评 - 使用智能截断
    comment = f"此茶{_truncate_chinese(user_input, 100)}。"

    if avg_score >= 8:
        comment += "品质卓越。"
    elif avg_score >= 6:
        comment += "品质优良。"
    elif avg_score >= 4:
        comment += "品质尚可。"
    else:
        comment += "品质有待提升。"

    # 添加特色描述
    high_scores = [k.split(' ')[0] for k, v in scores.items()
                   if isinstance(v, dict) and v.get('score', 0) >= 7]

    if high_scores:
        comment += f"在{'、'.join(high_scores[:2])}等方面表现突出。"

    return comment
