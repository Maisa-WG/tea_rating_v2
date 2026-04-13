"""
tab1_interactive.py
====================
交互评分 Tab - 升级版
"""

import streamlit as st

from config.constants import FACTORS
from config.settings import get_factor_color, get_score_color, FACTOR_COLORS
from utils.visualization import plot_flavor_shape


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
        st.info("💡 将参考知识库与判例库进行评分。确认结果可更新判例库。")

        # 参数设置区域
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 10, 1, 6, 1, 6, 1])
        rating_model = c2.selectbox(
            "选择评分模型",
            options=[
                "Qwen-8B （自部署小模型，可微调）",
                "DeepSeek-chat （DeepSeek最新通用模型）",
                "DeepSeek-reasoner （暂不可用）",
            ],
            index=0,
            key="rating_model"
        )
        model_map = {
            "Qwen-8B （自部署小模型，可微调）": "Qwen-8B",
            "DeepSeek-chat （DeepSeek最新通用模型）": "deepseek-chat",
            "DeepSeek-reasoner （DeepSeek推理模型）": "deepseek-reasoner",
        }
        selected_model_id = model_map[rating_model]
        
        r_num = c4.number_input("参考知识库条目数量", 1, 20, 3, key="r1")
        c_num = c6.number_input("参考进阶判例条目数量", 1, 20, 5, key="c1")

        # 用户输入区域
        if 'current_user_input' not in st.session_state:
            st.session_state.current_user_input = ""

        user_input = st.text_area(
            "请输入茶评描述",
            value=st.session_state.current_user_input,
            height=120,
            key="ui",
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
        if 'last_scored_input' not in st.session_state:
            st.session_state.last_scored_input = ""

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
        if st.button("开始评分", type="primary", width='stretch'):
            if not user_input:
                st.warning("⚠️ 请输入茶评描述")
            else:
                _handle_scoring(user_input, embedder, client, client_d, model_id, selected_model_id, r_num, c_num)

        # 查看提示词按钮 - 在评分按钮下方
        if st.session_state.get('last_llm_sys_prompt') or st.session_state.get('last_llm_user_prompt'):
            if st.button("🔍 查看本次提示词", key="view_prompt_tab1",
                        type="secondary", width='stretch'):
                _show_prompt_dialog()

        # 评分结果展示
        if st.session_state.last_scores:
            st.markdown("---")
            scored_input = st.session_state.get('last_scored_input', user_input)
            _render_scoring_results(scored_input, embedder)


def _handle_scoring(user_input, embedder, client, client_d, model_id, selected_model_id, r_num, c_num):
    """Handle scoring logic - Single line dynamic status display"""
    import time
    import threading

    # Import core scoring logic
    from core.ai_services import llm_normalize_user_input
    from core.scoring import run_scoring

    # 根据用户选择的评分模型，决定使用哪个 client 和 model_id
    if selected_model_id in ("deepseek-chat", "deepseek-reasoner"):
        scoring_client = client_d       # 使用 DeepSeek 客户端
        scoring_model_id = selected_model_id
    else:
        scoring_client = client          # 使用自部署 Qwen 客户端
        scoring_model_id = model_id      # 使用侧边栏配置的 model_id

    display_model_name = selected_model_id  # 用于状态栏展示

    # Create status placeholder (single line, no collapse)
    status_placeholder = st.empty()

    # Stage 1: Preprocessing
    status_placeholder.info(f"🍵 正在使用 {display_model_name} 品鉴... 📝 正在预处理茶评内容...")

    try:
        user_input_clean = llm_normalize_user_input(user_input, client_d)
    except Exception as e:
        status_placeholder.error(f"❌ 预处理失败: {str(e)}")
        return

    # Stage 2: Load knowledge base
    status_placeholder.info(f"🍵 正在使用 {display_model_name} 品鉴... 🔍 正在加载知识库与判例...")

    try:
        kb = st.session_state.kb
        basic_cases = st.session_state.basic_cases
        supp_cases = st.session_state.supp_cases
        prompt_config = st.session_state.prompt_config
    except Exception as e:
        status_placeholder.error(f"❌ 加载知识库失败: {str(e)}")
        return

    # Stage 3: AI thinking with timeout and progress tracking
    result = {'data': None, 'error': None, 'stage': 'thinking'}

    def run_scoring_thread():
        """Run scoring in background, pass progress via result dict"""
        try:
            scores, kb_h, case_h, sent_sys_p, sent_user_p = run_scoring(
                user_input=user_input_clean,
                kb=kb,
                basic_cases=basic_cases,
                supp_cases=supp_cases,
                prompt_config=prompt_config,
                embedder=embedder,
                client=scoring_client,
                model_id=scoring_model_id,
                r_num=r_num,
                c_num=c_num
            )
            result['data'] = (scores, kb_h, case_h, sent_sys_p, sent_user_p)
        except Exception as e:
            result['error'] = e

    # Start scoring thread
    thread = threading.Thread(target=run_scoring_thread, daemon=True)
    start_time = time.time()
    thread.start()

    # Track thinking stages (time-based, progressive, no loop)
    thinking_stages = [
        (5, "🎯 正在分析香气优雅性与辨识度..."),    
        (10, "👅 正在评估滋味协调性与饱和度..."),     
        (15, "⏳ 正在感受余韵持久性与苦涩度..."), 
        (20, "🤔 正在综合各项因子给出评分..."),      
        (25, "✨ 正在生成宗师总评..."),             
        (999, "🤖 AI 正在深度思考评分...")           
    ]

    current_stage_idx = 0
    timeout = 60  # 60s timeout

    # Dynamic status update (based on actual progress)
    while thread.is_alive():
        elapsed = time.time() - start_time

        # Check timeout
        if elapsed > timeout:
            status_placeholder.error(f"⏰ 评分超时（{timeout}秒）- 请稍后重试")
            st.error("""
            AI 思考时间过长，可能原因：
            1. 输入内容过长
            2. 服务器负载较高
            3. 网络连接不稳定

            💡 建议：稍后重试或简化输入内容
            """)
            return

        # Determine current stage based on elapsed time
        for i, (threshold, message) in enumerate(thinking_stages):
            if elapsed < threshold:
                current_stage_idx = i
                break

        # Display current stage (no loop, progressive)
        status_placeholder.info(
            f"🍵 正在使用 {display_model_name} 品鉴... {thinking_stages[current_stage_idx][1]}"
        )

        # Wait before checking again
        thread.join(timeout=0.8)

    # Scoring completed or error
    if result['error']:
        status_placeholder.error(f"❌ 评分失败: {str(result['error'])}")
        st.error(f"错误详情: {str(result['error'])}")
        return

    if result['data'] is None:
        status_placeholder.error("❌ 评分失败，请检查配置")
        return

    scores, kb_h, case_h, sent_sys_p, sent_user_p = result['data']

    # 关键检查：scores 为 None 说明 LLM/Embedding 调用失败
    if scores is None:
        status_placeholder.error("❌ 评分失败：模型未返回有效结果，请检查 Embedding 和 LLM 服务连接")
        st.session_state.last_llm_sys_prompt = sent_sys_p
        st.session_state.last_llm_user_prompt = sent_user_p
        return

    # Show completion status
    status_placeholder.success("🎉 评分完成！")
    time.sleep(0.5)
    status_placeholder.empty()

    # Save results
    st.session_state.last_scores = {
        "scores": scores,
        "kb_history": kb_h,
        "case_history": case_h,
        "sys_prompt": sent_sys_p,
        "user_prompt": sent_user_p
    }

    st.session_state.last_llm_sys_prompt = sent_sys_p
    st.session_state.last_llm_user_prompt = sent_user_p
    st.session_state.last_scored_input = user_input

    actual_scores = scores.get('scores', scores) if isinstance(scores, dict) else scores
    master_comment = scores.get("master_comment", "")
    st.session_state.last_master_comment = master_comment
    st.session_state.last_actual_scores = actual_scores

    # 关键修复：每次重新评分后都刷新校准区 widget key，避免校准区沿用旧值
    st.session_state.score_version += 1


def _render_scoring_results(user_input, embedder):
    """渲染评分结果 - 升级版"""

    # 空值保护：避免 scores 为 None 时崩溃
    raw_scores = st.session_state.last_scores
    if not raw_scores or not isinstance(raw_scores, dict):
        st.warning("⚠️ 暂无评分结果")
        return

    scores_data = raw_scores.get("scores")
    if scores_data is None:
        st.warning("⚠️ 评分结果为空，可能是 Embedding 或 LLM 服务不可用，请重试")
        return

    # 检查是否有嵌套的 'scores' 键（AI返回的数据结构）
    if isinstance(scores_data, dict) and 'scores' in scores_data:
        s = scores_data['scores']
    else:
        s = scores_data

    if s is None:
        st.warning("⚠️ 评分数据解析失败，请重试")
        return

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
        st.pyplot(plot_flavor_shape(s), width='stretch')

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
                                <span class="factor-name" style="display: flex; align-items: baseline;">
                                    <span style="color: {factor_color}; margin-left: 4px;">{f}</span>
                                    <span style="background-color: {factor_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-left: 6px;">
                                        {d['score']}/9
                                    </span>
                                </span>
                            </div>
                            <div class="factor-comment" style="margin-left: 8px;">{d['comment']}</div>
                            <div class="factor-suggestion" style="margin-left: 8px;">💡 {d.get('suggestion') or '暂无建议'}</div>
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

                # 添加评语输入框
                comment_value = st.text_area(
                    "评语",
                    s[f]['comment'],
                    key=f"c_{f}_{v}",
                    height=70,
                    placeholder="评语",
                    label_visibility="collapsed"
                )

                # 添加建议输入框
                suggestion_value = st.text_area(
                    "建议",
                    s[f].get('suggestion', ''),
                    key=f"sg_{f}_{v}",
                    height=60,
                    placeholder="建议",
                    label_visibility="collapsed"
                )

                cal_scores[f] = {
                    "score": new_score,
                    "comment": comment_value,
                    "suggestion": suggestion_value
                }

    # 保存按钮
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存校准评分", type="primary", width='stretch'):
            _save_calibrated_score(user_input, cal_scores, cal_master, embedder)

    with col2:
        if st.button("🔄 重置校准", width='stretch'):
            st.session_state.score_version += 1
            st.rerun()


def _save_calibrated_score(user_input, cal_scores, cal_master, embedder):
    """保存校准后的评分。

    新逻辑：
    1. 只对当前新增判例做 embedding。
    2. 其余进阶判例直接复用 supplementary_case.json 中缓存的 _embedding。
    3. 索引统一按缓存向量重建，不再因为删除/迁移而重新调用全部 embedding。
    """
    import time
    from core.resource_manager import ResourceManager

    nc = {
        "text": user_input,
        "scores": cal_scores,
        "master_comment": cal_master,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        ResourceManager.ensure_case_embedding(nc, embedder)

        _, supp_data = st.session_state.supp_cases
        supp_data.append(nc)
        new_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
        st.session_state.supp_cases = (new_idx, supp_data)

        st.success("✅ 校准已保存到进阶判例")
        st.session_state.score_version += 1
        time.sleep(0.3)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 保存校准评分失败: {e}")


@st.dialog("📝 本次发送给 LLM 的完整 Prompt", width="large")
def _show_prompt_dialog():
    """弹窗展示发送给 LLM 的系统提示词和用户提示词"""

    # System Prompt
    st.markdown("#### 🤖 System Prompt（系统提示词）")
    sys_prompt = st.session_state.get('last_llm_sys_prompt', '(暂无)')
    st.code(sys_prompt, language=None, height=300)

    st.markdown("---")

    # User Prompt
    st.markdown("#### 👤 User Prompt（用户提示词）")
    user_prompt = st.session_state.get('last_llm_user_prompt', '(暂无)')
    st.code(user_prompt, language=None, height=400)

