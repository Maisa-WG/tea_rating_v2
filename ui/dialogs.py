"""
dialogs.py
===========
各种弹窗 UI 组件 - 升级版
"""

import streamlit as st
from config.constants import TEA_EXAMPLES
from config.settings import PATHS
from core.resource_manager import ResourceManager


# ==========================================
# 提示词查看弹窗
# ==========================================

@st.dialog("📝 本次发送给 LLM 的完整 Prompt", width="large")
def show_prompt_dialog():
    """弹窗展示发送给 LLM 的系统提示词和用户提示词"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #F5F9F5 0%, #EDF5EB 100%); border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A5D53;">
        <span style="color: #4A5D53; font-size: 1.1em; font-weight: 600;">🔍 提示词详情</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔧 System Prompt（系统提示词）：**")
        system_prompt = st.session_state.get('last_llm_sys_prompt', '（暂无）')
        st.code(system_prompt, language=None, height=400)

    with col2:
        st.markdown("**💬 User Prompt（用户提示词）：**")
        user_prompt = st.session_state.get('last_llm_user_prompt', '（暂无）')
        st.code(user_prompt, language=None, height=400)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


# ==========================================
# 茶评示例弹窗
# ==========================================

@st.dialog("🍵 茶评示例", width="large")
def show_tea_examples_dialog():
    """展示预置茶评示例文本 - 参考 app.py 的简洁实现"""

    # 副标题区域
    st.info("📜 品鉴案例精选")

    # 提示信息（参考 app.py）
    st.caption("💡 以下是五组茶评示例，点击文本框即可选中复制，粘贴到「交互评分」中使用")

    # 从 session_state 加载示例
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES)

    # 显示示例数量
    st.markdown(f"**共 {len(examples)} 个示例**")

    st.divider()

    # 遍历展示每个示例（平铺展示，不用 expander）
    for i, ex in enumerate(examples):
        # 标题
        st.markdown(f"**{ex['title']}**")

        # 内容显示（使用 st.code()，方便选择复制）
        st.code(ex["text"], language=None)

        # 分隔线（最后一个示例后不加）
        if i < len(examples) - 1:
            st.markdown("")


# ==========================================
# 基础判例弹窗
# ==========================================

@st.dialog("📋 基础判例列表", width="large")
def show_basic_cases_dialog(embedder):
    """展示当前基础判例列表 - 升级版"""
    cases = st.session_state.basic_cases

    if not cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">暂无基础判例</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**📊 共 {len(cases)} 条基础判例**")
    st.markdown("---")

    for idx, case in enumerate(cases):
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"**判例 {idx + 1}**")
                st.caption(case.get('text', '')[:100] + "...")

            with col2:
                if st.button("✏️ 编辑", key=f"edit_basic_{idx}", width='stretch'):
                    st.session_state.editing_basic_idx = idx
                    st.rerun()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


@st.dialog("✏️ 编辑基础判例", width="large")
def edit_basic_case_dialog(idx: int):
    """编辑指定基础判例 - 升级版"""
    cases = st.session_state.basic_cases

    if idx >= len(cases):
        st.error("❌ 判例索引无效")
        return

    case = cases[idx]

    st.markdown(f"""
    <div style="padding: 12px; background: #F5F9F5; border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #4A5D53; font-weight: 600;">📝 编辑判例 #{idx + 1}</span>
    </div>
    """, unsafe_allow_html=True)

    st.text_area("判例文本", case.get('text', ''), height=100, key=f"edit_basic_text_{idx}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 保存修改", type="primary"):
            st.success("✅ 已保存（实际功能待实现）")

    with col3:
        if st.button("❌ 取消", type="secondary"):
            st.rerun()


# ==========================================
# 进阶判例弹窗
# ==========================================

@st.dialog("📋 进阶判例列表", width="large")
def show_supp_cases_dialog(embedder):
    """展示当前进阶判例列表 - 升级版"""
    _, cases = st.session_state.supp_cases

    if not cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">暂无进阶判例</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**📊 共 {len(cases)} 条进阶判例**")
    st.markdown("---")

    for idx, case in enumerate(cases):
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"**判例 {idx + 1}**")
                st.caption(case.get('text', '')[:100] + "...")

            with col2:
                if st.button("✏️ 编辑", key=f"edit_supp_{idx}", width='stretch'):
                    st.session_state.editing_supp_idx = idx
                    st.rerun()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


@st.dialog("✏️ 编辑进阶判例", width="large")
def edit_supp_case_dialog(idx: int, embedder):
    """编辑指定进阶判例 - 升级版"""
    _, cases = st.session_state.supp_cases

    if idx >= len(cases):
        st.error("❌ 判例索引无效")
        return

    case = cases[idx]

    st.markdown(f"""
    <div style="padding: 12px; background: #F5F9F5; border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #4A5D53; font-weight: 600;">📝 编辑判例 #{idx + 1}</span>
    </div>
    """, unsafe_allow_html=True)

    st.text_area("判例文本", case.get('text', ''), height=100, key=f"edit_supp_text_{idx}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 保存修改", type="primary"):
            st.success("✅ 已保存（实际功能待实现）")

    with col3:
        if st.button("❌ 取消", type="secondary"):
            st.rerun()


# ==========================================
# 茶评示例管理弹窗
# ==========================================

@st.dialog("⚙️ 茶评示例管理", width="large")
def manage_tea_examples_dialog():
    """管理茶评示例列表"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #F5F9F5 0%, #EDF5EB 100%); border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A5D53;">
        <span style="color: #4A5D53; font-size: 1.1em; font-weight: 600;">📚 茶评示例管理</span>
    </div>
    """, unsafe_allow_html=True)

    # 加载示例列表
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES[:])  # 复制一份

    st.markdown(f"**📊 共 {len(examples)} 个示例**")
    st.markdown("---")

    # 显示所有示例
    for idx, ex in enumerate(examples):
        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 1, 1])

            with col1:
                st.markdown(f"**{ex['title']}**")
                st.caption(ex.get('text', '')[:80] + "...")

            with col2:
                if st.button("✏️", key=f"edit_tea_{idx}", width='stretch', help="编辑"):
                    st.session_state.editing_tea_example_idx = idx
                    st.rerun()

            with col3:
                if st.button("🗑️", key=f"del_tea_{idx}", width='stretch', help="删除"):
                    examples.pop(idx)
                    ResourceManager.save_tea_examples(examples)
                    st.session_state.tea_examples = examples
                    st.success(f"✅ 已删除示例")
                    st.rerun()

    st.markdown("---")

    # 底部按钮
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ 新增示例", type="primary", width='stretch'):
            st.session_state.editing_tea_example_idx = -1  # -1 表示新增
            st.rerun()

    with col2:
        if st.button("🔄 恢复默认", width='stretch'):
            ResourceManager.save_tea_examples(TEA_EXAMPLES)
            st.session_state.tea_examples = TEA_EXAMPLES[:]
            st.success("✅ 已恢复为默认示例")
            st.rerun()

    with col4:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


@st.dialog("✏️ 编辑茶评示例", width="large")
def edit_tea_example_dialog(idx: int):
    """编辑指定茶评示例"""
    # 加载当前示例列表
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES[:])

    if idx == -1:
        # 新增模式
        st.markdown("""
        <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #4A5D53; font-weight: 600;">➕ 新增茶评示例</span>
        </div>
        """, unsafe_allow_html=True)
        current_title = ""
        current_text = ""
    else:
        # 编辑模式
        if idx >= len(examples):
            st.error("❌ 示例索引无效")
            return

        st.markdown(f"""
        <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #4A5D53; font-weight: 600;">✏️ 编辑茶评示例 #{idx + 1}</span>
        </div>
        """, unsafe_allow_html=True)
        current_title = examples[idx]['title']
        current_text = examples[idx]['text']

    # 编辑表单
    new_title = st.text_input(
        "标题",
        current_title,
        key=f"tea_title_{idx}",
        placeholder="例如：🌸 桂花乌龙",
        max_chars=50  # 限制标题长度
    )
    new_text = st.text_area(
        "内容",
        current_text,
        height=200,
        key=f"tea_text_{idx}",
        placeholder="请输入茶评描述...",
        max_chars=2000  # 限制内容长度
    )

    # 输入验证：去除首尾空格
    if new_title:
        new_title = new_title.strip()
    if new_text:
        new_text = new_text.strip()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 保存", type="primary", key=f"save_tea_{idx}"):
            if not new_title or not new_text:
                st.error("❌ 标题和内容不能为空")
            elif len(new_title) > 50:
                st.error("❌ 标题不能超过50个字符")
            elif len(new_text) > 2000:
                st.error("❌ 内容不能超过2000个字符")
            else:
                new_example = {"title": new_title, "text": new_text}

                if idx == -1:
                    # 新增
                    examples.append(new_example)
                    st.success("✅ 已添加新示例")
                else:
                    # 更新
                    examples[idx] = new_example
                    st.success("✅ 已保存修改")

                # 保存到文件和 session_state
                ResourceManager.save_tea_examples(examples)
                st.session_state.tea_examples = examples
                st.session_state.editing_tea_example_idx = None
                st.session_state.show_tea_examples = True  # 返回示例列表
                st.rerun()

    with col3:
        if st.button("❌ 取消", type="secondary", key=f"cancel_tea_{idx}"):
            st.session_state.editing_tea_example_idx = None
            st.session_state.show_tea_examples = True  # 返回主弹窗
            st.rerun()
