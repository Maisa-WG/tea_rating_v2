"""
tab6_prompts.py
================
提示词配置 Tab - 升级版
"""

import streamlit as st
import json


def render_tab6():
    """渲染提示词配置 Tab - 升级版"""
    with st.container():
        pc = st.session_state.prompt_config

        # 标题区域
        st.markdown("""
        <div style="padding: 15px; background: linear-gradient(135deg, #F5F9F5 0%, #EDF5EB 100%);
                    border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A5D53;">
            <div style="color: #4A5D53; font-size: 1.2em; font-weight: 600; margin-bottom: 5px;">📝 Prompt 配置</div>
            <div style="color: #666; font-size: 0.9em;">自定义系统提示词以优化模型输出</div>
        </div>
        """, unsafe_allow_html=True)

        # 说明区域
        st.markdown("""
        <div style="padding: 12px; background: #FDF6ED; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #C9B037;">
            <div style="color: #8B5A2B; font-weight: 600; margin-bottom: 6px;">💡 使用说明</div>
            <div style="color: #666; font-size: 0.9em;">
                • <strong>系统提示词</strong>：可以修改。完整全面的提示词会让大语言模型返回更准确的结果。<br>
                • <strong>用户提示词</strong>：不可修改。保证了发送内容与回答内容的基本结构，便于解析。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 系统提示词编辑区
        st.markdown("##### 🔧 系统提示词（可修改）")

        sys_t = st.text_area(
            "系统提示词内容",
            pc.get('system_template', ''),
            height=350,
            key="sys_prompt_edit",
            help="自定义系统提示词以优化模型行为"
        )

        # 操作按钮
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

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 保存修改", type="primary", width='stretch', key="save_prompt"):
                _save_prompt_config(pc, sys_t)

        with col2:
            if st.button("🔄 恢复默认", width='stretch', key="restore_prompt"):
                _restore_default_prompt()

        with col3:
            if st.button("📄 查看对比", type="primary", width='stretch'):
                _show_prompt_comparison(pc.get('system_template', ''), sys_t)

        st.markdown("---")

        # 用户提示词查看区
        st.markdown("##### 📋 用户提示词（不可修改）")

        st.caption("🔒 此提示词保证输出格式的一致性，不允许修改")

        st.text_area(
            "用户提示词",
            pc.get('user_template', ''),
            height=250,
            disabled=True,
            key="user_prompt_view"
        )


def _save_prompt_config(pc, sys_t):
    """保存提示词配置 - 升级版"""
    from config.constants import DEFAULT_USER_TEMPLATE

    if sys_t == pc.get('system_template'):
        st.info("💡 内容没有变化，无需保存。")
        return

    new_cfg = {
        "system_template": sys_t,
        "user_template": pc.get('user_template', DEFAULT_USER_TEMPLATE)
    }

    # 保存到session
    st.session_state.prompt_config = new_cfg

    # 保存到文件
    from config.settings import PATHS
    with open(PATHS.prompt_config_file, 'w', encoding='utf-8') as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    st.success("✅ 成功保存提示词配置！")
    st.balloons()


def _restore_default_prompt():
    """恢复默认提示词 - 升级版"""
    from config.settings import PATHS
    from core.resource_manager import ResourceManager

    default_cfg = None
    if PATHS.default_prompts.exists():
        default_cfg = ResourceManager.load_external_json(PATHS.default_prompts)

    if not default_cfg:
        st.error("❌ 未找到默认提示词配置文件")
        return

    if default_cfg and 'system_template' in default_cfg:
        st.session_state.prompt_config['system_template'] = default_cfg['system_template']

        with open(PATHS.prompt_config_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.prompt_config, f, ensure_ascii=False, indent=2)

        st.success("✅ 已恢复至初始设定")
        time.sleep(0.5)
        st.rerun()


def _show_prompt_comparison(original, modified):
    """显示提示词对比 - 升级版"""
    import time

    st.markdown("##### 📊 提示词对比")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**原始版本**")
        st.code(original, language=None, height=300)

    with col2:
        st.markdown("**修改版本**")
        st.code(modified, language=None, height=300)

    # 差异统计
    if original and modified:
        diff_chars = len(modified) - len(original)
        diff_lines = len(modified.split('\n')) - len(original.split('\n'))

        st.markdown(f"""
        <div style="padding: 10px; background: #EDF5EB; border-radius: 6px; margin-top: 15px;">
            <span style="color: #4A5D53; font-weight: 600;">📊 变更统计：</span>
            <span style="color: #666;">字符数 {diff_chars:+d} | 行数 {diff_lines:+d}</span>
        </div>
        """, unsafe_allow_html=True)
