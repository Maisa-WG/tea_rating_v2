"""
sidebar.py
===========
侧边栏 UI 组件 - 控制台风格
"""

import os
import time
import streamlit as st
import requests
from openai import OpenAI

from config.settings import PATHS
from config.constants import TEA_EXAMPLES
from core.resource_manager import ResourceManager


def render_sidebar():
    """
    渲染侧边栏 - 控制台风格

    Returns:
        tuple: (embedder, client, client_d, model_id)
    """
    with st.sidebar:
        # 系统配置区域
        st.markdown("**⚙️ 系统配置**")
        st.markdown("🔐 API 配置")

        aliyun_key = os.getenv("ALIYUN_API_KEY") or st.secrets.get("ALIYUN_API_KEY", "")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")

        if not aliyun_key or not deepseek_key:
            st.warning("未配置 API Key")
            st.stop()
        else:
            st.success("✅ API 就绪")

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 模型配置 - 简化版
        st.markdown("**模型配置**")
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>预处理：Deepseek-chat</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666;'>评分：Qwen3-14B</div>", unsafe_allow_html=True)
        model_id = "Qwen3-14B"

        # ========== 缓存微调模型状态 ==========
        if 'finetune_status_cache' not in st.session_state:
            st.session_state.finetune_status_cache = {
                'model_id': model_id,
                'ft_status': None,
                'last_check': None
            }

        ft_cache = st.session_state.finetune_status_cache

        # 只在首次或超过5分钟时检查
        should_check = (
            ft_cache['ft_status'] is None or
            (time.time() - ft_cache['last_check']) > 300  # 5分钟
        )

        if should_check:
            try:
                resp = requests.get("http://117.50.138.123:8001/status", timeout=2)
                if resp.status_code == 200 and resp.json().get("lora_available"):
                    model_id = "default_lora"
                    ft_cache['model_id'] = model_id
                    st.success("已启用微调模型")
            except:
                pass

            ft_cache['ft_status'] = ResourceManager.load_ft_status()
            ft_cache['last_check'] = time.time()
        else:
            # 从缓存读取
            if ft_cache['model_id'] == "default_lora":
                st.success("已启用微调模型")
            model_id = ft_cache['model_id']

        # 显示微调模型信息
        if ft_cache['ft_status'] and ft_cache['ft_status'].get("status") == "succeeded":
            st.info(f"发现微调模型：`{ft_cache['ft_status'].get('fine_tuned_model')}`")

        # 初始化服务实例
        from core.ai_services import AliyunEmbedder
        embedder = AliyunEmbedder(aliyun_key)
        client = OpenAI(
            api_key="dummy",
            base_url="http://117.50.138.123:8000/v1",
            timeout=120.0  # 120秒默认超时
        )
        client_d = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
            timeout=60.0  # 60秒默认超时
        )

        # RAG 延迟加载
        _handle_rag_loading(aliyun_key)

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # GitHub 配置状态
        st.markdown("**🔄 GitHub 备份**")

        # ========== 缓存机制 ==========
        # 初始化缓存状态
        if 'github_status_cache' not in st.session_state:
            st.session_state.github_status_cache = {
                'configured': False,
                'msg': '',
                'repo_info': None,
                'last_check': None,
                'needs_refresh': True  # 首次需要检查
            }

        # 检查是否需要刷新
        cache = st.session_state.github_status_cache
        if cache['needs_refresh']:
            with st.spinner("🔄 检查 GitHub 配置..."):
                github_configured, github_msg = _check_github_status()
                repo_info = _get_github_repo_info() if github_configured else None

                # 更新缓存
                cache['configured'] = github_configured
                cache['msg'] = github_msg
                cache['repo_info'] = repo_info
                cache['last_check'] = time.time()
                cache['needs_refresh'] = False

        # 从缓存读取
        github_configured = cache['configured']
        github_msg = cache['msg']
        repo_info = cache['repo_info']

        # ========== 显示状态 ==========
        if github_configured:
            st.success("✅ " + github_msg)

            # 显示仓库信息
            if repo_info:
                st.markdown(f"<div style='font-size: 0.75rem; color: #666; margin-top: 0.25rem;'>📁 {repo_info['full_name']}</div>", unsafe_allow_html=True)
                if repo_info.get('private'):
                    st.markdown(f"<div style='font-size: 0.75rem; color: #666;'>🔒 私有仓库</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size: 0.75rem; color: #666;'>🌐 公开仓库</div>", unsafe_allow_html=True)

            # 同步按钮和刷新按钮
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("📤 同步到 GitHub", use_container_width=True, key="sync_to_github"):
                    _handle_github_sync()
            with col2:
                if st.button("🔄", use_container_width=True, help="刷新状态"):
                    st.session_state.github_status_cache['needs_refresh'] = True
                    st.rerun()

            # 显示最后检查时间
            if cache['last_check']:
                from datetime import datetime
                last_check_time = datetime.fromtimestamp(cache['last_check']).strftime("%H:%M:%S")
                st.caption(f"最后检查: {last_check_time}")
        else:
            st.warning("⚠️ " + github_msg)
            st.markdown(f"<div style='font-size: 0.75rem; color: #999; margin-top: 0.25rem;'>在 .streamlit/secrets.toml 中配置</div>", unsafe_allow_html=True)

            # 刷新按钮
            if st.button("🔄 刷新", use_container_width=True, key="refresh_github_warn"):
                st.session_state.github_status_cache['needs_refresh'] = True
                st.rerun()

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 数据概览 - 简化版
        st.markdown("**数据概览**")

        kb_count = len(st.session_state.kb[1])
        basic_count = len(st.session_state.basic_cases)
        supp_count = len(st.session_state.supp_cases[1])

        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>知识库：{kb_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>基础判例：{basic_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>进阶判例：{supp_count}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 茶评示例区域
        st.markdown("**📚 茶评示例**")

        if st.button("茶评示例", use_container_width=True):
            st.session_state.show_tea_examples = True

    return embedder, client, client_d, model_id


def _handle_rag_loading(aliyun_key: str):
    """处理 RAG 延迟加载逻辑 - 纯本地模式"""
    from config.settings import PATHS

    # 检查知识库是否有数据
    kb_has_data = st.session_state.kb[1] is not None and len(st.session_state.kb[1]) > 0
    loading_status = st.session_state.get('rag_loading_status', 'pending')
    rag_loading_needed = st.session_state.get('rag_loading_needed', False)

    # 获取本地 RAG 目录中的文件数量
    try:
        local_files = list(PATHS.RAG_DIR.glob("*.txt")) + \
                     list(PATHS.RAG_DIR.glob("*.pdf")) + \
                     list(PATHS.RAG_DIR.glob("*.docx"))
        local_file_count = len(local_files) if PATHS.RAG_DIR.exists() else 0
    except:
        local_file_count = 0

    # 检查本地索引文件是否存在
    local_index_exists = PATHS.kb_index.exists()

    # ========== 本地模式检测逻辑 ==========
    # 情况1：加载失败（最高优先级）
    if loading_status == 'failed':
        st.warning("⚠️ 知识库加载失败")
        if st.button("🔄 重试加载", type="secondary", use_container_width=True):
            st.session_state.rag_loading_status = 'pending'
            st.session_state.rag_loading_needed = True
            st.rerun()
        return

    # 情况2：正在加载中
    if loading_status == 'loading':
        st.info("🔄 正在加载知识库，请稍候...")
        return

    # 情况3：知识库有数据 → 显示成功状态
    if kb_has_data:
        st.success("✅ 知识库加载成功")
        kb_count = len(st.session_state.kb[1])
        st.caption(f"📊 已加载 {kb_count} 个知识片段")
        st.caption(f"📁 本地 {local_file_count} 个文件")

        if st.button("🔄 重新加载", use_container_width=True):
            st.session_state.rag_loading_status = 'pending'
            st.session_state.rag_loading_needed = True
            st.rerun()
        return

    # 情况4：知识库为空，但需要加载 → 开始加载
    if rag_loading_needed and loading_status == 'pending':
        with st.status("🔄 正在加载知识库...", expanded=True) as status:
            st.write("📂 读取本地 RAG 文件...")
            st.session_state.rag_loading_status = 'loading'

            try:
                from core import load_rag_from_local
                success, msg = load_rag_from_local(aliyun_key)

                if success:
                    status.update(label="✅ 知识库加载完成", state="complete", expanded=False)
                    st.session_state.rag_loading_status = 'complete'
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="❌ 知识库加载失败", state="error", expanded=True)
                    st.error(msg)
                    st.info("💡 您可以在「知识库设计」手动上传 RAG 文件")
                    st.session_state.rag_loading_status = 'failed'

                    if st.button("🔄 重试加载", type="secondary"):
                        st.session_state.rag_loading_status = 'pending'
                        st.rerun()

            except Exception as e:
                status.update(label="❌ 加载出错", state="error", expanded=True)
                st.error(f"加载失败: {str(e)}")
                st.session_state.rag_loading_status = 'failed'

                if st.button("🔄 重试加载", type="secondary"):
                    st.session_state.rag_loading_status = 'pending'
                    st.rerun()
        return

    # 情况5：知识库为空且不需要加载 → 智能提示
    if not kb_has_data and not rag_loading_needed:
        # 情况5a：本地有文件，但无索引
        if local_file_count > 0 and not local_index_exists:
            st.info(f"📂 发现本地有 {local_file_count} 个文件")
            if st.button("📥 加载知识库", type="primary", use_container_width=True):
                st.session_state.rag_loading_needed = True
                st.session_state.rag_loading_status = 'pending'
                st.rerun()
            return

        # 情况5b：本地无文件
        if local_file_count == 0:
            st.info("💡 请在「知识库设计」上传文件")
            return

        # 情况5c：其他情况
        st.info("💡 请在「知识库设计」添加文件或点击加载")


def _check_github_status():
    """检查 GitHub 配置状态"""
    try:
        from core.github_sync import GithubSync
        return GithubSync.check_config()
    except Exception as e:
        return False, f"检查失败: {str(e)}"


def _get_github_repo_info():
    """获取 GitHub 仓库信息"""
    try:
        from core.github_sync import GithubSync
        return GithubSync.get_repo_info()
    except Exception as e:
        return None


def _handle_github_sync():
    """处理 GitHub 同步"""
    from core.github_sync import GithubSync

    with st.spinner("🔄 正在同步到 GitHub，请稍候..."):
        try:
            success, msg, synced_files = GithubSync.sync_all_data(st.session_state)

            if success:
                st.success(f"🎉 同步完成！已同步 {len(synced_files)} 个文件")
                # 同步成功后刷新 GitHub 状态缓存
                st.session_state.github_status_cache['needs_refresh'] = True
            else:
                st.error("❌ 同步失败")
                if synced_files:
                    st.info(f"💡 部分成功：{len(synced_files)} 个文件已同步")

        except Exception as e:
            st.error(f"❌ 同步失败: {str(e)}")
