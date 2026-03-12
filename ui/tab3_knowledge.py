"""
tab3_knowledge.py
==================
知识库设计 Tab - 本地模式
"""

import streamlit as st
import os
from pathlib import Path


def render_tab3(aliyun_key):
    """渲染知识库设计 Tab - 本地模式"""
    with st.container():
        # 1. 本地知识库文件列表
        st.markdown("##### 📁 本地知识库文件")
        _render_local_file_list()

        st.markdown("---")

        # 2. 添加新文件
        st.markdown("##### ➕ 添加新文件")
        _render_upload_section(aliyun_key)

        st.markdown("---")

        # 3. 手动维护
        st.markdown("##### 🔧 手动维护")
        _render_maintenance_section()


def _get_local_files():
    """获取本地 RAG 目录中的文件列表"""
    from config.settings import PATHS

    if not PATHS.RAG_DIR.exists():
        return []

    files = []
    for file_path in PATHS.RAG_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in ['.pdf', '.txt', '.docx']:
            files.append({
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'path': file_path
            })

    return sorted(files, key=lambda x: x['name'])


def _render_local_file_list():
    """渲染本地文件列表"""
    from config.settings import PATHS

    # 刷新文件列表
    if 'refresh_local_files' in st.session_state and st.session_state.refresh_local_files:
        st.session_state.local_rag_files = _get_local_files()
        st.session_state.refresh_local_files = False

    # 初始化文件列表
    if 'local_rag_files' not in st.session_state:
        st.session_state.local_rag_files = _get_local_files()

    local_files = st.session_state.local_rag_files

    if not local_files:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <div class="empty-state-text">暂无本地文件</div>
            <div style="font-size: 0.85rem; color: #999; margin-top: 0.5rem;">
                请上传文件到知识库
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 文件统计
    st.markdown(f"""
    <div style="padding: 10px; background: #EDF5EB; border-radius: 6px; margin-bottom: 15px;">
        <span style="color: #4A5D53; font-weight: 600;">📊 共 {len(local_files)} 个文件</span>
    </div>
    """, unsafe_allow_html=True)

    # 文件列表
    for idx, file_info in enumerate(local_files):
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 2, 1])

            with col1:
                st.markdown(f"**{file_info['name']}**")
                st.caption(f"{file_info['size'] / 1024:.1f} KB")

            with col2:
                st.caption("本地文件")

            with col3:
                if st.button("🗑️", key=f"del_local_{idx}", help="删除文件"):
                    try:
                        os.remove(file_info['path'])
                        st.success(f"✅ 已删除 {file_info['name']}")
                        st.session_state.refresh_local_files = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败: {str(e)}")


def _render_upload_section(aliyun_key):
    """渲染上传区域"""
    st.caption("支持格式：PDF、TXT、DOCX")

    # 上传区域
    up = st.file_uploader(
        "选择文件",
        accept_multiple_files=True,
        key="kb_uploader",
        type=['pdf', 'txt', 'docx'],
        help="可一次选择多个文件"
    )

    # 显示已选文件信息
    if up:
        st.markdown(f"""
        <div style="padding: 10px; background: #FDF6ED; border-radius: 6px; margin: 10px 0;">
            <span style="color: #8B5A2B; font-weight: 600;">📋 已选择 {len(up)} 个文件</span>
        </div>
        """, unsafe_allow_html=True)

        # 文件列表
        for file in up:
            st.markdown(f"- {file.name} ({file.size / 1024:.1f} KB)")

    # 上传按钮 - 始终显示
    if st.button("📤 添加到知识库", type="primary", width='stretch'):
        if not up or len(up) == 0:
            st.warning("⚠️ 请先选择要上传的文件")
        else:
            _handle_upload(up, aliyun_key)


def _render_maintenance_section():
    """渲染手动维护区域"""
    from config.settings import PATHS

    st.caption("管理本地知识库文件")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 刷新文件列表", width='stretch'):
            st.session_state.local_rag_files = _get_local_files()
            st.success(f"✅ 已刷新文件列表，共 {len(st.session_state.local_rag_files)} 个文件")
            st.rerun()

    with col2:
        if st.button("🗑️ 清空知识库", width='stretch'):
            try:
                import shutil

                # 1. 清空本地知识库文件
                if PATHS.kb_index.exists():
                    os.remove(PATHS.kb_index)
                if PATHS.kb_chunks.exists():
                    os.remove(PATHS.kb_chunks)

                # 2. 清空 session_state 中的知识库数据
                st.session_state.kb = (None, [])

                # 3. 重置加载状态
                st.session_state.rag_loading_needed = True
                st.session_state.rag_loading_status = "pending"

                # 4. 立即刷新
                st.success("✅ 知识库已清空")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 清空失败: {str(e)}")


def _handle_upload(files, aliyun_key):
    """处理文件上传 - 纯本地版"""
    import io
    import pickle
    from typing import List

    # 检查 API Key
    if not aliyun_key:
        st.error("❌ 请先配置阿里云 API Key")
        st.info("💡 在侧边栏的「API 配置」中配置 ALIYUN_API_KEY")
        return

    # 导入必要的模块
    from config.settings import PATHS
    from core.ai_services import AliyunEmbedder
    from core.resource_manager import ResourceManager
    import numpy as np
    import faiss

    # 显示处理进度
    with st.spinner("🔄 正在处理文件，请稍候..."):
        try:
            # ===== 步骤 1：保存文件 =====
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            saved_files = []
            for uploaded_file in files:
                file_path = PATHS.RAG_DIR / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)

            # ===== 步骤 2：解析文件 =====
            all_chunks = []
            for uploaded_file in files:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                    elif uploaded_file.name.endswith('.pdf'):
                        import PyPDF2
                        pdf_file = io.BytesIO(uploaded_file.getvalue())
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    elif uploaded_file.name.endswith('.docx'):
                        import docx
                        doc_file = io.BytesIO(uploaded_file.getvalue())
                        doc = docx.Document(doc_file)
                        text = "\n".join([para.text for para in doc.paragraphs])
                    else:
                        continue

                    # 文本分块
                    chunks = []
                    chunk_size = 500
                    overlap = 50
                    start = 0
                    text_length = len(text)

                    while start < text_length:
                        end = start + chunk_size
                        chunk = text[start:end].strip()
                        if chunk:
                            chunks.append(chunk)
                        start = end - overlap

                    all_chunks.extend(chunks)

                except Exception as e:
                    continue

            if not all_chunks:
                st.error("❌ 未能从文件中提取到任何文本内容")
                st.info("💡 请确认文件格式正确（PDF/TXT/DOCX）且包含可提取的文字")
                return

            # ===== 步骤 3：向量化处理 =====
            embedder = AliyunEmbedder(aliyun_key)
            embeddings = []
            batch_size = 5  # 小批量处理，避免超时

            total_chunks = len(all_chunks)
            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i+batch_size]
                try:
                    batch_embeddings = embedder.embed_texts(batch)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    st.error("❌ 向量化失败")
                    st.error(f"错误信息: {str(e)}")
                    st.info("💡 可能的原因：API 配额用完、网络连接问题、API Key 无效")
                    return

            # ===== 步骤 4：保存索引 =====
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]

            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            faiss.write_index(index, str(PATHS.kb_index))
            with open(PATHS.kb_chunks, 'wb') as f:
                pickle.dump(all_chunks, f)

            # 更新 session_state
            st.session_state.kb = (index, all_chunks)

            # 更新加载状态为 complete（表示有知识库了）
            st.session_state.rag_loading_status = 'complete'
            st.session_state.rag_loading_needed = True

            # 更新文件列表
            kb_files = ResourceManager.load_kb_files()
            new_kb_files = list(set(kb_files + saved_files))
            ResourceManager.save_kb_files(new_kb_files)
            st.session_state.kb_files = new_kb_files

            # 刷新本地文件列表
            st.session_state.refresh_local_files = True

        except Exception as e:
            import traceback
            st.error("❌ 处理失败")
            st.error(f"错误信息: {str(e)}")
            traceback.print_exc()
            return

    # ===== 显示最终结果 =====
    st.success(f"✅ 成功添加 {len(saved_files)} 个文件到知识库")
    st.success(f"📊 提取了 {len(all_chunks)} 个知识片段")
    st.info("💡 知识库已更新，现在可以使用这些文件进行检索")
