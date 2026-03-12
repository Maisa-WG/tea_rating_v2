# 🍵 茶品六因子 AI 评分器 Pro

> 基于大语言模型的智能茶饮感官评分系统 - 模块化重构版

[![Python](https://img.shields.io/badge/Python-3.10-blue)]
[![Streamlit](https://img.shields.io/badge/StreamLit-1.55-red)]
[![License](https://img.shields.io/badge/License-MIT-green)]

---

## 📖 项目简介

**茶品六因子 AI 评分器 Pro** 是一款基于大语言模型和检索增强生成（RAG）技术的智能茶饮感官评分系统。系统采用"罗马测评法2.0"，通过六因子（优雅性、辨识度、协调性、饱和度、持久性、苦涩度）对茶饮进行专业化评分，为茶饮产品研发和质量控制提供科学依据。

本项目已进行**模块化重构**，代码结构清晰，易于维护和扩展。

### 核心特性

- 🤖 **AI 智能评分**：基于 LLM 的自动化感官评分，支持多种主流模型 API
- 📚 **知识库管理**：支持 PDF/Word/TXT 文��的知识库构建，GraphRAG 知识图谱增强检索
- 📊 **批量处理**：支持批量茶评文档处理，自动生成评分报告
- 🔧 **判例库管理**：基础判例和进阶判例的双库管理，向量检索相似案例
- ⚙️ **模型微调**：支持 LoRA 模型微调，自动启用微调后的模型
- 🔄 **GitHub 全量同步**：自动同步所有项目文件到云端，支持增量备份
- 📈 **中国风可视化**：雷达图、风味形态图等专业可视化，完美支持中文显示
- 🎨 **茶文化主题 UI**：精心设计的中国风配色和界面，东方美学设计
- 💡 **Tree of Thoughts**：采用思维树决策模式，多路径评估提升评分准确性
- 🚀 **性能优化**：session_state 缓存机制，避免重复网络请求，提升响应速度
- ✨ **智能建议生成**：AI 基于专业知识为每个因子生成个性化改进建议

---

## 🛠️ 技术栈

### 后端框架
- **Streamlit** 1.55 - Web 应用框架
- **Python** 3.10 - 编程语言

### AI & ML
- **OpenAI 兼容 API** - GPT-4 等模型推理
- **阿里云 Dashscope** - 文本嵌入服务（向量化）
- **LangChain** - LLM 应用框架
- **FAISS** - 向量检索引擎
- **Sentence-Transformers** - 句子嵌入模型
- **GraphRAG** - 知识图谱增强检索

### 数据处理
- **Pandas** - 数据分析
- **NumPy** - 数值计算
- **SciPy** - 科学计算和样条插值
- **OpenPyXL** - Excel 文件处理

### 可视化
- **Plotly** - 交互式图表（雷达图）
- **Matplotlib** - 静态图表（风味形态图，支持中文）

### 其他
- **PyGithub** - GitHub API 集成
- **PyPDF2** - PDF 解析
- **python-docx** - Word 文档处理

---

## 📁 项目结构

```
tea_rating_v2-main/
│
├── main.py                          # ⭐ 应用主入口
│
├── config/                          # 🔧 配置模块
│   ├── __init__.py                   # 模块初始化
│   ├── settings.py                   # 页面配置、中国风 CSS 样式、PathConfig 路径管理类
│   └── constants.py                  # 全局常量（六因子定义、茶评示例、默认提示词模板）
│
├── core/                            # ⚙️ 核心业务逻辑
│   ├── __init__.py                   # 模块初始化，导出所有核心类和函数
│   ├── resource_manager.py          # ResourceManager 类：文件读写、数据持久化、微调数据管理
│   ├── github_sync.py               # GithubSync 类：GitHub 全量文件同步（递归遍历、排除模式、缓存）
│   ├── ai_services.py               # AliyunEmbedder 类、llm_normalize_user_input 函数
│   ├── scoring.py                   # run_scoring 函数：核心评分逻辑（RAG + LLM）
│   └── init.py                       # 初始化函数（bootstrap_cases、load_rag_from_github 等）
│
├── data/                            # 📊 数据处理模块
│   ├── __init__.py                   # 模块初始化
│   ├── excel_parser.py              # Excel 解析公共代码（复用于三个处理器）
│   ├── basic_case_processor.py      # 基础判例 Excel 批量导入
│   ├── supplementary_processor.py   # 进阶判例 Excel 批量导入
│   └── finetune_processor.py        # 微调数据 Excel 批量导入
│
├── storage/                         # 💾 数据存储模块
│   ├── __init__.py                   # 模块初始化
│   ├── database.py                  # CSV 数据库操作（initial_case.csv、adjusted_case.csv）
│   └── vector_store.py              # 向量索引存储（FAISS：vectors.npy、vectors_meta.json）
│
├── retrieval/                       # 🔍 检索与推理模块
│   ├── __init__.py                   # 模块初始化
│   ├── graphrag_retriever.py        # GraphRAG 检索器（知识图谱增强检索）
│   └── logic.py                     # 评分推理逻辑（fetch_evaluation 函数等）
│
├── ui/                              # 🎨 UI 组件模块
│   ├── __init__.py                   # 模块初始化
│   ├── sidebar.py                   # 侧边栏（API 配置、模型状态、数据统计、性能缓存）
│   ├── dialogs.py                   # 弹窗组件（判例编辑、提示词查看、茶评示例管理）
│   ├── tab1_interactive.py          # Tab1: 交互评分（单个评分、可视化展示）
│   ├── tab2_batch.py                # Tab2: 批量评分（批量处理、报告生成）
│   ├── tab3_knowledge.py            # Tab3: 知识库管理（文件上传、GraphRAG）
│   ├── tab4_cases.py                # Tab4: 判例库管理（双库管理、向量检索）
│   ├── tab5_finetune.py             # Tab5: 模型微调（数据准备、训练部署）
│   └── tab6_prompts.py              # Tab6: 提示词配置（系统提示词编辑）
│
├── utils/                           # 🛠️ 工具函数模块
│   ├── __init__.py                   # 模块初始化
│   ├── visualization.py             # 可视化函数（雷达图、风味图、中国风配色、中文字体支持）⭐
│   └── helpers.py                   # 辅助工具（文件解析、Word 报告生成等）
│
├── .streamlit/                       # Streamlit 配置
│   └── secrets.toml                  # API 密钥配置（ALIYUN_API_KEY、等）
│
├── tea_data/                        # 💾 运行时数据目录
│   ├── basic_case.json               # 基础判例库数据
│   ├── supplementary_case.json       # 进阶判例库数据
│   ├── supp_cases.index             # 进阶判例向量索引（FAISS）
│   ├── eval_logs.json               # 评分日志记录
│   ├── prompts.json                 # 提示词配置存储
│   ├── tea_examples.json            # 茶评示例配置
│   ├── kb_files.json                # 知识库文件清单
│   ├── RAG/                         # 知识库原始文件（PDF/TXT/DOCX）
│   └── graphrag_artifacts/          # GraphRAG 知识图谱缓存
│
├── tea_backup/                      # 📦 备份数据目录
│   ├── template.xlsx                # 判例模板文件
│   └── default_prompts.json         # 默认提示词配置
│
├── requirements.txt                 # Python 依赖包列表
├── sys_p.txt                        # 系统提示词（参考）
├── runtime.txt                      # Runtime 配置
└── README.md                        # 📖 项目说明文档
```

### 目录说明

| 目录/文件 | 说明 | 重要性 |
|----------|------|--------|
| `main.py` | 应用入口，初始化 session_state，渲染 UI | ⭐⭐⭐ |
| `core/` | 核心业务逻辑，评分、同步、资源管理 | ⭐⭐⭐ |
| `ui/` | 用户界面组件，6 个功能 Tab | ⭐⭐⭐ |
| `utils/visualization.py` | 可视化函数（中国风配色） | ⭐⭐⭐ |
| `config/` | 配置和常量 | ⭐⭐ |
| `data/` | 数据处理模块 | ⭐⭐ |
| `storage/` | 数据存储 | ⭐⭐ |
| `retrieval/` | 检索和推理 | ⭐⭐ |
| `tea_data/` | 运行时数据（自动生成） | ⭐ |
| `.streamlit/` | Streamlit 配置（需手动创建） | ⭐ |

---

## 📦 安装指南

### 环境要求

- Python 3.10
- Anaconda 或 Miniconda
- Windows 11 / macOS / Linux

### 安装步骤

#### 1. 创建 Conda 环境

```powershell
# 在 Anaconda PowerShell Prompt 中执行
conda create -n allenv python=3.10 -y
conda activate allenv
```

#### 2. 安装依赖

```powershell
cd D:\jinghan_research\tea_rating_v2-main
pip install -r requirements.txt
```

#### 3. 配置 API 密钥

创建 `.streamlit/secrets.toml` 文件：

```toml
# 阿里云 Dashscope API Key（用于文本嵌入和向量化）
ALIYUN_API_KEY = "sk-你的阿里云API密钥"

# OpenAI 兼容 API Key（用于文本预处理和推理）
# 支持多种兼容 OpenAI API 格式的服务
OPENAI_API_KEY = "sk-你的API密钥"
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可选，自定义 API 端点

# GitHub 配置（可选，用于云端全量备份）
# GITHUB_TOKEN = "你的GitHub令牌"
# GITHUB_REPO = "用户名/仓库名"
# GITHUB_BRANCH = "main"
```

### 获取 API 密钥

1. **阿里云 Dashscope**: https://dashscope.console.aliyun.com/apiKey
2. **OpenAI**: https://platform.openai.com/api-keys
3. **其他兼容服务**:
   - : https://platform./api_keys
   - 或其他支持 OpenAI API 格式的服务

---

## 🚀 快速开始

### 启动应用

```powershell
# 在 Anaconda PowerShell Prompt 中
conda activate allenv
cd D:\jinghan_research\tea_rating_v2-main
streamlit run main.py
```

### 访问应用

- **本地访问**: http://localhost:8501
- **局域网访问**: http://[你的IP地址]:8501

---

## 📚 功能使用指南

### 1. 交互评分（Tab 1）

**功能**: 单个茶品 AI 智能评分

**操作步骤**:
1. 设置参考知识库数量和判例库数量
2. 输入茶评描述（支持粘贴茶评文本）
3. 点击"开始评分"按钮
4. 查看评分结果：
   - 六因子得分（1-9分）
   - 雷达图可视化
   - 风味形态图（香/味/韵三段）
   - 宗师级总评（东方美学风格）
   - 每个因子的个性化建议
5. 可选：校准评分并保存到判例库

**特性**:
- 支持 Tree of Thoughts 多路径评估
- 自动文本预处理和规范化
- 中文完美显示（中国风配色）
- AI 生成专业改进建议
- 简洁界面，专注评分体验

### 2. 批量评分（Tab 2）

**功能**: 批量处理茶评文档，自动生成评分报告

**支持格式**: `.txt`, `.docx`

**输出**: Word 格式评分报告，包含所有评分结果和可视化图表

### 3. 知识库管理（Tab 3）

**功能**: 管理和同步知识库文件

**支持格式**: `.pdf`, `.txt`, `.docx`

**特性**:
- 查看 GitHub 云端文件列表
- 上传新文件到知识库
- 删除已有文件
- 自动向量化和索引构建
- GraphRAG 知识图谱自动生成

### 4. 判例库管理（Tab 4）

**功能**: 管理基础判例和进阶判例

**基础判例**:
- 手动添加单个判例
- 批量导入 Excel 文件
- 查看、编辑判例内容

**进阶判例**:
- 手动添加单个判例
- 批量导入 Excel 文件
- 向量检索相似案例（FAISS）
- 按相似度排序

### 5. 模型微调（Tab 5）

**功能**: LoRA 模型微调和部署

**操作流程**:
1. 准备微调数据（Excel 格式）
2. 生成微调数据 JSONL 文件
3. 上传至 GPU 服务器
4. 启动 LoRA 训练
5. 自动检测并启用微调模型

**特性**:
- 自动状态检测
- 微调模型优先使用
- 支持手动切换模型

### 6. 提示词配置（Tab 6）

**功能**: 自定义系统提示词

**说明**:
- 系统提示词可编辑（评分规则、因子定义等）
- 用户提示词固定（确保输出格式一致性）
- 支持恢复默认配置
- 实时预览提示词内容

---

## 🎯 六因子评分体系

系统采用"罗马测评法2.0"，包含以下六个因子：

### 前段：香
1. **优雅性** (1-9分) - 香气引发的愉悦感
2. **辨识度** (1-9分) - 香气可被识别记忆

### 中段：味
3. **协调性** (1-9分) - 茶汤内含物的融合度
4. **饱和度** (1-9分) - 整体茶汤的浓厚度

### 后段：韵
5. **持久性** (1-9分) - 茶汤在口腔中的余韵
6. **苦涩度** (1-9分) - 苦味、收敛拉扯感（舒适度越高分数越高）

### 评分特点

- **1-9 分制**：更精细的评分粒度
- **证据驱动**：每个分数都有对应的���论依据
- **保守原则**：信息不足时默认中性分 4
- **多路径评估**：Tree of Thoughts 决策模式
- **智能建议**：AI 基于专业知识生成个性化改进建议

---

## 🏗️ 模块说明

### config/ - 配置模块

| 文件 | 主要类/函数 | 说明 |
|------|-----------|------|
| `settings.py` | `PathConfig`, `apply_page_config()`, `apply_css_styles()` | 页面配置、中国风 CSS 样式、路径管理类 |
| `constants.py` | `FACTORS`, `TEA_EXAMPLES`, `DEFAULT_SYSTEM_TEMPLATE` | 全局常量（六因子定义、茶评示例、默认提示词模板） |

### core/ - 核心业务逻辑

| 文件 | 主要类/函数 | 说明 |
|------|-----------|------|
| `resource_manager.py` | `ResourceManager` | 资源管理器：文件读写、数据持久化、微调数据管理、茶评示例管理 |
| `github_sync.py` | `GithubSync` | GitHub 同步工具：全量文件同步、递归遍历、排除模式、性能缓存 |
| `ai_services.py` | `AliyunEmbedder`, `llm_normalize_user_input()` | AI 服务：阿里云嵌入、文本预处理和规范化 |
| `scoring.py` | `run_scoring()` | 核心评分逻辑：整合 RAG 检索、判例匹配、LLM 推理，支持向量索引自动重建和缓存 |
| `init.py` | `bootstrap_cases()`, `load_rag_from_github()` | 初始化函数：判例加载、RAG 初始化、GitHub 数据加载 |

### data/ - 数据处理模块

| 文件 | 主要类/函数 | 说明 |
|------|-----------|------|
| `excel_parser.py` | `parse_excel_file()` | Excel 解析公共代码（复用于三个处理器） |
| `basic_case_processor.py` | `process_basic_case_excel()` | 基础判例 Excel 批量导入 |
| `supplementary_processor.py` | `process_supp_case_excel()` | 进阶判例 Excel 批量导入 |
| `finetune_processor.py` | `process_finetune_excel()` | 微调数据 Excel 批量导入 |

### storage/ - 数据存储模块

| 文件 | 主要类/函数 | 说明 |
|------|-----------|------|
| `database.py` | `save_initial()`, `save_adjusted()`, `load_initial()`, `load_adjusted()` | CSV 数据库操作（初始评分、校准评分） |
| `vector_store.py` | `save_vectors()`, `load_vectors()` | 向量索引存储（FAISS：vectors.npy、vectors_meta.json） |

### retrieval/ - 检索与推理模块

| 文件 | 主要类/函数 | 说明 |
|------|-----------|------|
| `graphrag_retriever.py` | `GraphRAGRetriever` | GraphRAG 检索器：知识图谱增强检索、社区发现 |
| `logic.py` | `fetch_evaluation()` | 评分推理逻辑：从历史评分中提取相似评估 |

### ui/ - UI 组件模块

| 文件 | 主要函数 | 说明 |
|------|---------|------|
| `sidebar.py` | `render_sidebar()` | 侧边栏：API 配置检查、模型状态检测、数据统计、性能缓存 |
| `dialogs.py` | `show_prompt_dialog()`, `show_tea_examples_dialog()`, `manage_tea_examples_dialog()`, `edit_tea_example_dialog()` | 弹窗组件：提示词查看、茶评示例展示和管理、判例编辑 |
| `tab1_interactive.py` | `render_tab1()` | Tab1 交互评分：单个评分、可视化展示、校准保存、建议显示 |
| `tab2_batch.py` | `render_tab2()` | Tab2 批量评分：批量处理、Word 报告生成 |
| `tab3_knowledge.py` | `render_tab3()` | Tab3 知识库管理：文件上传、删除、GraphRAG 同步 |
| `tab4_cases.py` | `render_tab4()` | Tab4 判例库管理：双库管理、向量检索、Excel 导入 |
| `tab5_finetune.py` | `render_tab5()` | Tab5 模型微调：数据准备、JSONL 生成、训练部署 |
| `tab6_prompts.py` | `render_tab6()` | Tab6 提示词配置：系统提示词编辑、恢复默认 |

### utils/ - 工具函数模块

| 文件 | 主要函数 | 说明 |
|------|---------|------|
| `visualization.py` | `plot_radar_chart()`, `plot_flavor_shape()`, `calculate_section_scores()` | 可视化函数：雷达图（Plotly）、风味形态图（Matplotlib）、中国风配色、中文字体支持、嵌套数据结构处理 |
| `helpers.py` | `parse_pdf()`, `parse_docx()`, `generate_word_report()` | 辅助工具：文件解析、Word 报告生成 |

### tea_data/ - 数据文件说明

| 文件 | 说明 | 格式 |
|------|------|------|
| `basic_case.json` | 基础判例库数据 | JSON |
| `supplementary_case.json` | 进阶判例库数据 | JSON |
| `supp_cases.index` | 进阶判例向量索引（FAISS） | 二进制 |
| `eval_logs.json` | 评分日志记录 | JSON |
| `prompts.json` | 提示词配置存储 | JSON |
| `tea_examples.json` | 茶评示例配置 | JSON |
| `kb_files.json` | 知识库文件清单 | JSON |
| `RAG/` | 知识库原始文件 | PDF/TXT/DOCX |
| `graphrag_artifacts/` | GraphRAG 知识图谱缓存 | JSON/PKL |

---

## ⚙️ 配置说明

### API 密钥配置

在 `.streamlit/secrets.toml` 中配置：

```toml
# 必需配置
ALIYUN_API_KEY = "sk-你的阿里云密钥"
OPENAI_API_KEY = "sk-你的OpenAI密钥"
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可选

# 可选配置（GitHub 全量备份）
GITHUB_TOKEN = "你的GitHub令牌"
GITHUB_REPO = "用户名/仓库名"
GITHUB_BRANCH = "main"
```

### GraphRAG 配置

GraphRAG 知识图谱会在首次使用时自动构建并缓存到 `tea_data/graphrag_artifacts/` 目录。

如需重新构建知识图谱，删除该目录即可。

知识库文件存储在 `tea_data/RAG/` 目录中，支持以下格式：
- `.pdf` - PDF 文档
- `.txt` - 纯文本文件
- `.docx` - Word 文档

### GitHub 同步配置

GitHub 同步功能会**全量同步**项目文件，排除以下内容：
- `__pycache__/` - Python 缓存
- `.streamlit/` - Streamlit 配置（包含敏感信息）
- `.git/` - Git 仓库
- `.vscode/` - VSCode 配置
- 超过 100MB 的文件

同步时会在终端显示详细进度：
- `[OK]` - 同步成功
- `[FAIL]` - 同步失败
- `[SKIP]` - 跳过（文件过大）

---

## 🔧 开发指南

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 添加详细的文档字符串
- 保持模块职责单一

### 模块导入规范

```python
# 导入配置
from config.settings import PATHS, apply_page_config
from config.constants import FACTORS, TEA_EXAMPLES

# 导入核心逻辑
from core.resource_manager import ResourceManager
from core.github_sync import GithubSync
from core.ai_services import AliyunEmbedder

# 导入 UI 组件
from ui.sidebar import render_sidebar
from ui.tab1_interactive import render_tab1
```

### 添加新功能

1. **新增 UI 组件**: 在 `ui/` 目录下创建新文件，如 `tab7_newfeature.py`
2. **新增核心逻辑**: 在 `core/` 目录下创建新文件
3. **新增数据处理**: 在 `data/` 目录下创建新文件
4. **在 `main.py` 中注册新功能**

---

## 🎨 中国风设计

### 配色方案

系统采用中国茶文化传统配色：

- **黛绿** `#4A5D53` - 主色调
- **檀色** `#B36D61` - 中调（味）
- **紫檀** `#4A3728` - 后调（韵）
- **竹青** `#789262` - 辨识度
- **赭石** `#845538` - 饱和度
- **黛蓝** `#5B7C99` - 持久性

### 中文字体支持

可视化图表完美支持中文显示，使用以下字体优先级：
1. Noto Sans CJK SC
2. Noto Sans SC
3. SimHei（黑体）
4. Microsoft YaHei（微软雅黑）
5. WenQuanYi Micro Hei
6. Arial Unicode MS
7. DejaVu Sans

### UI 设计理念

- **东方美学**：简洁、优雅、含蓄
- **茶文化元素**：茶香、茶味、茶韵三段式设计
- **传统色彩**：使用中国传统色彩
- **对称平衡**：雷达图和风味图的对称设计

---

## 📊 系统架构图

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户界面层                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  main.py (入口)                                                          │
│      │                                                                   │
│      └──► ui/ (9 个组件)                                                 │
│            ├── sidebar.py              侧边栏 & API 配置（性能缓存）      │
│            ├── dialogs.py              弹窗组件                          │
│            ├── tab1_interactive.py     Tab1: 交互评分                    │
│            ├── tab2_batch.py           Tab2: 批量评分                    │
│            ├── tab3_knowledge.py       Tab3: 知识库管理                  │
│            ├── tab4_cases.py           Tab4: 判例库管理                  │
│            ├── tab5_finetune.py        Tab5: 模型微调                    │
│            └── tab6_prompts.py         Tab6: 提示词配置                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              业务逻辑层                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  core/ (6 个模块)                                                         │
│  ├── resource_manager.py      资源管理（文件读写、数据持久化）           │
│  ├── ai_services.py           AI 服务（嵌入、LLM 调用）                 │
│  ├── scoring.py               核心评分逻辑（向量索引自动重建）          │
│  ├── github_sync.py           GitHub 全量同步工具（递归、缓存）          │
│  └── init.py                  应用初始化（RAG 加载、判例初始化）         │
└─────────────────────────────────────────────────────────────────────────┘
                        │               │               │
                        ▼               ▼               ▼
        ┌───────────────────┐   ┌──────────────┐   ┌──────────────┐
        │   data/ (数据处理)  │   │storage/(存储) │   │retrieval/(检索)│
        ├───────────────────┤   ├──────────────┤   ├──────────────┤
        │excel_parser.py     │   │database.py   │   │graphrag_     │
        │basic_case_processor│   │vector_store  │   │  retriever.py│
        │supplementary_      │   │              │   │              │
        │  processor.py      │   │CSV + FAISS   │   │logic.py      │
        │finetune_processor  │   │              │   │(评分推理)    │
        └───────────────────┘   └──────────────┘   └──────────────┘
                        │               │               │
                        └───────────────┴───────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              工具层                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  utils/ (3 个模块)                                                         │
│  ├── visualization.py         可视化（中国风配色、中文字体）             │
│  └── helpers.py               辅助工具（解析、报告生成）                 │
│                                                                          │
│  config/ (3 个模块)                                                        │
│  ├── settings.py               页面配置、中国风 CSS、路径管理            │
│  └── constants.py              全局常量（六因子、提示词）                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 性能优化

### Session State 缓存

系统使用 session_state 缓存机制，避免重复网络请求：

1. **GitHub 状态缓存**
   - 首次检查：4-5 秒（可接受）
   - 后续操作：< 100ms（从缓存读取）
   - 手动刷新：可选操作
   - 自动刷新：同步完成后

2. **微调模型状态缓存**
   - 首次检查：2 秒
   - 有效期：5 分钟
   - 自动更新：超时后重新检查

3. **向量索引缓存**
   - 自动检测维度不匹配
   - 缓存重建的索引到 session_state
   - 避免重复编码（性能提升 10-20 倍）

### 缓存策略

```python
# GitHub 状态缓存
if 'github_status_cache' not in st.session_state:
    st.session_state.github_status_cache = {
        'configured': False,
        'msg': '',
        'repo_info': None,
        'last_check': None,
        'needs_refresh': True
    }

# 检查是否需要刷新
cache = st.session_state.github_status_cache
if cache['needs_refresh']:
    # 执行检查
    # 更新缓存
    cache['needs_refresh'] = False

# 向量索引缓存
if 'rebuilt_supp_idx' in st.session_state:
    # 使用缓存的索引，避免重复编码
    new_idx = st.session_state.rebuilt_supp_idx
```

---

## 🔬 关键技术点

| 技术点 | 实现位置 | 说明 |
|--------|---------|------|
| **中国风配色** | `utils/visualization.py` | 使用传统色彩（黛绿、檀色、紫檀等） |
| **中文字体支持** | `utils/visualization.py` | matplotlib 字体配置，支持中文显示，避免乱码 |
| **性能缓存** | `ui/sidebar.py` | session_state 缓存 GitHub 状态和模型状态，提升响应速度 |
| **Tree of Thoughts** | `tea_data/prompts.json` | 多路径评估决策模式，提升评分准确性 |
| **GraphRAG** | `retrieval/graphrag_retriever.py` | 知识图谱增强检索，社区发现算法 |
| **GitHub 全量同步** | `core/github_sync.py` | 递归遍历、排除模式、Windows 路径修复 |
| **向量检索** | `storage/vector_store.py` | FAISS 索引，快速相似度搜索 |
| **向量索引自动重建** | `core/scoring.py` | 自动检测维度不匹配，批量编码并���存索引 |
| **Excel 批量处理** | `data/*_processor.py` | 复用解析器，支持三种判例类型 |
| **Word 报告生成** | `utils/helpers.py` | 批量评分自动生成 Word 报告 |
| **Windows 路径修复** | `core/github_sync.py` | 自动将 `\` 转换为 `/`，适配 GitHub |
| **嵌套数据结构处理** | `utils/visualization.py` | 处理 AI 返回的嵌套 JSON 数据结构 |
| **智能建议生成** | `tea_data/prompts.json` | AI 基于专业知识生成个性化改进建议 |

---

## 🐛 已知问题

1. **根目录 `visualization.py`**：
   - 该文件为旧版本，缺少中文字体支持
   - 请使用 `utils/visualization.py` 替代
   - 建议删除根目录的 `visualization.py`

2. **Windows 路径分隔符**：
   - GitHub 同步已修复 Windows 路径问题
   - 自动将 `\` 转换为 `/`

3. **向量索引维度不匹配**：
   - 当嵌入器模型更换时，可能出现索引维度不匹配
   - 系统会自动重建索引并缓存，首次重建较慢（1-2分钟）
   - 后续使用缓存的索引，性能恢复正常

4. **建议生成重复度**：
   - AI 生成的建议可能存在一定重复度
   - 已在系统提示词中添加例外条款，允许 AI 在建议字段发挥专业能力
   - 建议内容基于每个因子的得分情况和专业知识

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证

---

## 👥 作者

国茶实验室

---

## 📮 联系方式

如有问题或建议，请提交 Issue 或 Pull Request

---

**🍵 意每一杯茶都得到公正的评价**
