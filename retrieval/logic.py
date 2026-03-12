import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import load_all_cases, load_json_kb, DATA_DIR, PATH_MANUAL
from storage.vector_store import load_vector_store, get_embedding, refresh_vector_index
import pandas as pd
import re
import numpy as np
import json
import dashscope
from dashscope import Generation


def retrieve_criteria(criteria_df):
    """加载评分标准 (核心依据)"""
    if criteria_df.empty:
        return "暂无评分标准，请依赖通用感官逻辑。"
    return criteria_df.to_markdown(index=False)

def retrieve_recent_history(history_df, top_k=3):
    """
    获取最近的打分记录作为参考
    注意：这里不再过滤茶类，只取最近的 k 条，用于让 AI 学习“打分手松还是手紧”。
    """
    if history_df.empty:
        return "暂无历史记录。"
    
    if 'expert_summary' not in history_df.columns:
        return "历史数据格式不完整。"
        
    # 只取有专家复核过的数据
    valid_cases = history_df[history_df['expert_summary'].notna()].tail(top_k)
    
    cases_str = ""
    for idx, (_, row) in enumerate(valid_cases.iterrows()):
        # 【关键修改】只展示“输入特征片段”和“分数”
        # 故意不展示完整的 expert_summary，防止 AI 偷懒抄袭文案
        cases_str += f"""
        [参考案例 {idx+1}]
        - 某历史样品的特征: "{str(row.get('input_review', ''))[:20]}..." (内容仅供区分)
        - 专家打分结果: 优雅性={row.get('分数_优雅性')}, 苦涩度={row.get('分数_苦涩度')}
        -------------------
        """
    return cases_str

# [替换整个 retrieve_expert_few_shot 函数]

def retrieve_expert_few_shot(target_review, top_k=3, threshold=0.75): # [修改1] 增加 threshold 参数默认值
    """
    [升级版] 基于语义相似度的 RAG 检索
    target_review: 当前用户输入的评价 (Text)
    """
    # 1. 获取当前输入的向量
    if not target_review: return "无输入内容。"
    
    target_emb = get_embedding(target_review)
    if target_emb is None:
        return "Embedding 服务暂不可用，无法进行语义检索(可能Key无效)。"
        
    # 2. 加载向量库
    vectors, meta_data = load_vector_store()
    if vectors is None or len(meta_data) == 0:
        return "知识库尚未初始化向量索引，暂无参考判例。"

    # 3. 计算相似度 (Cosine Similarity)
    # 归一化向量库
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / (norms + 1e-10)
    
    # 归一化目标向量
    target_norm = np.linalg.norm(target_emb)
    normalized_target = target_emb / (target_norm + 1e-10)
    
    # 点积计算相似度
    scores = np.dot(normalized_vectors, normalized_target)
    
    # 4. 排序并取 Top-K (argsort 是从小到大，所以要反转)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    # 5. 组装 Prompt
    prompt_text = ""
    # [修改2] 增加一个计数器，判断是否一条都没选中
    valid_count = 0

    for idx, i in enumerate(top_indices):
        # 边界检查
        if i >= len(meta_data): continue
        
        case = meta_data[i]
        score = scores[i]
        # [修改3] 核心：增加阈值拦截！如果不相似，直接跳过
        if score < threshold: 
            continue
        
        valid_count += 1
        prompt_text += f"""
        [参考判例 {valid_count} | 相似度: {score:.2f}]
        输入(Input): "{case.get('input_review')}"
        专家评语(Output): "{case.get('expert_summary')}"
        -------------------
        """
        
    # [修改4] 如果没有一条符合条件，给一个明确的提示
    if valid_count == 0: 
        return "未找到高相似度判例（均低于阈值），请严格依据通用标准评分，不要强行模仿无关案例。"
        
    return prompt_text

def extract_json_from_text(text):
    """JSON 提取器 (增强版：自动去除 Markdown 和修复常见格式错误)"""
    # 1. 尝试直接清洗 Markdown 标记
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except:
        # 2. 如果直接解析失败，尝试使用正则提取最外层的 {...} 结构
        try:
            # 这里的正则非贪婪匹配，寻找最外层的 {}
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
        except:
            pass
            
        # 3. 如果正则也挂了，尝试暴力截取（兜底方案）
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
                return json.loads(json_str)
        except:
            return None
    return None

def fetch_evaluation(comment, context, key, model_name="qwen-plus", threshold=0.75): # [修改1] 增加参数
    # --- [关键点 1] 校验并设置 API Key (Embedding 必须要有 Key 才能跑) ---
    if not key:
        return False, "Missing API Key", {}

    dashscope.api_key = key 
    
    # 初始化变量
    criteria_df = pd.DataFrame()
    history_df = pd.DataFrame()
    json_kb = []

    try:
        # 1. 加载数据 (保持原样)
        criteria_path = os.path.join(DATA_DIR, "criteria.csv")
        if not os.path.exists(criteria_path): criteria_path = "criteria.csv"
        
        if os.path.exists(criteria_path):
             criteria_df = pd.read_csv(criteria_path)

        history_df = load_all_cases()
        json_kb = load_json_kb()
        
        manual_content = "暂无手册内容"
        if os.path.exists(PATH_MANUAL):
            with open(PATH_MANUAL, 'r', encoding='utf-8') as f:
                manual_content = f.read()

        # --- [关键点 2] 刷新向量库 ---
        # 只有刷新了，新加入的案例才能被搜到。传入加载好的 history_df
        #try:
        #    refresh_vector_index(history_df)
        #except Exception as e:
        #    print(f"向量库刷新跳过或失败: {e}")
        # 相反，我们只打印一条日志，告诉自己正在使用旧索引
        print("正在使用现有向量索引进行检索...")
    
    except Exception as e:
        return False, f"数据加载失败: {str(e)}", {}

    # 2. 准备 Prompt 素材
    rag_criteria = retrieve_criteria(criteria_df)
    rag_history = retrieve_recent_history(history_df)
    
    # --- [关键点 3] 切换为语义检索 ---
    # [修改2] 调用时传入 threshold
    rag_expert_shots = retrieve_expert_few_shot(target_review=comment, top_k=3, threshold=threshold)
    
    # 3. 构建 Prompt (以下内容保持不变)
    prompt = f"""
    # Role
    你是由“国茶实验室”认证的顶级茶学风味师（Tea Flavorist）。你精通《中国茶感官品鉴手册》与《茶叶测评与鉴定手册》，熟练掌握“罗马测评法”（三段六因子九分制）。
    你的核心能力是将非专业的“用户口语”翻译为标准的“感官术语”，并基于严谨的扣分逻辑输出结构化评分。

    # Task
    阅读用户评价，利用【Reference】中的标准和判例，执行 Strict Reason-Act 流程，最后输出评分。

    # Input Data
    - 产品名称: "{context}" (尝试从中分析茶类)
    - 用户评价: "{comment}"
    - 饮用场景: "{context}"

    # Reference (RAG Context)
    
    1. 行业权威手册 (知识基座):
    {manual_content}

    2. 评分标准 (Codebook):
    {rag_criteria}

    3. 专家判例 (Calibration - 这里的案例是根据你的评价智能检索到的最相似案例):
    {rag_expert_shots}

    # ⚠️ Reason-Act 核心指令 (思维链)
    在打分前，你必须严格按照以下步骤在 `reasoning_chain` 中进行推演：
    (此处省略中间的 Prompt 文本，保持你原文件中的内容即可，或者如果你是直接复制粘贴，保留原文件的这部分内容)
    ...
    **Step 6: 证据加权与评分**
    - 仅仅提及（Mentioned）但无形容词 -> 中性分 (5.0)。
    - 提及且用强形容词（“极”、“非常”、“太”） -> 依据方向上下浮动 2-3 分。

    # Output Format (JSON Only)
    请严格输出以下 JSON 格式，不要包含Markdown代码块标记：
    {{
        "reasoning_chain": {{
            "category": "茶类识别结果及依据...",
            "parsing": "用户核心评价拆解...",
            "rag_retrieval": "已引用手册第X条规则；参考了判例[X]的打分尺度...",
            "defect_scan": "是否发现烟焦、酸馊等严重缺陷？...",
            "top_note": "提取前香描述 -> 映射专业术语 -> 评判结果...",
            "mid_note": "提取中味描述 -> 映射专业术语 -> 评判结果...",
            "base_note": "提取后韵描述 -> 映射专业术语 -> 评判结果..."
        }},
        "evaluation_summary": "基于感官科学的简短专业评语"，
        "scores": {{
            "优雅性": 0.0, "辨识度": 0.0, "协调性": 0.0, 
            "饱和度": 0.0, "持久性": 0.0, "苦涩度": 0.0
        }},
        "reasons": {{
            "优雅性": "...", 
            "辨识度": "...",
            "协调性": "...",
            "饱和度": "...",
            "持久性": "...",
            "苦涩度": "..."
        }}
    }}
    """
    

    # 4. 调用模型
    try:
        response = Generation.call(model=model_name, prompt=prompt, result_format='message')
        
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            data = extract_json_from_text(content)
            if data: return True, "Success", data
            else: return False, f"JSON解析失败: {content[:50]}...", {}
        else:
            return False, f"API Error: {response.message}", {}
    except Exception as e:
        return False, f"System Error: {str(e)}", {}