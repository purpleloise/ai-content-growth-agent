import os
import json
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ----------------- 初始化与配置 -----------------
load_dotenv()
api_key = os.getenv("MOONSHOT_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

st.set_page_config(page_title="开发者社区脉搏监测器", page_icon="📡", layout="wide")

# ----------------- 数据获取层 -----------------
def fetch_github_issues(repo, limit=10):
    """调用 GitHub REST API 获取指定开源仓库的实时 Issue 数据"""
    url = f"https://api.github.com/search/issues?q=repo:{repo}+is:issue+state:open&sort=created&order=desc&per_page={limit}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        items = response.json().get("items", [])
        issues = []
        for item in items:
            issues.append({
                "id": item["number"],
                "title": item["title"],
                "body": str(item["body"])[:500] # 截断处理文本以控制大模型输入 Token 消耗
            })
        return issues
    else:
        st.error(f"API 请求执行失败，返回状态码: {response.status_code}")
        return []

# ----------------- 模型处理层 -----------------
def analyze_issue(issue_text):
    """构建 Prompt 调用 Kimi API 执行自然语言理解任务（情感分析与主题提取）"""
    prompt = f"""
    请对以下开发者社区的反馈文本进行结构化数据提取。
    必须严格输出 JSON 格式，不包含任何其他说明文字或 Markdown 标记：
    {{"sentiment": "情绪分类（Positive/Negative/Neutral/Feature Request）", "topic": "核心技术主题（如：Bug, Documentation, Installation, Performance）", "summary": "一句话核心诉求总结"}}
    
    待评估文本：{issue_text}
    """
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # 采用低随机性配置以保证结构化数据输出的稳定性
        )
        raw_json = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw_json)
    except Exception as e:
        return {"sentiment": "Error", "topic": "Error", "summary": str(e)}

# ----------------- 展示层与运行逻辑 -----------------
st.title("📡 自动化开发者社区脉搏监测器")
st.markdown("集成公网 API 抓取与零样本学习（Zero-shot Learning）的开发者反馈自动化分析平台。")
st.divider()

# 侧边栏交互组件
with st.sidebar:
    st.header("系统参数配置")
    target_repo = st.text_input("目标 GitHub 仓库 (Format: owner/repo)", value="langchain-ai/langchain")
    fetch_limit = st.slider("单次获取样本容量", min_value=5, max_value=20, value=5)
    start_btn = st.button("执行数据采集与处理流", type="primary", use_container_width=True)

# 主程序执行流
if start_btn:
    if not api_key:
        st.error("环境变量异常：未检测到有效 API Key。")
    else:
        with st.spinner(f"正在建立网络连接并执行底层 NLP 分析流..."):
            issues = fetch_github_issues(target_repo, fetch_limit)
            
            if issues:
                results = []
                progress_bar = st.progress(0)
                
                # 遍历数据集进行批处理计算
                for i, issue in enumerate(issues):
                    analysis = analyze_issue(f"标题：{issue['title']} \n 内容：{issue['body']}")
                    results.append({
                        "Issue 编号": issue["id"],
                        "原始标题": issue["title"],
                        "情感特征": analysis.get("sentiment"),
                        "聚合主题": analysis.get("topic"),
                        "算法摘要": analysis.get("summary")
                    })
                    progress_bar.progress((i + 1) / len(issues))
                
                # 数据重组与格式化
                df = pd.DataFrame(results)
                st.success("数据处理流执行完毕。")
                
                # 渲染数据看板
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("全局情感特征分布")
                    sentiment_counts = df["情感特征"].value_counts()
                    st.bar_chart(sentiment_counts)
                
                with col2:
                    st.subheader("高频技术讨论主题")
                    topic_counts = df["聚合主题"].value_counts()
                    st.bar_chart(topic_counts)
                
                st.subheader("底层数据结构明细")
                st.dataframe(df, use_container_width=True)