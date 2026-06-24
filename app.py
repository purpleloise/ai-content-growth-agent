import streamlit as st
import feedparser
from openai import OpenAI
import datetime
import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件
api_key = os.getenv("MOONSHOT_API_KEY")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.moonshot.cn/v1",
)

def fetch_tech_news():
    """解析 RSS 摘要源以获取原始文本数据"""
    rss_url = "https://news.ycombinator.com/rss"
    feed = feedparser.parse(rss_url)
    
    news_items = []
    for entry in feed.entries[:10]:
        news_items.append(f"Title: {entry.title}\nLink: {entry.link}")
    return "\n\n".join(news_items)

def generate_summary(raw_text):
    """调用 Kimi 模型 API 生成结构化简报"""
    system_prompt = (
        "你是一个专业的技术资讯分析Agent。请分析以下Hacker News的最新动态，"
        "将其翻译为中文，提取出3个最核心的技术热点，并以Markdown格式输出一份简报。"
    )
    
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ----------------- GUI 架构渲染 ----------------- #

# 配置页面元数据
st.set_page_config(page_title="AI 资讯聚合 Agent", page_icon="🌐", layout="centered")

# 构建页面头部UI
st.title("🌐 AI 资讯自动化聚合器")
st.markdown("基于 Kimi 大模型实时抓取并解析 Hacker News 核心技术动态。")
st.divider() # 水平分割线

# 构建交互组件与状态控制
if st.button("🚀 立即生成今日技术简报", use_container_width=True):
    with st.spinner("正在连接数据源与调用大模型推理中，请稍候..."):
        try:
            # 1. 抓取数据
            raw_data = fetch_tech_news()
            
            # 2. 模型推理
            summary_markdown = generate_summary(raw_data)
            
            # 3. 结果渲染
            st.success(f"简报生成成功！时间戳：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 使用容器展示 Markdown 结果
            with st.container(border=True):
                st.markdown(summary_markdown)
                
        except Exception as e:
            st.error(f"运行过程中出现异常: {str(e)}")