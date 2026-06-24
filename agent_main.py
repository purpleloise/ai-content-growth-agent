import feedparser
from openai import OpenAI
import datetime
import os

# 1. 实例化客户端并配置鉴权信息
client = OpenAI(
    api_key="sk-uF237yjDtqmTOM7sSExnEGBPsBWdZAdwq77lkclnSxuCosTJ", 
    base_url="https://api.moonshot.cn/v1",
)

def fetch_tech_news():
    """解析 Hacker News 的 RSS 摘要源以获取原始文本数据"""
    rss_url = "https://news.ycombinator.com/rss"
    feed = feedparser.parse(rss_url)
    
    news_items = []
    for entry in feed.entries[:10]:  # 提取前10条数据以控制Token消耗
        news_items.append(f"Title: {entry.title}\nLink: {entry.link}")
    
    return "\n\n".join(news_items)

def generate_summary(raw_text):
    """调用 Kimi 模型 API 进行自然语言处理与结构化输出"""
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

def save_to_markdown(content):
    """将模型生成的结构化数据写入本地Markdown文件"""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"Tech_News_Summary_{date_str}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 科技资讯自动聚合简报 ({date_str})\n\n")
        f.write(content)
    
    print(f"执行完毕，文件已生成：{filename}")

if __name__ == "__main__":
    raw_news = fetch_tech_news()
    summary_output = generate_summary(raw_news)
    save_to_markdown(summary_output)