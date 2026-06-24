import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import httpx # 1. Import httpx

# ----------------- 1. 初始化与配置 -----------------
load_dotenv()
api_key = os.getenv("MOONSHOT_API_KEY")

# 2. Configure a custom httpx client that ignores proxies
custom_http_client = httpx.Client(trust_env=False)

# 3. Pass the custom client to the OpenAI initialization
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.moonshot.cn/v1",
    http_client=custom_http_client # Force bypass of system proxies
)

st.set_page_config(page_title="AI 内容增长实验舱", page_icon="📈", layout="wide")

# ----------------- 2. 数据层逻辑 -----------------
@st.cache_data
def load_mock_data():
    file_path = "xhs_mock_data.csv"
    if not os.path.exists(file_path):
        data = {
            "title": [
                "程序员搞钱必看！Cursor这个隐藏功能太牛了🔥",
                "别再手写代码了！Claude 3.5 自动化流全解析💻",
                "大学生速进！Kimi 帮我3分钟搞定文献综述🎓"
            ],
            "tags": ["#效率神器", "#AI工具", "#副业搞钱"],
            "ctr_benchmark": ["12.5%", "8.2%", "15.1%"],
            "collect_rate": ["5.1%", "3.0%", "8.5%"]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return df
    return pd.read_csv(file_path)

# ----------------- 3. 核心大模型逻辑 -----------------
def generate_xhs_content(raw_text, persona):
    prompt = f"""
    你是一个小红书千万级爆款操盘手。目标受众视角是：{persona}。
    请将以下技术资讯转化为小红书爆款文案。要求严格输出为 Markdown 格式：
    ### 方案 A（情绪痛点驱动）
    [生成标题A]
    [生成正文A：需要高密度 Emoji，短句分段]
    
    ### 方案 B（干货价值驱动）
    [生成标题B]
    [生成正文B：清晰列出步骤或价值点]
    
    ### 核心标签
    [生成 4-6 个高频 HashTag]

    资讯内容：{raw_text}
    """
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 
    )
    return response.choices[0].message.content

def predict_xhs_metrics(generated_content):
    prompt = f"""
    作为小红书推荐算法评估模型，请评估以下文案在小红书的潜在数据表现。
    必须严格输出 JSON 格式，不要包含任何其他说明文字或 Markdown 标记：
    {{"ctr": "预估点击率（如 10%-15%）", "collect_rate": "预估收藏率（如 3%-5%）", "rationale": "评估理由（不超过50字）"}}
    
    待评估文案：{generated_content}
    """
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 
    )
    raw_json = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
    return json.loads(raw_json)

# ----------------- 4. UI 界面构建 -----------------
st.title("📈 AI 内容增长实验舱 (小红书版)")
st.markdown("集成内容生成、A/B 测试方案输出及算法指标预测的自动化工作台。")
st.divider()

with st.sidebar:
    st.header("⚙️ 增长实验配置")
    target_persona = st.selectbox(
        "选择目标受众视角",
        ["独立开发者/程序员", "产品经理/运营", "大学生/学术研究", "副业搞钱/自媒体"]
    )
    input_text = st.text_area("输入原始技术资讯素材:", height=200, placeholder="例如：Kimi 刚刚发布了新的长文本能力...")
    start_btn = st.button("🚀 执行增长生成引擎", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📚 本地爆款特征库状态")
    df_mock = load_mock_data()
    st.dataframe(df_mock, use_container_width=True, hide_index=True)

if start_btn and input_text:
    if not api_key:
        st.error("系统检测到未配置 API Key，请检查 .env 文件。")
    else:
        with st.spinner("调用 Kimi 大模型进行内容重构与算法评估中..."):
            try:
                xhs_content = generate_xhs_content(input_text, target_persona)
                metrics_data = predict_xhs_metrics(xhs_content)
                
                st.success("增长方案生成完成！")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.container(border=True):
                        st.subheader("📝 A/B 测试文案库")
                        st.markdown(xhs_content)
                        
                with col2:
                    with st.container(border=True):
                        st.subheader("📊 算法指标预测")
                        st.metric(label="预估点击率 (CTR)", value=metrics_data.get("ctr", "N/A"))
                        st.metric(label="预估收藏率", value=metrics_data.get("collect_rate", "N/A"))
                        st.info(f"**模型评估逻辑：** {metrics_data.get('rationale', '无分析')}")

            except Exception as e:
                import traceback
                st.error("处理过程中出现详细异常：")
                st.code(traceback.format_exc()) # 将完整报错追踪打印在页面上