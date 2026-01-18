import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch.nn.functional as F

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="VQA-RAD 医疗诊断系统",
    page_icon="🏥",
    layout="wide"
)

# 自定义 CSS 让界面更像医疗软件
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #0e76a8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .diagnosis-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0e76a8;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 智能医疗影像辅助诊断系统 (BLIP-XAI)</div>', unsafe_allow_html=True)


# --- 2. 加载模型 (核心部分) ---
@st.cache_resource
def load_model():
    """
    加载模型并缓存，避免每次刷新页面都重新下载
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用 Salesforce 的基础 BLIP 模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device


# 显示加载状态
with st.spinner('正在初始化医疗 AI 引擎 (加载 BLIP 模型)...'):
    processor, model, device = load_model()


# --- 3. 核心预测与解释函数 ---

def predict_answer(image, question):
    """
    获取 BLIP 的文本回答
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer


def lime_predict_proba(images, question, target_label_idx=None):
    """
    适配 LIME 的预测函数。
    LIME 需要输入 numpy 数组，输出概率。
    这里我们计算模型生成特定答案的概率。
    """
    # LIME 传入的是 numpy 数组列表，转为 PIL
    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]

    # 构造 batch 输入
    inputs = processor(images=pil_images, text=[question] * len(pil_images), return_tensors="pt", padding=True).to(
        device)

    # 获取 logits
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取词表中文本 logits
        logits = outputs.text_outputs.logits[:, 0, :]  # 取第一个 token
        probs = F.softmax(logits, dim=-1)

    # 如果没有指定目标 label，就取当前最大概率的 label 作为目标
    if target_label_idx is None:
        target_label_idx = torch.argmax(probs[0]).item()

    # LIME 需要返回 (batch_size, num_classes)，为了简化，我们只返回目标类的概率
    # 构造一个伪概率：[目标类概率, 1-目标类概率]
    target_probs = probs[:, target_label_idx].cpu().numpy()
    return np.stack([1 - target_probs, target_probs], axis=1)


# --- 4. 侧边栏：控制区 ---
st.sidebar.title("🩺 诊断控制台")

uploaded_file = st.sidebar.file_uploader("1. 上传影像 (X-Ray/CT)", type=["jpg", "png", "jpeg"])

# 预设问题，方便演示
question_options = [
    "Is there a fracture?",
    "Is the lung normal?",
    "What is the abnormality?",
    "Is the heart enlarged?",
    "自定义问题..."
]
selected_q = st.sidebar.selectbox("2. 选择诊断问题", question_options)
if selected_q == "自定义问题...":
    question = st.sidebar.text_input("请输入问题 (英文)", "Is there a fracture?")
else:
    question = selected_q

# LIME 设置 (用于平衡速度和质量)
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 XAI 设置")
num_samples = st.sidebar.slider("LIME 采样数 (越高质量越好但越慢)", 50, 500, 100)

# --- 5. 主界面逻辑 ---

if uploaded_file is not None:
    # 布局：左图右文
    col1, col2 = st.columns([1, 1.2])

    # 加载并显示原图
    raw_image = Image.open(uploaded_file).convert('RGB')

    with col1:
        st.subheader("原始影像")
        st.image(raw_image, use_column_width=True, caption="Uploaded Patient Scan")

    # 按钮触发分析
    if st.sidebar.button("开始 AI 诊断与分析", type="primary"):
        with col2:
            st.subheader("诊断报告")

            # 步骤 A: 预测
            with st.spinner('🤖 AI 正在阅片并生成诊断...'):
                diagnosis = predict_answer(raw_image, question)

            # 显示漂亮的诊断框
            st.markdown(f"""
            <div class="diagnosis-box">
                <h4><b>Q:</b> {question}</h4>
                <h3><b>A:</b> {diagnosis}</h3>
            </div>
            """, unsafe_allow_html=True)

            # 步骤 B: XAI 可视化 (LIME)
            st.subheader("可解释性分析 (LIME)")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("正在初始化 LIME 解释器...")
                explainer = lime_image.LimeImageExplainer()

                # 定义针对当前问题和图片的预测函数 wrapper
                # 我们需要找到 answer 对应的 token ID
                inputs = processor(raw_image, question, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                predicted_token_id = out[0][1]  # 取生成的第一个有效 token (通常是 [CLS] 后的第一个)

                # 包装函数
                predict_fn_lime = lambda x: lime_predict_proba(x, question, target_label_idx=predicted_token_id)

                status_text.text(f"正在生成扰动样本 (Samples: {num_samples})... 这可能需要一分钟")
                progress_bar.progress(30)

                # 核心 LIME 计算
                explanation = explainer.explain_instance(
                    np.array(raw_image),
                    predict_fn_lime,
                    top_labels=1,
                    hide_color=0,
                    num_samples=num_samples
                )
                progress_bar.progress(80)

                # 获取图像和掩膜
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False
                )

                # 显示 LIME 结果
                fig, ax = plt.subplots()
                img_boundary = mark_boundaries(temp / 255.0 + 0.5, mask)  # 稍微调亮一点
                ax.imshow(img_boundary)
                ax.axis('off')
                ax.set_title(f"LIME Visualization for '{diagnosis}'")

                st.pyplot(fig)
                progress_bar.progress(100)
                status_text.text("✅ 分析完成")

                st.info(
                    f"**图解说明：** 黄色/高亮边缘区域表示 AI 在判定 '{diagnosis}' 时重点关注的图像特征 (Superpixels)。")

                # 生成可下载报告
                report_content = f"""
                === VQA-RAD DIAGNOSTIC REPORT ===
                Image: {uploaded_file.name}
                Clinical Question: {question}
                AI Diagnosis: {diagnosis}
                XAI Method: LIME (Local Interpretable Model-agnostic Explanations)
                Confidence Areas: Identified in the attached visualization.
                =================================
                """
                st.download_button("📥 下载完整诊断报告", report_content, "diagnosis_report.txt")

            except Exception as e:
                st.error(f"XAI 生成过程中发生错误: {str(e)}")
                st.write("建议：尝试减少 LIME 采样数或检查显存。")

else:
    # 欢迎页状态
    st.info("👈 请在左侧侧边栏上传一张医学影像以开始演示。")

    # 演示用的伪代码展示 (可选)
    with st.expander("查看 Dashboard 原理 (代码片段)"):
        st.code("""
        # 核心逻辑
        diagnosis = model.generate(image, question)
        explanation = lime.explain_instance(image, predict_fn)
        st.pyplot(explanation.show())
        """, language='python')