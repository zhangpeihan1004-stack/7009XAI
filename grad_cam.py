import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForQuestionAnswering
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VQA-RAD Clinical Decision Support",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for Professional Medical UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        color: #0e76a8; /* Medical Blue */
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        background-color: #f0f7fa;
        border-left: 6px solid #0e76a8;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0e76a8;
        color: white;
        font-size: 18px;
        height: 50px;
        border-radius: 8px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #095c85;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Intelligent Medical VQA System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by BLIP Model & Grad-CAM Visual Attention</div>', unsafe_allow_html=True)

# --- 2. Load Model (Cached) ---
@st.cache_resource
def load_model():
    """
    Load BLIP model and cache it for performance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device

with st.spinner('System Initializing: Loading AI Models...'):
    processor, model, device = load_model()

# --- 3. Core Logic & Wrapper Class (CRITICAL FIX) ---

class BlipGradCAMWrapper(torch.nn.Module):
    """
    Robust Wrapper to fix 'missing pixel_values' AND 'no attribute logits' errors.
    """
    def __init__(self, model, input_ids, decoder_input_ids):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        
    def forward(self, pixel_values):
        # 1. 扩充维度以匹配 batch size
        b = pixel_values.shape[0]
        input_ids_expanded = self.input_ids.expand(b, -1)
        decoder_input_ids_expanded = self.decoder_input_ids.expand(b, -1)
        
        # 2. 调用模型
        outputs = self.model(
            input_ids=input_ids_expanded,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids_expanded
        )
        
        # 3. 强壮的 Logits 获取逻辑 (关键修复点)
        if hasattr(outputs, 'logits'):
            # 情况 A: 标准输出，直接有 logits
            return outputs.logits[:, 0, :]
            
        elif hasattr(outputs, 'text_outputs') and hasattr(outputs.text_outputs, 'logits'):
            # 情况 B: Logits 藏在 text_outputs 里 (常见于某些 HF 版本)
            return outputs.text_outputs.logits[:, 0, :]
            
        elif isinstance(outputs, tuple):
            # 情况 C: 输出是 Tuple (logits, loss, ...)
            # 通常第一个元素是 loss (如果提供了 labels) 或 logits
            # 这里的 outputs[0] 可能是 logits
            return outputs[0][:, 0, :]
            
        else:
            # 情况 D: 最后的尝试 - 打印结构报错 (调试用)
            raise AttributeError(f"Cannot find logits in model output type: {type(outputs)}")

def reshape_transform(tensor, height=24, width=24):
    """
    Reshape Vision Transformer (ViT) 1D patches back to 2D spatial grid.
    """
    result = tensor[:, 1:, :] # Skip CLS token
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def predict_answer(image, question):
    """
    Standard text inference.
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def generate_gradcam(image, question):
    """
    Generate Grad-CAM heatmap using the Wrapper.
    """
    # 1. Prepare Text Inputs
    text_inputs = processor(text=question, return_tensors="pt").to(device)
    input_ids = text_inputs.input_ids

    # 2. Identify Target Token (The answer we want to explain)
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    inputs_merge = {**image_inputs, **text_inputs}
    out_gen = model.generate(**inputs_merge)
    
    # Get the ID of the first generated word
    target_token_id = out_gen[0][1].item() if len(out_gen[0]) > 1 else out_gen[0][0].item()
    decoder_input_ids = torch.tensor([[target_token_id]]).to(device)

    # 3. Initialize Wrapper
    model_wrapper = BlipGradCAMWrapper(model, input_ids, decoder_input_ids)
    
    # 4. Define Target Layer (Last Layer of ViT Encoder)
    target_layer = model_wrapper.model.vision_model.encoder.layers[-1].layer_norm1
    
    # 5. Run Grad-CAM
    cam = GradCAM(
        model=model_wrapper, 
        target_layers=[target_layer], 
        reshape_transform=reshape_transform
    )
    
    input_tensor = image_inputs.pixel_values
    targets = [ClassifierOutputTarget(target_token_id)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # 6. Overlay on Image
    img_resized = image.resize((384, 384))
    img_np = np.array(img_resized) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    return visualization

# --- 4. Sidebar UI ---
st.sidebar.title("🩺 Diagnostic Console")

st.sidebar.subheader("1. Patient Imaging")
uploaded_file = st.sidebar.file_uploader("Upload Medical Image (X-Ray/CT)", type=["jpg", "png", "jpeg"])

st.sidebar.subheader("2. Clinical Query")
question_options = [
    "Is there a fracture?",
    "Is the lung normal?",
    "What is the abnormality?",
    "Is the heart enlarged?",
    "Is there pleural effusion?",
    "Custom Question..."
]
selected_q = st.sidebar.selectbox("Select Question", question_options)

if selected_q == "Custom Question...":
    question = st.sidebar.text_input("Enter Question (English):", "Is there a fracture?")
else:
    question = selected_q

st.sidebar.markdown("---")
st.sidebar.info("💡 **Grad-CAM Active**: Visualizing model attention based on your question.")

# --- 5. Main Layout ---

if uploaded_file is not None:
    # Two-column layout
    col1, col2 = st.columns([1, 1.2])
    
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Scan")
        st.image(raw_image, use_column_width=True, caption=f"Source: {uploaded_file.name}")

    # Analysis Button
    if st.sidebar.button("RUN DIAGNOSIS & EXPLAIN", type="primary"):
        with col2:
            st.subheader("Diagnostic Report")
            
            # Step A: Prediction
            with st.spinner('🤖 AI is analyzing clinical features...'):
                diagnosis = predict_answer(raw_image, question)
            
            # Display Result
            st.markdown(f"""
            <div class="diagnosis-box">
                <p style="margin-bottom:5px; color:#555;"><b>Clinical Question:</b> {question}</p>
                <h3 style="color:#0e76a8; margin-top:0;"><b>AI Finding:</b> {diagnosis.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Step B: Grad-CAM
            st.subheader("Attention Map (Grad-CAM)")
            
            with st.spinner('Generating Gradient-weighted Class Activation Map...'):
                try:
                    vis_img = generate_gradcam(raw_image, question)
                    
                    st.image(vis_img, use_column_width=True, caption="Model Visual Attention Heatmap")
                    st.success("✅ Explanation Generated Successfully")
                    
                    # Interpretation
                    st.info(f"""
                    **Interpretation:** The **red/orange regions** in the heatmap indicate where the model focused its attention to conclude **"{diagnosis}"**.
                    In a clinical context, this highlights the anatomical structures or pathologies relevant to the question.
                    """)
                    
                    # Download Report
                    report_text = f"""
                    === VQA-RAD CLINICAL REPORT ===
                    Date: 2024-05-20
                    Image ID: {uploaded_file.name}
                    Query: {question}
                    AI Finding: {diagnosis}
                    XAI Method: Grad-CAM (Gradient-weighted Class Activation Mapping)
                    Result: See attached heatmap.
                    ===============================
                    """
                    st.download_button("📥 Download Clinical Report", report_text, "clinical_report.txt")
                    
                except Exception as e:
                    st.error(f"Visualization Error: {str(e)}")
                    st.write("Debug info: Check BlipGradCAMWrapper logic.")

else:
    # Empty State
    st.info("👈 Please upload a medical image in the sidebar to begin analysis.")
    
    with st.expander("System Architecture (For Presentation)"):
        st.code("""
        # Pipeline Workflow
        1. Input: Medical Image + Clinical Text Question
        2. Model: BLIP (Bootstrapping Language-Image Pre-training)
        3. Feature Extraction: Vision Transformer (ViT)
        4. Explainability: Grad-CAM on last ViT Layer
           -> Computes gradients of the predicted answer w.r.t image features
           -> Reshapes 1D attention sequence to 2D heatmap
        """, language='python')

