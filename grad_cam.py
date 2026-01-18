import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForQuestionAnswering
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VQA-RAD Clinical Decision Support (Grad-CAM)",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for Professional Medical UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        color: #2c3e50;
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
        background-color: #ecf0f1;
        border-left: 6px solid #3498db;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-size: 18px;
        height: 50px;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3498db;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Intelligent Medical VQA System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by BLIP Model & Grad-CAM Visual Attention</div>', unsafe_allow_html=True)

# --- 2. Load Model ---
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

# --- 3. Core Functions: Prediction & Grad-CAM ---

def predict_answer(image, question):
    """
    Generate text diagnosis.
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def reshape_transform(tensor, height=24, width=24):
    """
    Essential for Vision Transformers (ViT).
    Reshapes the 1D sequence of patches back into a 2D image grid.
    """
    result = tensor[:, 1:, :] # Skip CLS token
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # Transpose to (Batch, Channel, Height, Width)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam(image, question):
    """
    Generate the Grad-CAM heatmap overlay.
    """
    # 1. Prepare Input
    inputs = processor(image, question, return_tensors="pt").to(device)
    
    # 2. Define Target Layer (Last Layer of ViT Encoder)
    target_layer = model.vision_model.encoder.layers[-1].layer_norm1
    
    # 3. Initialize GradCAM
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer], 
        reshape_transform=reshape_transform
    )
    
    # 4. Generate Heatmap (Targeting global attention)
    # Using eigen_smooth for cleaner visuals
    grayscale_cam = cam(input_tensor=inputs.pixel_values, targets=None, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # 5. Overlay on Image
    # Resize original image to 384x384 (BLIP input size) for perfect alignment
    img_resized = image.resize((384, 384))
    img_np = np.array(img_resized) / 255.0
    
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# --- 4. Sidebar UI ---
st.sidebar.title("🩺 Diagnostic Console")

st.sidebar.subheader("1. Patient Imaging")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray/CT Scan", type=["jpg", "png", "jpeg"])

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
    question = st.sidebar.text_input("Enter your question (English):", "Is there a fracture?")
else:
    question = selected_q

st.sidebar.markdown("---")
st.sidebar.info("💡 **Grad-CAM Mode Active**\nVisualizes the model's attention gradients to explain the diagnosis.")

# --- 5. Main Layout ---

if uploaded_file is not None:
    # Two-column layout
    col1, col2 = st.columns([1, 1.2])
    
    # Load Image
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Scan")
        st.image(raw_image, use_column_width=True, caption=f"Source: {uploaded_file.name}")

    # Analysis Button
    if st.sidebar.button("RUN DIAGNOSIS & EXPLAIN", type="primary"):
        with col2:
            st.subheader("Diagnostic Report")
            
            # A. Text Prediction
            with st.spinner('🤖 AI is analyzing clinical features...'):
                diagnosis = predict_answer(raw_image, question)
            
            # Display Diagnosis
            st.markdown(f"""
            <div class="diagnosis-box">
                <p style="font-size:16px; margin-bottom:5px;"><b>Clinical Question:</b> {question}</p>
                <h3 style="color:#2c3e50; margin-top:0;"><b>AI Finding:</b> {diagnosis.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # B. Grad-CAM Visualization
            st.subheader("Attention Map (Grad-CAM)")
            
            with st.spinner('Generating Gradient-weighted Class Activation Map...'):
                try:
                    # Run Grad-CAM
                    vis_img = generate_gradcam(raw_image, question)
                    
                    # Display Result
                    st.image(vis_img, use_column_width=True, caption="Model Attention Heatmap")
                    st.success("✅ Explanation Generated Successfully")
                    
                    # Interpretation Text
                    st.info(f"""
                    **Interpretation:** The **red/orange** regions in the heatmap indicate where the model focused its attention to conclude **"{diagnosis}"**. 
                    In a clinical setting, this should correspond to the pathological area (e.g., fracture line, nodule, or heart border).
                    """)
                    
                    # Download Report
                    report_text = f"""
                    === VQA-RAD DIAGNOSTIC REPORT ===
                    Image ID: {uploaded_file.name}
                    Query: {question}
                    AI Diagnosis: {diagnosis}
                    XAI Method: Grad-CAM (Gradient-weighted Class Activation Mapping)
                    Result: See attached attention map.
                    =================================
                    """
                    st.download_button("📥 Download Clinical Report", report_text, "report_gradcam.txt")
                    
                except Exception as e:
                    st.error(f"Visualization Error: {str(e)}")
                    st.warning("Please ensure 'grad-cam' is installed in requirements.txt")

else:
    # Empty State
    st.info("👈 Please upload a medical image in the sidebar to start the analysis.")
    
    with st.expander("System Architecture (For Presentation)"):
        st.code("""
        # Grad-CAM Workflow
        1. Forward Pass: Image -> ViT Encoder -> Text Decoder
        2. Backward Pass: Calculate Gradients for predicted token
        3. Visualization: Weighted sum of last encoder layer maps
        """, language='python')