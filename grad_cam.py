import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForQuestionAnswering
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(
    page_title="VQA-RAD Clinical Support System",
    page_icon="🩺",
    layout="wide"
)

# Professional Medical CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        color: #0e76a8;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
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
    }
    .stButton>button {
        background-color: #0e76a8;
        color: white;
        font-size: 16px;
        height: 50px;
        border-radius: 8px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #085a82;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Intelligent Medical VQA System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable AI (XAI) with Grad-CAM Visualization</div>', unsafe_allow_html=True)

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    """
    Load BLIP model and cache it to speed up the app.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device

with st.spinner('System Initializing: Loading AI Models...'):
    processor, model, device = load_model()

# --- 3. Robust Wrapper Class (The Fix) ---

class BlipGradCAMWrapper(torch.nn.Module):
    """
    A robust wrapper to handle BLIP's variable output structure 
    and ensure Grad-CAM receives the correct logits.
    """
    def __init__(self, model, input_ids, decoder_input_ids):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        
    def forward(self, pixel_values):
        # Expand dimensions to match batch size
        b = pixel_values.shape[0]
        input_ids_expanded = self.input_ids.expand(b, -1)
        decoder_input_ids_expanded = self.decoder_input_ids.expand(b, -1)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids_expanded,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids_expanded
        )
        
        # --- ROBUST LOGIT EXTRACTION ---
        # 1. Check for 'logits' (Standard)
        if hasattr(outputs, 'logits'):
            return outputs.logits[:, 0, :]
            
        # 2. Check for 'text_model_output' (Specific fix for your error)
        elif hasattr(outputs, 'text_model_output') and hasattr(outputs.text_model_output, 'logits'):
            return outputs.text_model_output.logits[:, 0, :]
            
        # 3. Check for 'text_outputs' (Older versions)
        elif hasattr(outputs, 'text_outputs') and hasattr(outputs.text_outputs, 'logits'):
            return outputs.text_outputs.logits[:, 0, :]
            
        else:
            # Panic mode: print attributes to help debug
            raise AttributeError(f"Cannot find logits. Available keys: {dir(outputs)}")

def reshape_transform(tensor, height=24, width=24):
    """
    Reshape Vision Transformer (ViT) patches back to 2D images.
    """
    result = tensor[:, 1:, :] # Skip CLS token
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# --- 4. Core Logic Functions ---

def predict_answer(image, question):
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def generate_gradcam(image, question):
    # 1. Prepare Text Inputs
    text_inputs = processor(text=question, return_tensors="pt").to(device)
    input_ids = text_inputs.input_ids

    # 2. Predict Target Token (What answer are we explaining?)
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    inputs_merge = {**image_inputs, **text_inputs}
    out_gen = model.generate(**inputs_merge)
    target_token_id = out_gen[0][1].item() if len(out_gen[0]) > 1 else out_gen[0][0].item()
    decoder_input_ids = torch.tensor([[target_token_id]]).to(device)

    # 3. Initialize Wrapper
    model_wrapper = BlipGradCAMWrapper(model, input_ids, decoder_input_ids)
    
    # 4. Target Layer (ViT Encoder Last Layer)
    target_layer = model_wrapper.model.vision_model.encoder.layers[-1].layer_norm1
    
    # 5. Run Grad-CAM
    cam = GradCAM(
        model=model_wrapper, 
        target_layers=[target_layer], 
        reshape_transform=reshape_transform
    )
    
    input_tensor = image_inputs.pixel_values
    targets = [ClassifierOutputTarget(target_token_id)]
    
    # Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # 6. Overlay on Image
    img_resized = image.resize((384, 384))
    img_np = np.array(img_resized) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    return visualization

# --- 5. Sidebar Controls ---
st.sidebar.title("🩺 Control Panel")

st.sidebar.subheader("1. Patient Imaging")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray / CT Scan", type=["jpg", "png", "jpeg"])

st.sidebar.subheader("2. Clinical Question")
question_options = [
    "Is there a fracture?",
    "Is the lung normal?",
    "What is the abnormality?",
    "Is the heart enlarged?",
    "Is there pleural effusion?",
    "Custom Question..."
]
selected_q = st.sidebar.selectbox("Select Query", question_options)

if selected_q == "Custom Question...":
    question = st.sidebar.text_input("Enter Question (English):", "Is there a fracture?")
else:
    question = selected_q

st.sidebar.markdown("---")
st.sidebar.info("💡 **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)")

# --- 6. Main Layout ---

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])
    
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Patient Scan")
        st.image(raw_image, use_column_width=True, caption="Original Image")

    # Run Button
    if st.sidebar.button("RUN ANALYSIS", type="primary"):
        with col2:
            st.subheader("Analysis Report")
            
            # Text Prediction
            with st.spinner('🤖 Analyzing features...'):
                diagnosis = predict_answer(raw_image, question)
            
            # Display Diagnosis
            st.markdown(f"""
            <div class="diagnosis-box">
                <p style="margin-bottom:5px; color:#555;"><b>Question:</b> {question}</p>
                <h3 style="color:#0e76a8; margin-top:0;"><b>AI Finding:</b> {diagnosis.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual Explanation
            st.subheader("Visual Evidence (Grad-CAM)")
            
            with st.spinner('Generating Attention Heatmap...'):
                try:
                    vis_img = generate_gradcam(raw_image, question)
                    
                    st.image(vis_img, use_column_width=True, caption="Model Attention Heatmap")
                    st.success("✅ Explanation Generated")
                    
                    st.info(f"""
                    **Interpretation:** The **red/orange areas** show where the AI model looked to answer "{diagnosis}". 
                    This confirms the model is focusing on relevant anatomical structures.
                    """)
                    
                except Exception as e:
                    st.error(f"Visualization Error: {e}")
                    st.caption("Please check the logs for details.")

else:
    st.info("👈 Please upload a medical image to start the diagnosis.")
    st.markdown("---")
    with st.expander("Show System Architecture"):
        st.code("""
        System Pipeline:
        [Image] + [Text] -> BLIP Model -> Vision Transformer (ViT)
                                        |
        (Backpropagation) <------------ Output Token
                                        |
        [Grad-CAM Heatmap] <----------- Gradients
        """)