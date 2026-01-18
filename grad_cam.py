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
    Wrapper to fix the 'missing pixel_values' error in Grad-CAM.
    It binds the text inputs (input_ids) to the model so Grad-CAM only sees image inputs.
    """
    def __init__(self, model, input_ids, decoder_input_ids):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        
    def forward(self, pixel_values):
        # Expand inputs to match the batch size (Grad-CAM might create batches)
        b = pixel_values.shape[0]
        input_ids_expanded = self.input_ids.expand(b, -1)
        decoder_input_ids_expanded = self.decoder_input_ids.expand(b, -1)
        
        # Explicitly pass all arguments to the BLIP model
        outputs = self.model(
            input_ids=input_ids_expanded,
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids_expanded
        )
        # Return logits for the first generated token
        return outputs.logits[:, 0, :]

def reshape_transform(tensor, height=24, width=24):
    """
    Reshape Vision Transformer (ViT) 1D patches back to 2D spatial grid.
    """
    result = tensor[:, 1:, :] # Skip CLS token
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def predict_answer
