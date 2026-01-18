import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch.nn.functional as F
import io

# --- 1. Page Configuration (Medical Theme) ---
st.set_page_config(
    page_title="VQA-RAD Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0056b3;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    .diagnosis-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #0056b3;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0056b3;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #004494;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 VQA-RAD Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by BLIP Model & LIME Explainability</p>", unsafe_allow_html=True)

# --- 2. Load Model (Cached) ---
@st.cache_resource
def load_model():
    """
    Load BLIP model and cache it to improve performance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using Salesforce BLIP Base VQA model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device

# Show spinner while loading
with st.spinner('Initializing AI Engine (Loading Model weights)...'):
    processor, model, device = load_model()

# --- 3. Core Logic Functions (FIXED) ---

def predict_answer(image, question):
    """
    Standard Inference: Get text answer from BLIP
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def lime_predict_proba(images, question, target_label_idx):
    """
    FIXED LIME Prediction Function (Robust Version):
    Uses 'model.generate' with 'output_scores=True' to get probabilities.
    This bypasses the 'BlipTextVisionModelOutput' error completely.
    """
    # 1. Convert numpy array (LIME input) to PIL list
    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    batch_size = len(pil_images)
    
    # 2. Tokenize inputs
    inputs = processor(
        images=pil_images, 
        text=[question] * batch_size, 
        return_tensors="pt", 
        padding=True
    ).to(device)

    # 3. CRITICAL FIX: Use generate() to get scores directly
    # This works on ALL versions of transformers and avoids the "no attribute logits" error
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,             # We only need the first token probability
            output_scores=True,           # Request logits/scores
            return_dict_in_generate=True  # Return object with 'scores' attribute
        )
    
    # 4. Extract logits for the first generated token
    # outputs.scores is a tuple (one per step), we take the first step [0]
    # shape: (batch_size, vocab_size)
    logits = outputs.scores[0]
    probs = F.softmax(logits, dim=-1)
    
    # 5. Extract probability of the specific target token
    target_probs = probs[:, target_label_idx].cpu().numpy()
    
    # 6. Return format required by LIME: [Prob of NOT target, Prob of Target]
    return np.stack([1 - target_probs, target_probs], axis=1)

# --- 4. Sidebar UI ---
st.sidebar.title("🩺 Control Panel")

st.sidebar.subheader("1. Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload Medical Image (X-Ray/CT)", type=["jpg", "png", "jpeg"])

st.sidebar.subheader("2. Diagnostic Query")
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
    question = st.sidebar.text_input("Enter clinical question (English):", "Is there a fracture?")
else:
    question = selected_q

st.sidebar.markdown("---")
st.sidebar.subheader("3. XAI Parameters")
num_samples = st.sidebar.slider(
    "LIME Perturbation Samples", 
    min_value=30, 
    max_value=300, 
    value=60, 
    help="Higher values = Better heatmap quality but slower speed. Keep low (50-100) for live demos."
)

# --- 5. Main Layout ---

if uploaded_file is not None:
    # 2-Column Layout
    col1, col2 = st.columns([1, 1.2])
    
    # Load and Preprocess Image
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Patient Scan")
        st.image(raw_image, use_column_width=True, caption=f"Source: {uploaded_file.name}")

    # Start Analysis Button
    if st.sidebar.button("RUN DIAGNOSIS", type="primary"):
        with col2:
            st.subheader("Diagnostic Report")
            
            # --- Step A: Get Text Prediction ---
            with st.spinner('🤖 Analyzing image content...'):
                diagnosis = predict_answer(raw_image, question)
            
            # Display Prediction
            st.markdown(f"""
            <div class="diagnosis-box">
                <p><b>Question:</b> {question}</p>
                <h3><b>AI Finding:</b> {diagnosis.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Step B: Generate Explanation (LIME) ---
            st.subheader("Visual Explanation (LIME Heatmap)")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preparing explainability engine...")
                
                # 1. Get the Token ID for the predicted answer
                # We run generation again to extract the exact token ID
                inputs_check = processor(raw_image, question, return_tensors="pt").to(device)
                out_check = model.generate(**inputs_check)
                
                # BLIP typically outputs [BOS, token, ...] -> we want the token at index 1
                if len(out_check[0]) > 1:
                    predicted_token_id = out_check[0][1].item()
                else:
                    predicted_token_id = out_check[0][0].item() # Fallback

                # 2. Define the wrapped prediction function for LIME
                # This locks the question and target token, so LIME only varies the image
                predict_fn_lime = lambda x: lime_predict_proba(x, question, target_label_idx=predicted_token_id)

                # 3. Initialize Explainer
                explainer = lime_image.LimeImageExplainer()
                
                status_text.text(f"Generating {num_samples} perturbations... (This may take a moment)")
                progress_bar.progress(20)
                
                # 4. Run Explanation
                explanation = explainer.explain_instance(
                    np.array(raw_image), 
                    predict_fn_lime, 
                    top_labels=1, 
                    hide_color=0, 
                    num_samples=num_samples
                )
                progress_bar.progress(80)
                
                # 5. Extract Image and Mask
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], 
                    positive_only=True, 
                    num_features=5, 
                    hide_rest=False
                )
                
                # 6. Plotting
                fig, ax = plt.subplots()
                img_boundary = mark_boundaries(temp / 255.0 + 0.5, mask)
                ax.imshow(img_boundary)
                ax.axis('off')
                ax.set_title(f"Evidence for '{diagnosis}'")
                
                st.pyplot(fig)
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete")
                
                # Explanation text
                st.info(f"**Interpretation:** The highlighted areas indicate the visual features that convinced the AI model to predict '{diagnosis}'.")
                
                # Report Download
                report_content = f"""
                === VQA-RAD CLINICAL REPORT ===
                Image: {uploaded_file.name}
                Query: {question}
                AI Diagnosis: {diagnosis}
                Confidence Validation: LIME Heatmap Generated
                ===============================
                """
                st.download_button("📥 Download Report (.txt)", report_content, "clinical_report.txt")
                
            except Exception as e:
                st.error(f"XAI Error: {str(e)}")
                st.warning("Try reducing the 'Perturbation Samples' in the sidebar if memory is an issue.")

else:
    # Empty State - Instructional
    st.info("👈 Please upload an X-Ray or CT scan in the sidebar to begin analysis.")
    
    # Optional: Architecture Demo for Pre
    with st.expander("Show System Architecture (For Presentation)"):
        st.code("""
        # System Workflow
        1. Input: Image + Clinical Question
        2. Model: Salesforce BLIP (Bootstrapping Language-Image Pre-training)
        3. Interpretation: LIME (Local Interpretable Model-agnostic Explanations)
           -> Generates perturbations
           -> Calculates probability of target token
           -> Visualizes superpixels
        """, language='python')