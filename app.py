import streamlit as st
import joblib
import re

# Load Models
@st.cache_resource
def load_models():
    clf = joblib.load('model_class.pkl')
    reg = joblib.load('model_score.pkl')
    return clf, reg

try:
    class_model, score_model = load_models()
except FileNotFoundError:
    st.error("Models not found! Please run 'train_model.py' first.")
    st.stop()

# UI Layout
st.set_page_config(page_title="AutoJudge", page_icon="⚖️")
st.title("⚖️ AutoJudge: Problem Difficulty Predictor")
st.markdown("Predict the **Difficulty Class** and **Numerical Score** of a coding problem based on its description.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Problem Title", placeholder="Input the heading of the problem")
    
    desc = st.text_area("Problem Description", height=150, placeholder="Paste the main problem statement here...")
    input_desc = st.text_area("Input Description", height=100, placeholder="Constraints, format of input...")
    output_desc = st.text_area("Output Description", height=100, placeholder="Format of output...")
    
    submitted = st.form_submit_button("Predict Difficulty")

if submitted:
    if not desc:
        st.warning("Please enter at least a Problem Description.")
    else:
        # Preprocess Input
        full_text = f"{title} {desc} {input_desc} {output_desc}"
        
        # Simple cleaning (must match training cleaning)
        clean_text = re.sub(r'\s+', ' ', full_text.lower())
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        
        # Predict
        pred_class = class_model.predict([clean_text])[0]
        pred_score = score_model.predict([clean_text])[0]
        
        # Display Results
        st.divider()
        st.subheader("Prediction Results")
        
        c1, c2 = st.columns(2)
        with c1:
            color = "green" if pred_class == "Easy" else "orange" if pred_class == "Medium" else "red"
            st.markdown(f"### Class: :{color}[{pred_class}]")
        
        with c2:
            st.metric(label="Predicted Score", value=f"{int(round(pred_score,2))}")
            
        st.info(f"The model estimates this problem is **{pred_class}** with a rating around **{int(round(pred_score,2))}**.")
