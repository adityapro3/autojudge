import streamlit as st
import joblib
import re
import pandas as pd
from datetime import datetime

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

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.set_page_config(
    page_title="AutoJudge", 
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("⚖️ AutoJudge")
    st.markdown("---")
    
    # Add interactive features in sidebar
    st.subheader("📊 Quick Stats")
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        st.metric("Total Predictions", total_predictions)
        
        classes = [p['class'] for p in st.session_state.prediction_history]
        most_common = max(set(classes), key=classes.count) if classes else "N/A"
        st.metric("Most Common Class", most_common)
    else:
        st.info("No predictions yet!")
    
    st.markdown("---")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Main content
st.title("⚖️ AutoJudge: Problem Difficulty Predictor")
st.markdown("### Predict the **Difficulty** and **Rating** of codeforces problems")

# Info expander
with st.expander("ℹ️ How it works", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📝 Step 1**")
        st.markdown("Enter your problem details")
    with col2:
        st.markdown("**🤖 Step 2**")
        st.markdown("AI analyzes the text")
    with col3:
        st.markdown("**🎯 Step 3**")
        st.markdown("Get difficulty prediction")

st.markdown("---")

# Input Form with better layout
with st.form("prediction_form", clear_on_submit=False):
    st.subheader("📋 Problem Details")
    
    # Get values from session state if example was loaded
    default_title = st.session_state.get('example_title', '')
    default_desc = st.session_state.get('example_desc', '')
    default_input = st.session_state.get('example_input', '')
    default_output = st.session_state.get('example_output', '')
    
    title = st.text_input(
        "Problem Title *", 
        value=default_title,
        placeholder="e.g., Two Sum, Binary Search, etc.",
        help="Enter a concise title for the problem"
    )
    
    desc = st.text_area(
        "Problem Description *", 
        value=default_desc,
        height=150, 
        placeholder="Paste the main problem statement here...",
        help="Detailed description of what the problem asks"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        input_desc = st.text_area(
            "Input Description", 
            value=default_input,
            height=100, 
            placeholder="Constraints, format of input...",
            help="Description of input format and constraints"
        )
    
    with col2:
        output_desc = st.text_area(
            "Output Description", 
            value=default_output,
            height=100, 
            placeholder="Format of output...",
            help="Expected output format"
        )
    
    # Character counter
    total_chars = len(title) + len(desc) + len(input_desc) + len(output_desc)
    st.caption(f"Total characters: {total_chars}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button("🔍 Predict Difficulty")

# Clear example data after form submission
if submitted:
    for key in ['example_title', 'example_desc', 'example_input', 'example_output']:
        if key in st.session_state:
            del st.session_state[key]

if submitted:
    if not desc or not title:
        st.warning("⚠️ Please enter at least a Problem Title and Description.")
    else:
        with st.spinner("🔄 Analyzing problem difficulty..."):
            # Preprocess Input
            full_text = f"{title} {desc} {input_desc} {output_desc}"
            clean_text = re.sub(r'\s+', ' ', full_text.lower())
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            
            # Predict
            pred_class = class_model.predict([clean_text])[0]
            pred_score = score_model.predict([clean_text])[0]
            
            # Try to get confidence scores if available
            try:
                confidence = class_model.predict_proba([clean_text])[0]
                classes = class_model.classes_
                confidence_dict = dict(zip(classes, confidence))
            except:
                confidence_dict = None
            
            # Save to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'title': title,
                'class': pred_class,
                'score': int(round(pred_score, -2))
            })
            
            # Display Results
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            
            # Main metrics
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                color = "green" if pred_class == "Easy" else "orange" if pred_class == "Medium" else "red"
                st.markdown(f"### Difficulty: :{color}[{pred_class}]")
                
                # Visual indicator
                if pred_class == "Easy":
                    st.progress(0.33)
                elif pred_class == "Medium":
                    st.progress(0.66)
                else:
                    st.progress(1.0)
            
            with col2:
                st.metric(
                    label="Predicted Score", 
                    value=f"{int(round(int(pred_score), -2))}",
                )
            
            with col3:
                # Difficulty emoji
                emoji = "🟢" if pred_class == "Easy" else "🟡" if pred_class == "Medium" else "🔴"
                st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
            
            # Confidence scores
            if confidence_dict:
                st.subheader("📊 Confidence Breakdown")
                conf_df = pd.DataFrame({
                    'Difficulty': list(confidence_dict.keys()),
                    'Confidence': [f"{v*100:.1f}%" for v in confidence_dict.values()],
                    'Score': list(confidence_dict.values())
                })
                
                # Details in expander
                with st.expander("View confidence scores"):
                    for diff, conf in confidence_dict.items():
                        st.progress(conf, text=f"{diff}: {conf*100:.1f}%")
            
            # Interpretation
            st.info(f"💡 The model estimates this problem is **{pred_class}** with a rating around **{int(round(pred_score, -2))}**.")

# Show prediction history
if st.session_state.prediction_history:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display as interactive table
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download history
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name=f"autojudge_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | AutoJudge v1.0</div>",
    unsafe_allow_html=True
)
