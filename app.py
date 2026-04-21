# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #00ffe0;
    animation: glow 2s infinite alternate;
}

/* Glow Animation */
@keyframes glow {
    from {text-shadow: 0 0 10px #00ffe0;}
    to {text-shadow: 0 0 25px #00ffe0;}
}

/* Card Style */
.card {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg,#00ffe0,#00c3ff);
    color: black;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    height: 3em;
    width: 100%;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00ffe0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🚀 AI Prediction App</p>', unsafe_allow_html=True)
st.markdown("### 🔥 Deploy Your Machine Learning Model Beautifully")

# ---------------- MAIN SECTION ----------------
col1, col2 = st.columns([1,1])

# -------- INPUT SIDE --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Enter Input Values")

    # Change number of features according to your model
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)

    predict_btn = st.button("🔮 Predict Now")
    st.markdown('</div>', unsafe_allow_html=True)

# -------- OUTPUT SIDE --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_btn:
        if model is not None:
            try:
                input_data = np.array([[f1, f2, f3, f4]])
                prediction = model.predict(input_data)

                st.success(f"✅ Prediction: {prediction[0]}")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
        else:
            st.warning("⚠️ Model not loaded.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>✨ Built with ❤️ using Streamlit | Animated UI</center>",
    unsafe_allow_html=True
)
