import streamlit as st

st.set_page_config(page_title="Helmet Safety Detection", layout="wide")

st.title("🪖 Helmet Safety Detection System")
st.write("App is running successfully!")

# Check if model exists
import os
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
if os.path.exists(model_path):
    st.success(f"✅ Model found at: {model_path}")
    st.info(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
else:
    st.error(f"❌ Model NOT found at: {model_path}")
    st.info("Make sure 'best.pt' is in your GitHub repository")

# Upload test
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Detection will work once model is properly loaded")