import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="SafetyVision AI", page_icon="🪖")

st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
st.success("✅ Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    results = model(img_array)
    annotated_img = results[0].plot()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original")
    with col2:
        st.image(annotated_img, caption="Detection")
    
    boxes = results[0].boxes
    if boxes is not None:
        helmet_count = sum(1 for box in boxes if int(box.cls[0]) == 0)
        person_count = sum(1 for box in boxes if int(box.cls[0]) == 1)
        
        st.markdown("### 📊 Detection Summary")
        st.success(f"Helmets: {helmet_count}")
        st.warning(f"Persons: {person_count}")
        
        if helmet_count < person_count:
            st.error("WARNING: Workers without helmets detected!")
        else:
            st.success("All safe!")