import os
import sys
import builtins

# Block cv2 completely before anything else
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Prevent any cv2 import
original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == 'cv2' or name.startswith('cv2.'):
        # Return a fake module
        import types
        fake_module = types.ModuleType('cv2')
        sys.modules['cv2'] = fake_module
        return fake_module
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

# Now import ultralytics (which tries to import cv2)
from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image
import tempfile

# Restore original import
builtins.__import__ = original_import

st.set_page_config(page_title="Helmet Safety Detection", layout="wide")
st.title("🪖 Real-time Construction Site Safety Monitor")
st.markdown("Upload an image to detect helmets and people")

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("✅ Model loaded!")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("🔍 Detecting helmets..."):
            results = model(tmp_path)
        
        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Original Image", use_container_width=True)
        col2.image(results[0].plot(), caption="Detection Result", use_container_width=True)
        
        # Show statistics
        if results[0].boxes is not None:
            boxes = results[0].boxes
            class_names = results[0].names
            person_count = 0
            helmet_count = 0
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id].lower()
                if class_name == 'person':
                    person_count += 1
                elif 'helmet' in class_name:
                    helmet_count += 1
            st.success(f"📊 Results: {person_count} persons | {helmet_count} helmets")
        
        os.unlink(tmp_path)
else:
    st.error("Failed to load model")