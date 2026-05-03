import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
from collections import defaultdict
import plotly.express as px
import torch
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="MAEA Tech - AI Vision Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .detection-badge {
        background: #0d6efd;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    @media (max-width: 768px) {
        .detection-badge { font-size: 0.7rem; }
        .main-header h1 { font-size: 1.5rem; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1 style="color: white;">MAEA Tech</h1>
        <h3 style="color: white;">AI Vision Assistant</h3>
        <p style="color: white;">Détection d'objets par IA • YOLOv8</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("Seuil IOU", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.markdown("### 👥 Équipe MAEA Tech")
    st.markdown("**Angèle EYENGA**")
    st.markdown("**Sajidat AKANNI**")
    st.markdown("---")
    st.markdown("© 2024 - Tous droits réservés")

# Chargement du modèle
@st.cache_resource
def load_model():
    with st.spinner("Chargement du modèle YOLO..."):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO("yolov8n.pt")
            model.to(device)
            return model
        except Exception as e:
            st.error(f"Erreur: {e}")
            return None

model = load_model()

if model is None:
    st.stop()

def load_image(file):
    image = Image.open(file)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    return np.array(image)

def process_frame(frame):
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)
    
    detections = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append((cls_name, conf))
    
    annotated_frame = results[0].plot()
    return annotated_frame, detections

# Interface principale
st.markdown("## 📸 Analyse d'images")

uploaded_files = st.file_uploader(
    "Choisissez des images à analyser", 
    type=["jpg", "png", "jpeg", "bmp"], 
    accept_multiple_files=True
)

if uploaded_files:
    for idx, file in enumerate(uploaded_files[:10]):  # Limite à 10 images
        st.markdown(f"### 📷 {file.name}")
        
        img = load_image(file)
        
        with st.spinner("Analyse en cours..."):
            result_img, detections = process_frame(img)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(result_img, use_container_width=True)
        
        with col2:
            if detections:
                st.markdown("**Objets détectés:**")
                for cls_name, conf in detections[:8]:
                    st.markdown(f'<span class="detection-badge">{cls_name} ({conf:.2f})</span>', 
                              unsafe_allow_html=True)
                if len(detections) > 8:
                    st.caption(f"+ {len(detections) - 8} autres")
            else:
                st.info("Aucune détection")
        
        st.markdown("---")
    
    # Résumé global
    st.markdown("## 📊 Résumé de l'analyse")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images analysées", len(uploaded_files))
    with col2:
        st.metric("Modèle", "YOLOv8n")
    with col3:
        st.metric("Seuil confiance", f"{confidence_threshold:.0%}")
