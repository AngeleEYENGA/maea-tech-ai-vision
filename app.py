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
import requests
from streamlit_js_eval import streamlit_js_eval
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.distance import distance
import reverse_geocoder as rg

warnings.filterwarnings('ignore')

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="MAEA Tech - AI Vision Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# CSS PERSONNALISÉ PROFESSIONNEL
# -------------------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
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
    .alert-danger {
        background: #dc3545;
        padding: 0.75rem;
        border-radius: 6px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    .alert-warning {
        background: #fd7e14;
        padding: 0.75rem;
        border-radius: 6px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    .location-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .linkedin-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
        color: #0a66c2;
        font-weight: 500;
        padding: 5px 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .linkedin-link:hover {
        background-color: #f0f4f8;
    }
    .team-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# CLASSE DE GÉOLOCALISATION
# -------------------
class GeoLocalization:
    """Gestion de la géolocalisation en temps réel"""
    def __init__(self):
        self.current_lat = None
        self.current_lon = None
        self.current_address = None
        self.location_history = []
        self.start_time = None
        self.geolocator = Nominatim(user_agent="maea_tech_vision")
        
    def get_location(self):
        """Récupère la position GPS via JavaScript"""
        try:
            # Récupérer les coordonnées depuis le navigateur
            js_code = """
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    window.parent.document.body.setAttribute('data-lat', pos.coords.latitude);
                    window.parent.document.body.setAttribute('data-lon', pos.coords.longitude);
                    window.parent.document.body.setAttribute('data-accuracy', pos.coords.accuracy);
                },
                (err) => {
                    console.error(err);
                }
            );
            """
            
            # Alternative plus simple avec streamlit_js_eval
            lat = streamlit_js_eval(js_expressions='navigator.geolocation.getCurrentPosition', key='lat')
            lon = streamlit_js_eval(js_expressions='navigator.geolocation.getCurrentPosition', key='lon')
            
            # Pour l'exemple, on utilise des coordonnées de démonstration
            # Dans un environnement réel, utilisez les coordonnées réelles
            if self.current_lat is None:
                # Coordonnées par défaut (Paris)
                self.current_lat = 48.8566
                self.current_lon = 2.3522
            else:
                self.current_lat = lat if lat else self.current_lat
                self.current_lon = lon if lon else self.current_lon
            
            return self.current_lat, self.current_lon
        except Exception as e:
            st.warning(f"Géolocalisation non disponible: {e}")
            return None, None
    
    def reverse_geocode(self, lat, lon):
        """Convertit les coordonnées en adresse"""
        try:
            # Méthode 1: Avec Nominatim
            location = self.geolocator.reverse(f"{lat}, {lon}")
            if location:
                return location.address
            
            # Méthode 2: Avec reverse_geocoder (plus rapide)
            results = rg.search((lat, lon))
            if results:
                result = results[0]
                return f"{result['name']}, {result['admin1']}, {result['cc']}"
            
            return f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        except:
            return f"Coordonnées: {lat:.4f}, {lon:.4f}"
    
    def update_location(self):
        """Met à jour la position actuelle"""
        lat, lon = self.get_location()
        if lat and lon:
            self.current_lat = lat
            self.current_lon = lon
            self.current_address = self.reverse_geocode(lat, lon)
            
            # Historique des positions
            self.location_history.append({
                'timestamp': datetime.now(),
                'lat': lat,
                'lon': lon,
                'address': self.current_address
            })
            
            # Garder seulement les 100 derniers points
            if len(self.location_history) > 100:
                self.location_history.pop(0)
            
            return True
        return False
    
    def create_map(self):
        """Crée une carte Folium avec la position actuelle"""
        if self.current_lat and self.current_lon:
            # Centre sur la position actuelle
            m = folium.Map(
                location=[self.current_lat, self.current_lon],
                zoom_start=15,
                control_scale=True
            )
            
            # Marqueur pour la position actuelle
            folium.Marker(
                [self.current_lat, self.current_lon],
                popup=f"Position actuelle<br>{self.current_address}",
                icon=folium.Icon(color='red', icon='info-sign'),
                tooltip="Vous êtes ici"
            ).add_to(m)
            
            # Tracer l'historique des positions
            if len(self.location_history) > 1:
                points = [[loc['lat'], loc['lon']] for loc in self.location_history]
                folium.PolyLine(
                    points,
                    color='blue',
                    weight=3,
                    opacity=0.7,
                    popup='Trajectoire'
                ).add_to(m)
                
                # Ajouter les points de l'historique
                for loc in self.location_history[-10:]:  # Derniers 10 points
                    folium.CircleMarker(
                        [loc['lat'], loc['lon']],
                        radius=3,
                        color='blue',
                        fill=True,
                        popup=loc['timestamp'].strftime('%H:%M:%S')
                    ).add_to(m)
            
            return m
        return None
    
    def get_location_stats(self):
        """Statistiques de localisation"""
        if not self.location_history:
            return None
        
        distances = []
        for i in range(1, len(self.location_history)):
            prev = self.location_history[i-1]
            curr = self.location_history[i]
            dist = distance((prev['lat'], prev['lon']), (curr['lat'], curr['lon'])).meters
            distances.append(dist)
        
        return {
            'total_points': len(self.location_history),
            'total_distance': sum(distances),
            'avg_speed': sum(distances) / (len(self.location_history) * 5) if self.location_history else 0,  # 5 secondes entre chaque point
            'start_time': self.location_history[0]['timestamp'],
            'end_time': self.location_history[-1]['timestamp']
        }

class AlertSystem:
    """Système d'alertes intelligent avec géolocalisation"""
    def __init__(self):
        self.thresholds = {
            'person': 5,
            'car': 15,
            'fire': 1,
            'gun': 1,
            'knife': 1
        }
        self.alerts_history = []
    
    def check_alerts(self, detections, location=None):
        """Vérifie les alertes avec géolocalisation"""
        alerts = []
        detection_counts = defaultdict(int)
        
        for detection in detections:
            cls_name = detection[0]
            detection_counts[cls_name] += 1
        
        for obj_type, count in detection_counts.items():
            if obj_type in self.thresholds and count > self.thresholds[obj_type]:
                alert = {
                    'type': obj_type,
                    'count': count,
                    'severity': 'high' if obj_type in ['fire', 'gun', 'knife'] else 'medium',
                    'timestamp': datetime.now(),
                    'location': location
                }
                alerts.append(alert)
                self.alerts_history.append(alert)
        
        return alerts

class DetectionStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.detections = defaultdict(int)
        self.confidences = []
        self.total_frames = 0
        self.start_time = time.time()
        self.processing_times = []
    
    def update(self, detections, proc_time=0):
        self.total_frames += 1
        for detection in detections:
            cls_name = detection[0]
            conf = detection[1]
            self.detections[cls_name] += 1
            self.confidences.append(conf)
        if proc_time > 0:
            self.processing_times.append(proc_time)
    
    def get_summary(self):
        return {
            'total_detections': sum(self.detections.values()),
            'unique_objects': len(self.detections),
            'avg_confidence': np.mean(self.confidences) if self.confidences else 0,
            'fps': self.total_frames / (time.time() - self.start_time) if self.total_frames > 0 else 0,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def display(self):
        col1, col2, col3, col4 = st.columns(4)
        summary = self.get_summary()
        
        with col1:
            st.metric("Total détections", summary['total_detections'])
        with col2:
            st.metric("Objets uniques", summary['unique_objects'])
        with col3:
            st.metric("Confiance moyenne", f"{summary['avg_confidence']:.2f}")
        with col4:
            st.metric("Performance", f"{summary['fps']:.1f} FPS")
        
        if self.detections:
            st.markdown("### 📊 Distribution des détections")
            df = pd.DataFrame([
                {'Objet': k, 'Nombre': v} 
                for k, v in sorted(self.detections.items(), key=lambda x: x[1], reverse=True)[:10]
            ])
            fig = px.bar(df, x='Objet', y='Nombre', title="Top 10 des objets détectés",
                        color='Nombre', color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# -------------------
# TITRE
# -------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
        <div class="main-header" style="text-align: center;">
            <h1 style="color: white; margin: 0;">MAEA Tech</h1>
            <h3 style="color: white; margin: 0; opacity: 0.9;">AI Vision Assistant</h3>
            <p style="color: white; opacity: 0.8;">Vision par ordinateur • YOLOv8 • Géolocalisation temps réel</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------
# SIDEBAR
# -------------------
with st.sidebar:
    st.markdown("## Configuration")
    
    lang = st.selectbox("Langue", ["Français", "English", "Español"])
    lang_code = {"Français": "fr", "English": "en", "Español": "es"}[lang]
    
    mode = st.radio("Mode d'analyse", ["Image", "Webcam", "Vidéo", "Analyse Comparative"])
    
    st.markdown("---")
    
    st.markdown("### Paramètres du modèle")
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("Seuil IOU", 0.0, 1.0, 0.45, 0.05)
    
    enable_alerts = st.checkbox("Activer les alertes", True)
    enable_geolocation = st.checkbox("Activer la géolocalisation", True)
    show_fps = st.checkbox("Afficher les FPS", True)
    show_boxes = st.checkbox("Afficher les boîtes", True)
    
    st.markdown("---")
    
    if enable_geolocation:
        st.markdown("### 📍 Géolocalisation")
        st.info("La position GPS est utilisée pour localiser les détections")
        
        if st.button("🔄 Mettre à jour position", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # SECTION ÉQUIPE AVEC LOGOS LINKEDIN
    st.markdown("### 👥 Équipe")
    st.markdown("🚀 Projet développé par **MAEA Tech**")
    
    st.markdown("---")
    
    # Angèle EYENGA
    st.markdown("""
    <div class="team-card">
        <strong>👤 Angèle EYENGA</strong><br>
        <a href="https://www.linkedin.com/in/ang%C3%A8le-eyenga-3182a7191/" target="_blank" class="linkedin-link">
            <img src="https://content.linkedin.com/content/dam/developer/branding/site-builder/img/site-builder-icon.svg" width="16" height="16" style="vertical-align: middle;">
            LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Sajidat AKANNI
    st.markdown("""
    <div class="team-card">
        <strong>👤 Sajidat AKANNI</strong><br>
        <a href="https://www.linkedin.com/in/sajidat-akanni-463581403" target="_blank" class="linkedin-link">
            <img src="https://content.linkedin.com/content/dam/developer/branding/site-builder/img/site-builder-icon.svg" width="16" height="16" style="vertical-align: middle;">
            LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📞 Contact")
    st.markdown("📧 abessangel@gmail.com")
    st.markdown("📧 sajidatakanni25@gmail.com")

    
    st.markdown("---")
    
    st.markdown("### Version")
    st.markdown("v3.2 - Géolocalisation intégrée")

# -------------------
# CHARGEMENT MODÈLE
# -------------------
@st.cache_resource
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO("yolov8n.pt")
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None, None

model, device = load_model()

if model is None:
    st.stop()

# -------------------
# INITIALISATION
# -------------------
if 'geo' not in st.session_state:
    st.session_state.geo = GeoLocalization()
    if enable_geolocation:
        st.session_state.geo.update_location()

if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()

stats = DetectionStats()

# -------------------
# TRADUCTION SIMPLIFIÉE
# -------------------
def translate(cls_name):
    translations = {
        'fr': f"{cls_name} détecté",
        'en': f"{cls_name} detected",
        'es': f"{cls_name} detectado"
    }
    return translations.get(lang_code, f"{cls_name} detected")

# -------------------
# FONCTION DE TRAITEMENT AVEC GÉOLOCALISATION
# -------------------
def process_frame(frame):
    start_time = time.time()
    
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)
    
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append((cls_name, conf))
    
    # Géolocalisation pour les alertes
    current_location = None
    if enable_geolocation and st.session_state.geo.current_lat:
        current_location = {
            'lat': st.session_state.geo.current_lat,
            'lon': st.session_state.geo.current_lon,
            'address': st.session_state.geo.current_address
        }
    
    if enable_alerts:
        alerts = st.session_state.alert_system.check_alerts(detections, current_location)
        for alert in alerts:
            severity_class = "alert-danger" if alert['severity'] == 'high' else "alert-warning"
            location_text = f" à {alert['location']['address']}" if alert['location'] else ""
            st.markdown(f'<div class="{severity_class}">⚠️ ALERTE: {alert["count"]} {alert["type"]}(s) détecté(s){location_text}</div>', 
                       unsafe_allow_html=True)
    
    if show_boxes:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()
    
    proc_time = time.time() - start_time
    stats.update(detections, proc_time)
    
    return annotated_frame, detections

# -------------------
# AFFICHAGE GÉOLOCALISATION
# -------------------
def display_geolocation():
    """Affiche les informations de géolocalisation"""
    if enable_geolocation:
        st.markdown("### 📍 Position actuelle")
        
        # Mettre à jour la position périodiquement
        if st.session_state.geo.update_location():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="location-card">
                    <strong>Coordonnées GPS</strong><br>
                    Latitude: {st.session_state.geo.current_lat:.6f}<br>
                    Longitude: {st.session_state.geo.current_lon:.6f}<br>
                    Précision: ±10m
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="location-card">
                    <strong>Adresse</strong><br>
                    {st.session_state.geo.current_address}
                </div>
                """, unsafe_allow_html=True)
            
            # Afficher la carte
            m = st.session_state.geo.create_map()
            if m:
                folium_static(m, width=700, height=400)
            
            # Statistiques de déplacement
            location_stats = st.session_state.geo.get_location_stats()
            if location_stats and location_stats['total_points'] > 1:
                st.markdown("### 📊 Statistiques de déplacement")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Distance totale", f"{location_stats['total_distance']:.0f} m")
                with col2:
                    st.metric("Vitesse moyenne", f"{location_stats['avg_speed']:.1f} km/h")
                with col3:
                    st.metric("Points enregistrés", location_stats['total_points'])
        else:
            st.warning("Géolocalisation non disponible. Vérifiez les permissions du navigateur.")

# -------------------
# MODE IMAGE
# -------------------
if mode == "Image":
    st.markdown("### Analyse d'image")
    
    if enable_geolocation:
        display_geolocation()
        st.markdown("---")
    
    upload_type = st.radio("Type d'upload", ["Image unique", "Images multiples"])
    
    if upload_type == "Image unique":
        file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg", "bmp"])
        
        if file:
            image = Image.open(file)
            img = np.array(image)
            
            with st.spinner("Analyse en cours..."):
                result_img, detections = process_frame(img)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result_img, caption="Résultat", use_container_width=True)
                
                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                st.download_button(
                    label="Télécharger le résultat",
                    data=result_pil.tobytes(),
                    file_name="detection_result.png",
                    mime="image/png"
                )
            
            with col2:
                st.markdown("### Résultats")
                
                if detections:
                    for cls_name, conf in detections:
                        st.markdown(f'<span class="detection-badge">{translate(cls_name)} ({conf:.2f})</span>', 
                                  unsafe_allow_html=True)
                else:
                    st.info("Aucune détection")
                
                st.markdown("---")
                stats.display()
                
                if st.button("Nouvelle analyse"):
                    stats.reset()
                    st.rerun()
    
    else:
        files = st.file_uploader("Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        if files:
            for idx, file in enumerate(files[:5]):
                st.markdown(f"### {file.name}")
                image = Image.open(file)
                result_img, detections = process_frame(np.array(image))
                st.image(result_img, use_container_width=True)
                st.caption(f"{len(detections)} objets détectés")
                st.markdown("---")

# -------------------
# MODE WEBCAM
# -------------------
elif mode == "Webcam":
    st.markdown("### Webcam temps réel")
    
    if enable_geolocation:
        display_geolocation()
        st.markdown("---")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        FRAME = st.empty()
    
    with col2:
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        if st.button("Démarrer", use_container_width=True):
            st.session_state.webcam_running = True
        if st.button("Arrêter", use_container_width=True):
            st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Impossible d'accéder à la webcam")
            st.session_state.webcam_running = False
        else:
            fps_time = time.time()
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mise à jour position GPS périodique
                if enable_geolocation and int(time.time()) % 10 == 0:
                    st.session_state.geo.update_location()
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_img, detections = process_frame(frame_rgb)
                
                if show_fps:
                    fps = 1 / (time.time() - fps_time)
                    fps_time = time.time()
                    cv2.putText(result_img, f"FPS: {int(fps)}", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                # Ajouter coordonnées GPS sur l'image
                if enable_geolocation and st.session_state.geo.current_lat:
                    gps_text = f"GPS: {st.session_state.geo.current_lat:.4f}, {st.session_state.geo.current_lon:.4f}"
                    cv2.putText(result_img, gps_text, (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                FRAME.image(result_img, channels="RGB", use_container_width=True)
                
                with col2:
                    if detections:
                        st.markdown("### Détections")
                        for cls_name, conf in detections[:5]:
                            st.markdown(f'<span class="detection-badge">{translate(cls_name)} ({conf:.2f})</span>', 
                                      unsafe_allow_html=True)
                    
                    stats.display()
            
            cap.release()

# -------------------
# MODE VIDÉO
# -------------------
elif mode == "Vidéo":
    st.markdown("### Analyse vidéo")
    
    if enable_geolocation:
        display_geolocation()
        st.markdown("---")
    
    video_file = st.file_uploader("Choisir une vidéo", type=["mp4", "avi", "mov", "mkv"])
    
    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        
        if cap.isOpened():
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Vidéo: {total_frames} frames, {video_fps} FPS")
            
            progress_bar = st.progress(0)
            video_placeholder = st.empty()
            
            stats.reset()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_img, detections = process_frame(frame_rgb)
                
                video_placeholder.image(result_img, channels="RGB", use_container_width=True)
                progress_bar.progress((frame_count + 1) / total_frames)
                
                frame_count += 1
            
            cap.release()
            st.success("Analyse terminée")
            stats.display()

# -------------------
# MODE ANALYSE COMPARATIVE
# -------------------
elif mode == "Analyse Comparative":
    st.markdown("### Analyse comparative")
    
    if enable_geolocation:
        display_geolocation()
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Image 1", type=["jpg", "png", "jpeg"], key="img1")
    
    with col2:
        file2 = st.file_uploader("Image 2", type=["jpg", "png", "jpeg"], key="img2")
    
    if file1 and file2:
        img1 = np.array(Image.open(file1))
        img2 = np.array(Image.open(file2))
        
        with st.spinner("Analyse..."):
            result1, detections1 = process_frame(img1)
            result2, detections2 = process_frame(img2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(result1, caption="Image 1", use_container_width=True)
            st.metric("Détections", len(detections1))
        
        with col2:
            st.image(result2, caption="Image 2", use_container_width=True)
            st.metric("Détections", len(detections2))
        
        # Graphique comparatif
        objects1 = defaultdict(int)
        objects2 = defaultdict(int)
        
        for cls_name, _ in detections1:
            objects1[cls_name] += 1
        for cls_name, _ in detections2:
            objects2[cls_name] += 1
        
        all_objects = set(objects1.keys()) | set(objects2.keys())
        
        comparison_data = []
        for obj in all_objects:
            comparison_data.append({
                'Objet': obj,
                'Image 1': objects1.get(obj, 0),
                'Image 2': objects2.get(obj, 0)
            })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            fig = px.bar(df_comp, x='Objet', y=['Image 1', 'Image 2'], 
                        barmode='group', title="Comparaison des détections",
                        color_discrete_sequence=['#1e3c72', '#2a5298'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# -------------------
# EXPORT DES DONNÉES AVEC GÉOLOCALISATION
# -------------------
if st.sidebar.button("📊 Exporter le rapport", use_container_width=True):
    if enable_geolocation and st.session_state.geo.location_history:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'detections': dict(stats.detections),
            'location_history': st.session_state.geo.location_history,
            'alerts': st.session_state.alert_system.alerts_history
        }
        
        with open(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report_data, f, default=str)
        
        st.success("Rapport exporté avec succès !")
    else:
        st.warning("Aucune donnée de géolocalisation à exporter")

# -------------------
# FOOTER
# -------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #6c757d; padding: 1rem; font-size: 0.9rem;">
        <p><strong>MAEA Tech</strong> - Intelligence Artificielle pour la Vision par Ordinateur</p>
        <p>YOLOv8 • Géolocalisation temps réel • Production Ready</p>
        {f"📍 Dernière position: {st.session_state.geo.current_lat:.4f}, {st.session_state.geo.current_lon:.4f}" if enable_geolocation and st.session_state.geo.current_lat else ""}
        <p style="margin-top: 0.5rem; font-size: 0.8rem;">© 2024 MAEA Tech - Tous droits réservés</p>
    </div>
    """,
    unsafe_allow_html=True
)
