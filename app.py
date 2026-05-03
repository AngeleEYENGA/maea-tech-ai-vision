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
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.distance import distance
import requests
import os
from pathlib import Path
import subprocess
import platform
import random

# Reverse geocoder optionnel
try:
    import reverse_geocoder as rg
    RG_AVAILABLE = True
except ImportError:
    RG_AVAILABLE = False

warnings.filterwarnings('ignore')

# -------------------
# PAGE CONFIG - Optimisé mobile
# -------------------
st.set_page_config(
    page_title="MAEA Tech - AI Vision Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# CSS RESPONSIVE POUR MOBILE
# -------------------
st.markdown("""
    <style>
    /* Styles généraux */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Responsive pour mobile */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        .main-header h3 {
            font-size: 1rem !important;
        }
        .detection-badge {
            font-size: 0.7rem !important;
            padding: 0.2rem 0.5rem !important;
        }
        .location-card {
            padding: 0.5rem !important;
            font-size: 0.8rem !important;
        }
        .stMetric {
            font-size: 0.8rem !important;
        }
        button {
            font-size: 0.8rem !important;
            padding: 0.3rem !important;
        }
    }
    
    /* Styles desktop */
    @media (min-width: 769px) {
        .main-header h1 {
            font-size: 2.5rem !important;
        }
        .main-header h3 {
            font-size: 1.5rem !important;
        }
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
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .alert-warning {
        background: #fd7e14;
        padding: 0.75rem;
        border-radius: 6px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .location-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .team-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .badge-gps {
        background: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-left: 5px;
    }
    
    /* Optimisation pour les boutons tactiles */
    button, .stButton button {
        min-height: 44px !important;
        min-width: 44px !important;
    }
    
    /* Amélioration de la lisibilité sur mobile */
    .stMarkdown, .stText, .stMetric {
        font-size: 1rem !important;
    }
    
    /* Ajustement des colonnes sur mobile */
    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        .stColumn {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# CONFIGURATION DE SAUVEGARDE
# -------------------
class SaveManager:
    """Gestionnaire de sauvegarde automatique"""
    
    def __init__(self, base_path="maea_saves"):
        self.base_path = Path(base_path)
        self.create_directories()
        
    def create_directories(self):
        """Crée la structure de dossiers"""
        directories = [
            "images/raw",
            "images/detected",
            "videos/raw",
            "videos/detected",
            "reports",
            "logs",
            "detections"
        ]
        
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Créer un fichier README
        readme_path = self.base_path / "README.txt"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"""=== MAEA Tech - Données sauvegardées ===

Structure des dossiers :
- images/raw : Images originales uploadées
- images/detected : Images après détection
- videos/raw : Vidéos originales uploadées
- videos/detected : Vidéos après détection
- reports : Rapports JSON des analyses
- logs : Logs d'activité
- detections : Données de détection (CSV)

Dernière mise à jour : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
    
    def save_image(self, image_array, filename, subfolder="detected"):
        """Sauvegarde une image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{filename}"
            filepath = self.base_path / f"images/{subfolder}" / safe_filename
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                save_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                save_img = image_array
            
            cv2.imwrite(str(filepath), save_img)
            return str(filepath)
        except Exception as e:
            print(f"Erreur sauvegarde image: {e}")
            return None
    
    def save_report(self, report_data, report_name="analysis"):
        """Sauvegarde un rapport JSON"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{report_name}.json"
            filepath = self.base_path / "reports" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            return str(filepath)
        except Exception as e:
            print(f"Erreur sauvegarde rapport: {e}")
            return None
    
    def save_detections_csv(self, detections, filename="detections"):
        """Sauvegarde les détections en CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.base_path / "detections" / f"{timestamp}_{filename}.csv"
            
            df = pd.DataFrame([
                {'timestamp': datetime.now(), 'objet': cls, 'confiance': conf}
                for cls, conf in detections
            ])
            df.to_csv(filepath, index=False, encoding='utf-8')
            return str(filepath)
        except Exception as e:
            print(f"Erreur sauvegarde CSV: {e}")
            return None
    
    def log_action(self, action, details=""):
        """Enregistre une action dans les logs"""
        try:
            log_file = self.base_path / "logs" / f"log_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action}: {details}\n")
        except Exception as e:
            print(f"Erreur log: {e}")

# -------------------
# FONCTIONS DE TRAITEMENT D'IMAGE
# -------------------
def prepare_image(image):
    """Convertit l'image en RGB (3 canaux) pour le modèle"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    return image

def load_image(file):
    """Charge et convertit une image correctement"""
    image = Image.open(file)
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    elif image.mode == 'L':
        image = image.convert('RGB')
    
    return np.array(image)

# -------------------
# CLASSE DE GÉOLOCALISATION
# -------------------
class PreciseGeoLocalization:
    def __init__(self):
        self.current_lat = None
        self.current_lon = None
        self.current_address = None
        self.location_history = []
        self.geolocator = Nominatim(user_agent="maea_tech_vision")
        self.gps_source = "IP API"
        self.last_ip_update = None
        
    def get_location_by_ip(self):
        """Récupère la localisation réelle via l'adresse IP"""
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                lat = data.get('latitude')
                lon = data.get('longitude')
                if lat and lon:
                    return {
                        'lat': float(lat),
                        'lon': float(lon),
                        'city': data.get('city', 'Inconnue'),
                        'country': data.get('country_name', 'Inconnu'),
                        'address': f"{data.get('city', '')}, {data.get('country_name', '')}",
                        'source': 'IP API'
                    }
        except Exception as e:
            pass
        
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return {
                        'lat': float(data.get('lat', 0)),
                        'lon': float(data.get('lon', 0)),
                        'city': data.get('city', 'Inconnue'),
                        'country': data.get('country', 'Inconnu'),
                        'address': f"{data.get('city', '')}, {data.get('country', '')}",
                        'source': 'IP-API'
                    }
        except Exception as e:
            pass
        
        return None
    
    def get_location_simulation(self):
        """Version démo avec mouvement réaliste"""
        import math
        
        if self.current_lat is None:
            self.current_lat = 48.8584
            self.current_lon = 2.2945
        else:
            t = time.time() * 0.05
            self.current_lat += math.sin(t) * 0.0003
            self.current_lon += math.cos(t * 0.7) * 0.0003
        
        return {
            'lat': self.current_lat,
            'lon': self.current_lon,
            'address': self.current_address or "Paris, France",
            'source': 'Simulation'
        }
    
    def update_location(self, use_real_gps=True):
        location_data = None
        
        if use_real_gps:
            location_data = self.get_location_by_ip()
        
        if location_data and location_data.get('lat'):
            self.current_lat = location_data['lat']
            self.current_lon = location_data['lon']
            self.current_address = location_data['address']
            self.gps_source = location_data['source']
        else:
            sim_data = self.get_location_simulation()
            self.current_lat = sim_data['lat']
            self.current_lon = sim_data['lon']
            self.current_address = sim_data['address']
            self.gps_source = sim_data['source']
        
        self.location_history.append({
            'timestamp': datetime.now(),
            'lat': self.current_lat,
            'lon': self.current_lon,
            'address': self.current_address,
            'source': self.gps_source
        })
        
        if len(self.location_history) > 200:
            self.location_history.pop(0)
        
        return True
    
    def create_map(self):
        if self.current_lat and self.current_lon:
            m = folium.Map(
                location=[self.current_lat, self.current_lon],
                zoom_start=16,
                control_scale=True
            )
            
            radius = 50 if self.gps_source == "IP API" else 500
            
            folium.Circle(
                [self.current_lat, self.current_lon],
                radius=radius,
                color='red' if self.gps_source == "IP API" else 'orange',
                fill=True,
                fill_opacity=0.3,
                popup=f'Précision: {radius}m'
            ).add_to(m)
            
            folium.Marker(
                [self.current_lat, self.current_lon],
                popup=f"""
                <b>Position actuelle</b><br>
                📍 {self.current_address}<br>
                🛰️ Source: {self.gps_source}
                """,
                icon=folium.Icon(color='green' if self.gps_source == "IP API" else 'orange'),
                tooltip="Position GPS"
            ).add_to(m)
            
            if len(self.location_history) > 1:
                points = [[loc['lat'], loc['lon']] for loc in self.location_history[-20:]]
                folium.PolyLine(points, color='blue', weight=2, opacity=0.6).add_to(m)
            
            return m
        return None
    
    def get_location_stats(self):
        if len(self.location_history) < 2:
            return None
        
        distances = []
        for i in range(1, len(self.location_history)):
            prev = self.location_history[i-1]
            curr = self.location_history[i]
            try:
                dist = distance((prev['lat'], prev['lon']), (curr['lat'], curr['lon'])).meters
                distances.append(dist)
            except:
                pass
        
        total_distance = sum(distances)
        time_diff = (self.location_history[-1]['timestamp'] - self.location_history[0]['timestamp']).total_seconds()
        avg_speed = (total_distance / 1000) / (time_diff / 3600) if time_diff > 0 else 0
        
        return {
            'total_points': len(self.location_history),
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'max_speed': max(distances) / 5 if distances else 0,
            'duration_seconds': time_diff,
            'source': self.gps_source
        }

class AlertSystem:
    def __init__(self):
        self.thresholds = {'person': 5, 'car': 15, 'fire': 1, 'gun': 1, 'knife': 1}
        self.alerts_history = []
    
    def check_alerts(self, detections, location=None):
        alerts = []
        detection_counts = defaultdict(int)
        for detection in detections:
            detection_counts[detection[0]] += 1
        
        for obj_type, count in detection_counts.items():
            if obj_type in self.thresholds and count > self.thresholds[obj_type]:
                alerts.append({
                    'type': obj_type,
                    'count': count,
                    'severity': 'high' if obj_type in ['fire', 'gun', 'knife'] else 'medium',
                    'timestamp': datetime.now(),
                    'location': location
                })
                self.alerts_history.append(alerts[-1])
        return alerts

class DetectionStats:
    def __init__(self):
        self.reset()
        self.chart_counter = 0
    
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
        elapsed_time = time.time() - self.start_time
        return {
            'total_detections': sum(self.detections.values()),
            'unique_objects': len(self.detections),
            'avg_confidence': np.mean(self.confidences) if self.confidences else 0,
            'fps': self.total_frames / elapsed_time if elapsed_time > 0 and self.total_frames > 0 else 0,
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
            fig.update_layout(height=400, autosize=True)
            # Clé unique pour éviter les doublons
            self.chart_counter += 1
            st.plotly_chart(fig, use_container_width=True, key=f"stats_chart_{self.chart_counter}_{int(time.time())}")

# -------------------
# TITRE
# -------------------
st.markdown("""
    <div class="main-header" style="text-align: center;">
        <h1 style="color: white; margin: 0;">MAEA Tech</h1>
        <h3 style="color: white; margin: 0;">AI Vision Assistant</h3>
        <p style="color: white; opacity: 0.9; margin: 0;">Vision par ordinateur • YOLOv8 • Mobile Ready</p>
    </div>
""", unsafe_allow_html=True)

# -------------------
# INITIALISATION DES SESSIONS (AVANT LA SIDEBAR)
# -------------------
if 'stats' not in st.session_state:
    st.session_state.stats = DetectionStats()
if 'geo' not in st.session_state:
    st.session_state.geo = PreciseGeoLocalization()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'save_manager' not in st.session_state:
    st.session_state.save_manager = SaveManager()

stats = st.session_state.stats
save_manager = st.session_state.save_manager

# -------------------
# SIDEBAR
# -------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    lang = st.selectbox("🌐 Langue", ["Français", "English", "Español"])
    lang_code = {"Français": "fr", "English": "en", "Español": "es"}[lang]
    
    mode = st.radio("📱 Mode d'analyse", ["📷 Image", "🎥 Webcam", "🎬 Vidéo", "📊 Analyse Comparative"])
    
    st.markdown("---")
    st.markdown("### 🎯 Paramètres")
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("Seuil IOU", 0.0, 1.0, 0.45, 0.05)
    
    enable_alerts = st.checkbox("🔔 Activer les alertes", True)
    enable_geolocation = st.checkbox("📍 Activer la géolocalisation", True)
    use_real_gps = st.checkbox("🛰️ GPS réel (IP)", True) if enable_geolocation else False
    show_fps = st.checkbox("📊 Afficher les FPS", True)
    show_boxes = st.checkbox("📦 Afficher les boîtes", True)
    
    st.markdown("---")
    st.markdown("### 💾 Sauvegarde")
    
    enable_auto_save = st.checkbox("Sauvegarde automatique", True)
    save_images = st.checkbox("Sauvegarder les images", True)
    save_reports = st.checkbox("Sauvegarder les rapports", True)
    save_detections_csv = st.checkbox("Sauvegarder CSV", True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📂 Dossier", use_container_width=True):
            save_path = os.path.abspath("maea_saves")
            if platform.system() == "Windows":
                os.startfile(save_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", save_path])
            else:
                subprocess.run(["xdg-open", save_path])
    
    with col2:
        if st.button("💾 Stats", use_container_width=True):
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'detections': dict(stats.detections),
                'summary': stats.get_summary()
            }
            save_manager.save_report(report_data, "manual_stats")
            st.success("✅ Stats sauvegardées!")
    
    st.markdown("---")
    st.markdown("### 👥 Équipe")
    st.markdown("""
    <div class="team-card">
        <strong>👤 Angèle EYENGA</strong><br>
        <a href="https://www.linkedin.com/in/ang%C3%A8le-eyenga-3182a7191/" target="_blank">🔗 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("📧 abessangel@gmail.com")
    st.markdown("---")
    st.markdown("### 📱 Version")
    st.markdown("v3.6 - Mobile Ready")

# -------------------
# CHARGEMENT MODÈLE
# -------------------
@st.cache_resource
def load_model():
    with st.spinner("Chargement du modèle YOLO..."):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO("yolov8n.pt")
            model.to(device)
            return model, device
        except Exception as e:
            st.error(f"Erreur: {e}")
            return None, None

model, device = load_model()

if model is None:
    st.stop()

# Mise à jour initiale de la géolocalisation
if enable_geolocation:
    st.session_state.geo.update_location(use_real_gps)

# -------------------
# TRADUCTION
# -------------------
def translate(cls_name):
    translations = {
        'fr': f"{cls_name} détecté",
        'en': f"{cls_name} detected",
        'es': f"{cls_name} detectado"
    }
    return translations.get(lang_code, f"{cls_name} detected")

# -------------------
# FONCTION DE TRAITEMENT
# -------------------
def process_frame(frame, source_name="unknown", save_frame=False):
    start_time = time.time()
    
    frame = prepare_image(frame)
    
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)
    
    detections = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append((cls_name, conf))
    
    current_location = None
    if enable_geolocation and st.session_state.geo.current_lat:
        current_location = {
            'lat': st.session_state.geo.current_lat,
            'lon': st.session_state.geo.current_lon,
            'address': st.session_state.geo.current_address,
            'source': st.session_state.geo.gps_source
        }
    
    if enable_alerts:
        alerts = st.session_state.alert_system.check_alerts(detections, current_location)
        for alert in alerts:
            severity_class = "alert-danger" if alert['severity'] == 'high' else "alert-warning"
            location_text = f" à {alert['location']['address']}" if alert.get('location') else ""
            st.markdown(f'<div class="{severity_class}">⚠️ ALERTE: {alert["count"]} {alert["type"]}(s) détecté(s){location_text}</div>', 
                       unsafe_allow_html=True)
            
            if enable_auto_save and alert['severity'] == 'high':
                save_manager.log_action("ALERTE_CRITIQUE", f"{alert['count']} {alert['type']}(s)")
    
    if show_boxes and results[0].boxes is not None:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame.copy()
    
    if enable_auto_save and save_frame:
        if save_images:
            save_path = save_manager.save_image(annotated_frame, source_name, "detected")
            if save_path:
                save_manager.log_action("SAUVEGARDE_IMAGE", save_path)
        
        if save_detections_csv and detections:
            csv_path = save_manager.save_detections_csv(detections, source_name)
            if csv_path:
                save_manager.log_action("SAUVEGARDE_CSV", csv_path)
    
    proc_time = time.time() - start_time
    stats.update(detections, proc_time)
    
    return annotated_frame, detections

# -------------------
# AFFICHAGE GÉOLOCALISATION
# -------------------
def display_geolocation():
    if enable_geolocation:
        st.markdown("### 📍 Géolocalisation")
        
        col_refresh, col_status = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄", key="refresh_geo", help="Actualiser la position"):
                st.session_state.geo.update_location(use_real_gps)
                st.rerun()
        
        with col_status:
            source_badge = "🛰️ GPS Réel" if st.session_state.geo.gps_source == "IP API" else "🎮 Simulation"
            st.markdown(f'<span class="badge-gps">{source_badge}</span>', unsafe_allow_html=True)
        
        st.session_state.geo.update_location(use_real_gps)
        
        if st.session_state.geo.current_lat:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="location-card">
                    <strong>🎯 Coordonnées</strong><br>
                    📐 {st.session_state.geo.current_lat:.6f}<br>
                    📏 {st.session_state.geo.current_lon:.6f}<br>
                    📡 {st.session_state.geo.gps_source}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="location-card">
                    <strong>🏙️ Lieu</strong><br>
                    {st.session_state.geo.current_address[:50]}
                </div>
                """, unsafe_allow_html=True)
            
            m = st.session_state.geo.create_map()
            if m:
                folium_static(m, width=None, height=300)
            
            location_stats = st.session_state.geo.get_location_stats()
            if location_stats and location_stats['total_points'] > 1:
                with st.expander("📊 Statistiques de déplacement"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Points GPS", location_stats['total_points'])
                    with col2:
                        st.metric("Distance", f"{location_stats['total_distance']:.0f} m")
                    with col3:
                        st.metric("Vitesse", f"{location_stats['avg_speed']:.1f} km/h")
        else:
            st.warning("Géolocalisation en cours...")

# -------------------
# MODE IMAGE
# -------------------
if mode == "📷 Image":
    st.markdown("### 📸 Analyse d'image")
    
    if enable_geolocation:
        display_geolocation()
        st.markdown("---")
    
    upload_type = st.radio("Type", ["Image unique", "Images multiples"])
    
    if upload_type == "Image unique":
        file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg", "bmp"], key="single_image")
        
        if file:
            img = load_image(file)
            original_filename = file.name
            
            if enable_auto_save and save_images:
                save_manager.save_image(img, f"original_{original_filename}", "raw")
            
            with st.spinner("Analyse en cours..."):
                result_img, detections = process_frame(img, original_filename, save_frame=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result_img, caption="Résultat", use_container_width=True)
                
                result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                is_success, buffer = cv2.imencode(".png", result_bgr)
                if is_success:
                    st.download_button(
                        label="📥 Télécharger",
                        data=buffer.tobytes(),
                        file_name=f"detection_{original_filename}.png",
                        mime="image/png",
                        key="download_single"
                    )
            
            with col2:
                st.markdown("### Résultats")
                if detections:
                    for cls_name, conf in detections[:10]:
                        st.markdown(f'<span class="detection-badge">{translate(cls_name)} ({conf:.2f})</span>', 
                                  unsafe_allow_html=True)
                    if len(detections) > 10:
                        st.caption(f"+ {len(detections) - 10} autres")
                else:
                    st.info("Aucune détection")
                
                st.markdown("---")
                stats.display()
                
                if enable_auto_save and save_reports:
                    report_data = {
                        'type': 'image_analysis',
                        'filename': original_filename,
                        'timestamp': datetime.now().isoformat(),
                        'detections': [
                            {'object': cls, 'confidence': conf}
                            for cls, conf in detections
                        ],
                        'location': {
                            'lat': st.session_state.geo.current_lat,
                            'lon': st.session_state.geo.current_lon,
                            'address': st.session_state.geo.current_address
                        } if enable_geolocation else None,
                        'stats': stats.get_summary()
                    }
                    save_manager.save_report(report_data, f"image_{original_filename}")
                
                if st.button("🔄 Nouvelle analyse", key="new_analysis"):
                    stats.reset()
                    st.rerun()
    
    else:
        files = st.file_uploader(
            "Choisir plusieurs images", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            key="multiple_images"
        )
        
        if files:
            for idx, file in enumerate(files[:5]):
                st.markdown(f"### 📷 {file.name}")
                
                img = load_image(file)
                
                with st.spinner(f"Analyse de {file.name}..."):
                    result_img, detections = process_frame(img, file.name, save_frame=True)
                
                st.image(result_img, use_container_width=True)
                
                if detections:
                    st.markdown("**Détections:**")
                    cols = st.columns(min(len(detections), 4))
                    for i, (cls_name, conf) in enumerate(detections[:8]):
                        with cols[i % 4]:
                            st.markdown(f'<span class="detection-badge">{translate(cls_name)}</span>', 
                                      unsafe_allow_html=True)
                
                st.markdown("---")
            
            stats.display()

# -------------------
# MODE WEBCAM
# -------------------
elif mode == "🎥 Webcam":
    st.markdown("### 🎥 Webcam temps réel")
    st.info("💡 La géolocalisation utilise votre adresse IP")
    
    if enable_geolocation:
        display_geolocation()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        FRAME = st.empty()
    
    with col2:
        st.markdown("### Contrôles")
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            start_btn = st.button("▶️ Démarrer", key="webcam_start", use_container_width=True)
        
        with btn_col2:
            stop_btn = st.button("⏹️ Arrêter", key="webcam_stop", use_container_width=True)
        
        if enable_auto_save:
            capture_interval = st.slider("Capture auto (secondes)", 5, 60, 30)
    
    if start_btn:
        st.session_state.webcam_running = True
    
    if stop_btn:
        st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Webcam inaccessible")
            st.session_state.webcam_running = False
        else:
            fps_time = time.time()
            frame_count = 0
            last_save_time = time.time()
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if enable_auto_save and save_images:
                    if (time.time() - last_save_time) > capture_interval:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        timestamp = datetime.now().strftime("%H%M%S")
                        save_manager.save_image(frame_rgb, f"webcam_auto_{timestamp}", "raw")
                        last_save_time = time.time()
                
                if enable_geolocation and frame_count % 30 == 0:
                    st.session_state.geo.update_location(use_real_gps)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_img, detections = process_frame(frame_rgb, "webcam", save_frame=False)
                
                if show_fps and frame_count > 1:
                    current_fps = 1 / (time.time() - fps_time) if (time.time() - fps_time) > 0 else 0
                    fps_time = time.time()
                    cv2.putText(result_img, f"FPS: {int(current_fps)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                FRAME.image(result_img, channels="RGB", use_container_width=True)
                
                with col2:
                    if detections:
                        st.markdown("### 🔍 Détections")
                        for cls_name, conf in detections[:5]:
                            st.markdown(f'<span class="detection-badge">{translate(cls_name)}</span>', 
                                      unsafe_allow_html=True)
                    
                    stats.display()
                
                time.sleep(0.03)
            
            cap.release()
            st.session_state.webcam_running = False
            st.info("Webcam arrêtée")

# -------------------
# MODE VIDÉO
# -------------------
elif mode == "🎬 Vidéo":
    st.markdown("### 🎬 Analyse vidéo")
    
    if enable_geolocation:
        display_geolocation()
    
    video_file = st.file_uploader("Choisir une vidéo", type=["mp4", "avi", "mov", "mkv"], key="video_file")
    
    if video_file:
        video_filename = video_file.name
        
        with st.spinner("Chargement..."):
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"📹 {total_frames} frames")
            
            progress_bar = st.progress(0)
            video_placeholder = st.empty()
            
            stats.reset()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_img, detections = process_frame(frame_rgb, video_filename, save_frame=False)
                
                video_placeholder.image(result_img, channels="RGB", use_container_width=True)
                progress_bar.progress((frame_count + 1) / total_frames)
                frame_count += 1
            
            cap.release()
            
            import os
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")
            
            st.success("✅ Analyse terminée")
            stats.display()

# -------------------
# MODE ANALYSE COMPARATIVE
# -------------------
elif mode == "📊 Analyse Comparative":
    st.markdown("### 📊 Analyse comparative")
    
    if enable_geolocation:
        display_geolocation()
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Image 1", type=["jpg", "png", "jpeg"], key="compare_img1")
    
    with col2:
        file2 = st.file_uploader("Image 2", type=["jpg", "png", "jpeg"], key="compare_img2")
    
    if file1 and file2:
        img1 = load_image(file1)
        img2 = load_image(file2)
        
        with st.spinner("Analyse..."):
            result1, detections1 = process_frame(img1, file1.name, save_frame=True)
            result2, detections2 = process_frame(img2, file2.name, save_frame=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(result1, caption="Image 1", use_container_width=True)
            st.metric("Détections", len(detections1))
        
        with col2:
            st.image(result2, caption="Image 2", use_container_width=True)
            st.metric("Détections", len(detections2))
        
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
                        barmode='group', title="Comparaison")
            fig.update_layout(height=400, autosize=True)
            # Clé unique pour le graphique comparatif
            st.plotly_chart(fig, use_container_width=True, key=f"compare_chart_{int(time.time())}_{random.randint(0, 10000)}")

# -------------------
# FOOTER
# -------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p><strong>MAEA Tech</strong> - AI Vision Assistant</p>
        <p>YOLOv8 • Mobile Ready</p>
        {f"<p>📍 {st.session_state.geo.current_lat:.4f}, {st.session_state.geo.current_lon:.4f}</p>" if enable_geolocation and st.session_state.geo.current_lat else ""}
        <p style="font-size: 0.7rem;">© 2024 MAEA Tech</p>
    </div>
    """,
    unsafe_allow_html=True
)
