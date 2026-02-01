import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates
from upscale import JewelryUpscaler
from search import StructuralSearchEngine

st.set_page_config(page_title="Jewelry AI - Task 1", layout="wide")
st.title("💎 Structural Jewelry Search")

@st.cache_resource
def load_all():
    engine = StructuralSearchEngine()
    if not os.path.exists('catalog') or not os.listdir('catalog'):
        os.makedirs('catalog', exist_ok=True)
    elif engine.index is None: 
        engine.build_index('catalog') 
    upscaler = JewelryUpscaler()
    detector = YOLO("yolov8n.pt")
    return engine, upscaler, detector

try:
    search_engine, upscaler, detector = load_all()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

if 'selected_match' not in st.session_state: st.session_state.selected_match = None
if 'customized_image' not in st.session_state: st.session_state.customized_image = None

tab1, tab2 = st.tabs(["1. Visual Search", "2. Customization"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Query Image")
        uploaded_file = st.file_uploader("Upload Ring", type=['jpg', 'png'])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.write("Crop the ring:")
            final_crop = st_cropper(img, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
            
            if final_crop and st.button("🔍 Search by Structure"):
                final_crop.save("query.jpg")
                
                st.divider()
                st.caption("How the AI sees your ring:")
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    processed_view = search_engine.preprocess_structural("query.jpg")
                    st.image(processed_view, caption="Structure", use_container_width=True)
                
                with v_col2:
                    # Debug print helps confirm new code runs
                    heatmap = search_engine.explain_match(final_crop)
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_rgb, caption="AI Focus", use_container_width=True)

                matches = search_engine.find_similar("query.jpg", top_k=3)
                
                with col2:
                    st.subheader("Structural Matches")
                    if not matches: 
                        st.error("🚫 No similar designs found.")
                    else:
                        m_cols = st.columns(3)
                        for i, (name, score) in enumerate(matches):
                            with m_cols[i]:
                                match_path = os.path.join('catalog', name)
                                if score >= 0.99:
                                    st.success(f"PERFECT: {int(score*100)}%")
                                else:
                                    st.caption(f"Confidence: {int(score*100)}%")
                                st.image(match_path, use_container_width=True)
                                if st.button(f"Customize", key=f"btn_{i}"):
                                    st.session_state.selected_match = match_path
                                    st.session_state.customized_image = Image.open(match_path).convert("RGBA")
                                    st.success("Selected! Go to Tab 2.")

with tab2:
    if st.session_state.customized_image:
        st.header("Customize Variable Attributes")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("Click center of gemstone:")
            coords = streamlit_image_coordinates(st.session_state.customized_image, width=400)
        with c2:
            gem = st.selectbox("Choose New Stone:", ["Ruby", "Sapphire", "Diamond"])
            if coords and st.button("Apply Change"):
                base = st.session_state.customized_image.copy()
                stone_path = f"stone_{gem.lower()}.png"
                if os.path.exists(stone_path):
                    stone = Image.open(stone_path).convert("RGBA").resize((80,80))
                    base.paste(stone, (coords['x']-40, coords['y']-40), stone)
                    st.session_state.customized_image = base
                    st.rerun()
                else:
                    st.error("Stone image not found.")
    else:
        st.info("Please select a ring from the Search tab first.")