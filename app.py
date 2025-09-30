# app.py
import os
import streamlit as st
from helper import show_predictions
from predictor import predict_on_image

st.set_page_config(page_title="Animal Threat Detection", layout="centered")
st.title("🐾 Animal Threat Detection with YOLOv8 OBB")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = os.path.join("temp.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.image(image_path, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Inference"):
        with st.spinner("Running YOLOv8 OBB Inference..."):
            results = predict_on_image(image_path)
            show_predictions(results)