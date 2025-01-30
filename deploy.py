import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os

st.title("Surgical Tool Detection in Real-time")

option = st.selectbox(
    "Choose an option:",
    ("Start Detection", "Stop Detection", "Settings")
)

# Use a relative path for the model
model_path = os.path.join("runs", "detect", "train24", "weights", "best.pt")

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check your file path.")
else:
    model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if option == "Start Detection":
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        
        annotated_frame = results[0].plot() 
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        time.sleep(0.1)

    cap.release()

elif option == "Stop Detection":
    st.write("Detection stopped. Please restart the application.")

elif option == "Settings":
    st.write("Adjust detection settings here.")
