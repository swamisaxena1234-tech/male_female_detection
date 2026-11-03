import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# Load model
model = load_model("gender_detection_model.h5")

st.title("Gender Detection App with Webcam")
st.write("You can upload an image or capture one using your webcam to predict Male or Female.")

# Option to choose input method
input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            img_resized = img.resize((64, 64))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.5:
                st.success("Prediction: **Female**")
            else:
                st.success("Prediction: **Male**")

else:  # Webcam
    st.write("Click on 'Capture Image' to take a photo using your webcam.")
    capture = st.button("Capture Image")
    
    if capture:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Convert BGR to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            st.image(img_pil, caption="Captured Image", use_column_width=True)

            # Predict
            img_resized = img_pil.resize((64, 64))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.5:
                st.success("Prediction: **Female**")
            else:
                st.success("Prediction: **Male**")
        else:
            st.error("Failed to capture image. Please try again.")

