import streamlit as st
from PIL import Image
import io
from model_inference import load_model, predict_image
import os

# 1. Load the model once at startup
# Assuming 'best_model.pth' is in the same directory as app.py or in the root of the project
MODEL_PATH = 'best_model.pth'
model = load_model(MODEL_PATH)

# 2. Set page config title to "Pneumonia Detector" with a centered layout
st.set_page_config(
    page_title="Pneumonia Detector",
    layout="centered",
    initial_sidebar_state="auto",
)

# 3. Add a header and brief subtitle
st.title("🩺 X-Ray Pneumonia Detector")
st.write("Upload a chest X-ray image to detect pneumonia.")

# 4. Add an image uploader widget
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image preview
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    st.write("")
    
    # Add an "Analyze Image" button
    if st.button("Analyze Image"):
        # Create a temporary file to save the uploaded image
        # This is necessary because predict_image expects a file path
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Pass the temporary file path and model to predict_image()
        prediction, confidence = predict_image(temp_image_path, model)
        
        # Display the prediction
        if prediction == "Pneumonia":
            st.error(f"Pneumonia Detected! Confidence: {confidence:.2f}%")
        else:
            st.success(f"Normal. Confidence: {confidence:.2f}%")
            
        # Clean up the temporary file
        os.remove(temp_image_path)
