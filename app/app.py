import sys
import os

# Add app directory to Python path
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from predict import predict
from PIL import Image
import tempfile


st.set_page_config(
    page_title="AI-Based Crop Pest Detection",
    layout="centered"
)

st.title("🌱 AI-Based Crop Pest Detection & Pesticide Recommendation")
st.write(
    "Upload a tomato leaf image to detect the disease and get the recommended pesticide."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            result = predict(temp_path)

        st.success("Prediction Complete!")

        st.markdown(f"### 🦠 Disease Detected: **{result['disease']}**")
        st.markdown(f"**Confidence:** {result['confidence']}%")
        st.markdown(f"### 🧪 Recommended Chemical")
        st.info(result["recommended_chemical"])
        st.markdown(f"**Note:** {result['note']}")
