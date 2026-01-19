import streamlit as st
from predict import predict

st.set_page_config(page_title="Crop Pest Detection AI", layout="centered")

st.title("🌱 Crop Pest Detection & Pesticide Recommendation")
st.write("Upload a tomato leaf image to detect disease and get recommendations.")

st.warning(
    "⚠️ Disclaimer: This system is for educational purposes only. "
    "Always consult an agricultural expert before applying chemicals."
)

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    result = predict("temp.jpg")

    st.subheader("Prediction Result")
    st.write(f"**Disease:** {result['disease']}")
    st.write(f"**Confidence:** {result['confidence']}%")
    st.write(f"**Recommended Chemical:** {result['recommended_chemical']}")
    st.write(f"**Note:** {result['note']}")
