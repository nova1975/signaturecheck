import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SignatureCheck - Nova", page_icon="✍️", layout="centered")

# Header
st.title("✍️ SignatureCheck")
st.subheader("By Nova")
st.markdown("---")

# Fungsi untuk ekstrak fitur tanda tangan
def extract_features(image):
    image = cv2.resize(image, (200, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features.reshape(1, -1)

# Upload file
uploaded_file1 = st.file_uploader("Upload Tanda Tangan Asli", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Upload Tanda Tangan Yang Ingin Diperiksa", type=["png", "jpg", "jpeg"])

if uploaded_file1 and uploaded_file2:
    file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
    file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
    image1 = cv2.imdecode(file_bytes1, 1)
    image2 = cv2.imdecode(file_bytes2, 1)

    st.image([image1, image2], caption=["Asli", "Dibandingkan"], width=300)

    feat1 = extract_features(image1)
    feat2 = extract_features(image2)

    similarity = cosine_similarity(feat1, feat2)[0][0]
    similarity_percent = similarity * 100

    st.markdown(f"### 🔎 Hasil Kemiripan: **{similarity_percent:.2f}%**")

    if similarity_percent > 85:
        st.success("✅ Tanda tangan cocok!")
    else:
        st.error("❌ Tanda tangan tidak cocok.")

# Footer
st.markdown("---")
st.caption("SignatureCheck © 2025 by Nova")