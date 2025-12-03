import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from model_loader import load_model
from predictor import predict_stage
from gradcam import generate_gradcam
from clinical_report import generate_clinical_report


# -----------------------------
# Image Enhancement
# -----------------------------
def enhance_image(image_bgr: np.ndarray, alpha: float = 1.2, beta: int = 15) -> np.ndarray:
    enhanced = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    return enhanced


# -----------------------------
# Cached Model Loader
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_retinal_model():
    return load_model()


# -----------------------------
# Main App
# -----------------------------
def main():

    st.set_page_config(
        page_title="Diabetic Retinopathy Severity Classification",
        layout="wide",
    )

    st.title(" Diabetic Retinopathy Severity Classification & Clinical Reporting")
    st.markdown("""
Analyze retinal fundus images and generate **DR severity**, **Grad-CAM**, and a **clinical-style report**.
""")

    # --------------------------------------------
    # Initialize session-state values only ONCE
    # --------------------------------------------
    if "alpha" not in st.session_state:
        st.session_state.alpha = 1.2
    if "beta" not in st.session_state:
        st.session_state.beta = 15
    if "uploaded_flag" not in st.session_state:
        st.session_state.uploaded_flag = False

    # --------------------------------------------
    # Sidebar — restored including About section
    # --------------------------------------------
    with st.sidebar:

        st.header("About")
        st.write(
            """
**Project:** Diabetic Retinopathy Severity Classification  
**Output:** 5 severity stages, confidence score, Grad-CAM visualization, and a generated clinical report.
"""
        )

        st.markdown("---")
        st.subheader("Image Enhancement Settings")
        st.caption("Adjust before analysis to improve image visibility.")

        # Reset button
        if st.button("Reset Enhancements"):
            st.session_state.alpha = 1.2
            st.session_state.beta = 15
            st.rerun()

        # Sliders
        alpha = st.slider("Contrast (α)", 1.0, 3.0, st.session_state.alpha, 0.1, key="alpha_slider")
        beta = st.slider("Brightness (β)", 0, 50, st.session_state.beta, 1, key="beta_slider")

        # Update values
        st.session_state.alpha = alpha
        st.session_state.beta = beta

    # --------------------------------------------
    # Load Model
    # --------------------------------------------
    with st.spinner("Loading retinal model..."):
        model = load_retinal_model()

    st.markdown("### Upload a Retinal Fundus Image")
    uploaded_file = st.file_uploader("Upload JPG / PNG retinal scan.", type=["jpg", "jpeg", "png"])

    # --------------------------------------------
    # Auto-reset sliders on new upload
    # --------------------------------------------
    if uploaded_file is not None:
        if not st.session_state.uploaded_flag:
            st.session_state.alpha = 1.2
            st.session_state.beta = 15
            st.session_state.uploaded_flag = True
            st.rerun()
    else:
        st.session_state.uploaded_flag = False

    # --------------------------------------------
    # Process Image
    # --------------------------------------------
    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Image could not be read.")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image_rgb, use_column_width=True)

        # Enhanced image
        enhanced_bgr = enhance_image(image_bgr, alpha=st.session_state.alpha, beta=st.session_state.beta)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Enhanced")
            st.image(enhanced_rgb, use_column_width=True)

        st.markdown("---")
        st.subheader("Run Analysis")

        if st.button("Analyze Retinal Image"):
            with st.spinner("Predicting severity..."):
                severity_label, confidence = predict_stage(model, enhanced_rgb)

            st.success("Prediction complete.")
            st.write(f"**Predicted Severity:** `{severity_label}`")
            st.write(f"**Confidence:** `{confidence:.4f}`")

            # Grad-CAM
            with st.spinner("Generating Grad-CAM..."):
                heatmap_rgb = generate_gradcam(model, enhanced_rgb)

            colA, colB = st.columns(2)
            with colA:
                st.image(enhanced_rgb, caption="Enhanced Input")
            with colB:
                st.image(heatmap_rgb, caption="Grad-CAM Highlighted Regions")

            # Clinical Report
            report_text = generate_clinical_report(severity_label, confidence)

            st.markdown("### Clinical Report")
            st.text(report_text)

            st.download_button(
                "Download Report (.txt)",
                report_text.encode("utf-8"),
                "dr_clinical_report.txt",
                "text/plain",
            )

    else:
        st.info("Upload a retinal fundus image to start.")


if __name__ == "__main__":
    main()
