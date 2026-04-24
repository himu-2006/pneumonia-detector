import json
import streamlit as st
import matplotlib.pyplot as plt

from train import (
    run_training,
    is_xray,
    calibrate,
    load_pneumonia_model,
    load_temperature,
    predict,
    create_pdf,
)

st.set_page_config(page_title="Radiology AI", layout="wide")



st.markdown("<h1 style='text-align:center;'>🏥 Radiology AI System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based Chest X-ray Analysis</p>", unsafe_allow_html=True)


st.sidebar.header("🧑‍⚕️ Patient Info")
name      = st.sidebar.text_input("Patient Name", "John Doe")
age       = st.sidebar.number_input("Age", 1, 100, 45)
gender    = st.sidebar.selectbox("Gender", ["Male", "Female"])
physician = st.sidebar.text_input("Physician Name", "Dr. Smith")
st.sidebar.markdown("---")
st.sidebar.write("⚠️ Educational use only")

st.sidebar.markdown("---")


# ================================================================
#  LOAD MODEL & TEMPERATURE
# ================================================================

model, model_loaded = load_pneumonia_model()



TEMPERATURE = load_temperature()
if TEMPERATURE > 5.0:
    st.warning(
        f"⚠️ Temperature T={TEMPERATURE:.1f} is very high — model was overconfident. "
        "Consider retraining with class weights."
    )



col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 X-ray Viewer")
    uploaded = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])
    raw_prob = None

    if uploaded and model_loaded:
        st.image(uploaded, use_container_width=True)

        uploaded.seek(0)
        valid, reason = is_xray(uploaded)

        if not valid:
            st.error(f"🚫 Invalid Image: {reason}")
            st.info("💡 Please upload a real chest X-ray image (grayscale, dark background).")
            uploaded = None  # block diagnosis panel
        else:
            st.success("✅ Image validated as X-ray")
            with st.spinner("🔍 Analyzing X-ray..."):
                raw_prob = predict(model, uploaded)

with col2:
    st.subheader("🧾 Diagnosis Report")

    if uploaded and model_loaded and raw_prob is not None:
        pneumonia_prob = calibrate(raw_prob, TEMPERATURE)
        normal_prob    = 1 - pneumonia_prob

        
        if pneumonia_prob > 0.6:
            result = "Pneumonia Detected"
            st.error("🫁 Pneumonia Detected")
        elif pneumonia_prob < 0.4:
            result = "Normal"
            st.success("✅ Normal")
        else:
            result = "Uncertain"
            st.warning("⚠️ Uncertain — Needs further review")

    
        st.markdown("### 📊 Probability Scores")
        st.write(f"🟢 Normal: {normal_prob * 100:.2f}%")
        st.write(f"🔴 Pneumonia: {pneumonia_prob * 100:.2f}%")
        st.write(f"🌡️ Calibration Temperature: **T = {TEMPERATURE:.2f}**")
        st.progress(float(pneumonia_prob))

        # Confidence level (distance from 0.5 = uncertain)
        dist = abs(pneumonia_prob - 0.5)
        if dist > 0.35:
            st.write("Confidence Level: 🔴 High")
        elif dist > 0.2:
            st.write("Confidence Level: 🟡 Moderate")
        else:
            st.write("Confidence Level: 🟢 Low")

        st.markdown("---")

    
        st.markdown("### 🩺 Recommendations")
        if pneumonia_prob > 0.7:
            st.write("- ⚠️ High likelihood of pneumonia\n- Immediate medical consultation required\n- Additional diagnostic tests recommended")
        elif pneumonia_prob > 0.5:
            st.write("- Possible infection detected\n- Monitor symptoms closely\n- Consult physician if needed")
        elif pneumonia_prob < 0.4:
            st.write("- No infection detected\n- Maintain healthy lifestyle")
        else:
            st.write("- ⚠️ Uncertain result\n- Recommend further testing")

        st.markdown("---")

        st.markdown("### 👨‍⚕️ Physician")
        st.write(physician)

        st.markdown("### 📋 Patient Summary")
        st.write(f"Name: {name}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")
        st.markdown("⚠️ AI-generated result. Not a medical diagnosis.")

        st.markdown("---")

       
        pdf_path = create_pdf(name, age, gender, physician, normal_prob, pneumonia_prob, result)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")



st.markdown("---")
st.subheader("📊 Model Performance")

try:
    with open("history.json", "r") as f:
        history = json.load(f)

    fig1, ax1 = plt.subplots()
    ax1.plot(history['accuracy'],     label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(history['loss'],     label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    st.pyplot(fig2)

except Exception:
    st.warning("⚠️ No training history found. Run training first.")



st.markdown("---")
st.markdown("<p style='text-align:center;'>AI Pneumonia Detection System</p>", unsafe_allow_html=True)
