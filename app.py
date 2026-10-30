from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spam_classifier_model.joblib"

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message below. The model predicts whether it is **Ham** (normal) or **Spam**.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.warning("The trained model is missing. Run `python train_model.py` first.")
    st.stop()

message = st.text_area("Enter your message", placeholder="Example: Hey, are we still meeting at 5?", height=140)

if st.button("Check message", type="primary", use_container_width=True):
    if not message.strip():
        st.error("Please enter a message first.")
    else:
        spam_probability = float(model.predict_proba([message])[0][1])
        prediction = int(model.predict([message])[0])
        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: HAM (Not spam)")
        st.metric("Spam probability", f"{spam_probability * 100:.1f}%")
        st.progress(spam_probability)
        st.caption("Probability is the model's estimated score, not a guarantee. Review suspicious messages carefully.")
