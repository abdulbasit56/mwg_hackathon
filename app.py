import joblib
import pandas as pd
import streamlit as st
from gemini_helper import get_crop_advice, analyze_leaf

st.set_page_config(page_title="Smart Crop Advisor", page_icon="🌾", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("model/crop_model.pkl")


bundle = load_bundle()
model, FEATURES, stats = bundle["model"], bundle["features"], bundle["stats"]

LABELS = {
    "N": "Nitrogen (N), kg/ha",
    "P": "Phosphorus (P), kg/ha",
    "K": "Potassium (K), kg/ha",
    "temperature": "Temperature, °C",
    "humidity": "Humidity, %",
    "ph": "Soil pH",
    "rainfall": "Rainfall, mm",
}

st.title("Smart Crop & Plant Health Advisor")
st.caption("Helping farmers choose the right crop and spot plant problems early. SDG 2: Zero Hunger.")
language = st.sidebar.radio("Advice language", ["English", "Urdu"])
st.sidebar.info("AI suggestions are guidance only. Confirm with a local agriculture extension officer.")

tab1, tab2, tab3 = st.tabs(["Crop Advisor", "Leaf Check", "Model & About"])

with tab1:
    st.subheader("Enter soil and weather readings")
    cols = st.columns(2)
    inputs = {}
    for i, f in enumerate(FEATURES):
        lo, hi, med = float(stats[f]["min"]), float(stats[f]["max"]), float(stats[f]["median"])
        with cols[i % 2]:
            inputs[f] = st.slider(LABELS.get(f, f), lo, hi, med)

    if st.button("Recommend crop", type="primary"):
        X = pd.DataFrame([inputs])[FEATURES]
        probs = model.predict_proba(X)[0]
        top_idx = probs.argsort()[::-1][:3]
        top3 = [(model.classes_[i], round(float(probs[i]), 2)) for i in top_idx]
        st.session_state["top3"] = top3
        st.session_state["inputs"] = inputs
        st.session_state.pop("advice", None)

    if "top3" in st.session_state:
        top3 = st.session_state["top3"]
        st.success(f"Best crop: **{top3[0][0].title()}** ({top3[0][1]:.0%} confidence)")
        chart_df = pd.DataFrame(top3, columns=["Crop", "Probability"]).set_index("Crop")
        st.bar_chart(chart_df)

        if st.button("Get AI advice"):
            with st.spinner("Asking Gemini..."):
                st.session_state["advice"] = get_crop_advice(
                    st.session_state["inputs"], top3, language
                )
        if "advice" in st.session_state:
            st.markdown(st.session_state["advice"])

with tab2:
    st.subheader("Leaf health check")
    st.warning("This is an AI suggestion, not a confirmed diagnosis. Verify with an agriculture expert.")
    up = st.file_uploader("Upload a clear photo of one leaf", type=["jpg", "jpeg", "png"])
    if up is not None:
        st.image(up, width=300)
        if st.button("Analyze leaf"):
            with st.spinner("Gemini is looking at the leaf..."):
                st.markdown(analyze_leaf(up.getvalue(), language))

with tab3:
    st.subheader("How it works")
    st.markdown(
        """
- **Crop recommendation:** a Random Forest model we trained on 2,200 soil and weather samples covering 22 crops.
- **AI advice and leaf check:** Google Gemini API (text and vision).
- **Built with:** Python, scikit-learn, Streamlit, Google Gemini.
- **SDG alignment:** SDG 2 Zero Hunger, SDG 13 Climate Action.
"""
    )
    c1, c2 = st.columns(2)
    c1.image("model/feature_importance.png", caption="Which inputs matter most")
    c2.image("model/confusion_matrix.png", caption="Test-set confusion matrix")