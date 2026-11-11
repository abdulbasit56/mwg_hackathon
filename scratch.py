import streamlit as st
from gemini_helper import get_crop_advice
sample = {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph":
6.5, "rainfall": 203}
st.write(get_crop_advice(sample, [("rice", 0.82), ("jute", 0.10), ("maize",
0.05)], "English"))
st.write(get_crop_advice(sample, [("rice", 0.82), ("jute", 0.10), ("maize",
0.05)], "Urdu"))
