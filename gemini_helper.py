import io
import os
import tomllib
from pathlib import Path

import streamlit as st
from google import genai
from PIL import Image
MODEL = "gemini-3.6-flash" # use the exact name shown in AI Studio

# secrets.toml next to this file, so it works whatever folder `streamlit run` is started from
_LOCAL_SECRETS = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


class MissingKeyError(Exception):
    pass


def _get_api_key() -> str:
    # 1. Streamlit secrets (works on Streamlit Cloud and when run from the project folder)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # 2. secrets.toml next to this file
    if _LOCAL_SECRETS.exists():
        with open(_LOCAL_SECRETS, "rb") as f:
            key = tomllib.load(f).get("GEMINI_API_KEY")
        if key:
            return key
    # 3. Environment variable
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    raise MissingKeyError("GEMINI_API_KEY not found")


@st.cache_resource
def get_client():
    return genai.Client(api_key=_get_api_key())


def _friendly_error(e: Exception) -> str:
    print(f"[gemini_helper] {type(e).__name__}: {e}")  # full error in the terminal for debugging
    msg = str(e).lower()
    if isinstance(e, MissingKeyError):
        return "AI advice unavailable: GEMINI_API_KEY is missing from .streamlit/secrets.toml."
    if "api key" in msg or "401" in msg or "403" in msg or "permission" in msg:
        return "AI advice unavailable: the Gemini API key was rejected. Check that it is valid."
    if "404" in msg or "not found" in msg:
        return f"AI advice unavailable: model '{MODEL}' was not found. Check the model name."
    if "429" in msg or "quota" in msg or "rate" in msg:
        return "AI advice is busy right now (rate limit reached). Please try again in a minute."
    return f"AI advice is temporarily unavailable ({type(e).__name__}). Please try again."


@st.cache_data(show_spinner=False)
def _ask_text(prompt: str) -> str:
    return get_client().models.generate_content(model=MODEL,
    contents=prompt).text


@st.cache_data(show_spinner=False)
def _ask_image(image_bytes: bytes, prompt: str) -> str:

    img = Image.open(io.BytesIO(image_bytes))
    return get_client().models.generate_content(model=MODEL, contents=[img,
    prompt]).text

def get_crop_advice(inputs: dict, top3: list, language: str) -> str:

    prompt = f"""You are an agricultural advisor helping small farmers in
    Pakistan.
    Soil and weather readings: {inputs}
    A machine learning model recommended these crops (crop, probability): {top3}
    Write in {language}, in simple words a farmer understands:
    1. Why the top crop suits these conditions (2-3 sentences, refer to the
    actual readings).
    2. Three practical tips for growing it (soil, water, fertilizer).
    3. One sentence on when the second or third option might be better.
    Keep it under 200 words. Do not invent statistics.
    End with: "Please confirm with your local agriculture extension officer."
    """
    try:
        return _ask_text(prompt)
    except Exception as e:
        return _friendly_error(e)

def analyze_leaf(image_bytes: bytes, language: str) -> str:
    
    prompt = f"""You are a plant health assistant. Look at this photo.
    First, say whether it shows a plant leaf. If it does not, or it is too
    blurry, say so and stop.
    If it is a leaf, answer in {language} with simple words:
    - Likely plant (if recognizable)
    - Visible signs (spots, yellowing, holes, mold, etc.)
    - Possible causes (disease, pest, or nutrient deficiency), described as
    possibilities, not certainties
    - 2-3 safe next steps
    Keep it under 150 words. Make clear this is not a confirmed diagnosis.
    """
    try:
        return _ask_image(image_bytes, prompt)
    except Exception as e:
        return _friendly_error(e)