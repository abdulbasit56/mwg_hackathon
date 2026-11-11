# 🌾 Smart Crop & Plant Health Advisor

A Streamlit web app that helps small farmers in Pakistan choose the right crop for their soil and weather, and spot plant health problems early. Advice is available in **English and Urdu**.

Aligned with **SDG 2: Zero Hunger** and **SDG 13: Climate Action**.

## Features

- **Crop Advisor**: enter soil nutrients (N, P, K), pH, temperature, humidity and rainfall. A Random Forest model recommends the top 3 crops with confidence scores.
- **AI advice**: Google Gemini explains *why* the top crop suits the readings and gives three practical growing tips (soil, water, fertilizer), in simple language.
- **Leaf Check**: upload a photo of a leaf and Gemini Vision describes visible symptoms, possible causes and safe next steps.
- **Model & About**: feature importance and confusion matrix from the trained model.

> AI suggestions are guidance only, not a confirmed diagnosis. Farmers should confirm with a local agriculture extension officer.

## How it works

| Part | Technology |
|------|-----------|
| Crop recommendation | scikit-learn `RandomForestClassifier` (200 trees) |
| Training data | `Crop_recommendation.csv`: 2,200 samples, 22 crops (100 each) |
| AI advice & leaf analysis | Google Gemini API (`google-genai`) |
| Web app | Streamlit |

**Model performance:** about **99.5% accuracy** on a held-out 20% test set (stratified split, `random_state=42`).

## Project structure

```
mwg_hackathon/
├── app.py                  # Streamlit app (3 tabs)
├── gemini_helper.py        # Gemini calls for crop advice and leaf analysis
├── train_model.ipynb       # Data exploration, training, evaluation, export
├── scratch.py              # Quick manual test of the Gemini helper
├── requirements.txt
├── data/
│   └── Crop_recommendation.csv
├── model/
│   ├── crop_model.pkl          # Trained model + feature list + slider ranges
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── .streamlit/
    └── secrets.toml        # Your Gemini API key (not committed)
```

## Setup

**1. Clone and install**

```bash
git clone https://github.com/abdulbasit56/mwg_hackathon.git
cd mwg_hackathon
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**2. Add your Gemini API key**

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), then create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

This file is listed in `.gitignore`. Never commit it.

**3. Run the app**

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Retraining the model

Open `train_model.ipynb` and run all cells. It will:

1. Load and inspect `data/Crop_recommendation.csv`
2. Train the Random Forest on an 80/20 stratified split
3. Save the confusion matrix and feature importance charts to `model/`
4. Save `model/crop_model.pkl`, which bundles the model, feature names and min/max/median stats used for the app's sliders

Keep the scikit-learn version in `requirements.txt` the same as the one used for training. A `.pkl` saved with one version may not load in another.

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (make sure `model/crop_model.pkl` is included).
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) with `app.py` as the entry point.
3. In **Settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `StreamlitSecretNotFoundError` | Create `.streamlit/secrets.toml` with `GEMINI_API_KEY` (see Setup). |
| "No runtime found" warnings | Start the app with `streamlit run app.py`, not `python app.py`. |
| "model ... was not found" in the app | Update `MODEL` in `gemini_helper.py` to a model name listed in AI Studio. |
| Error loading `crop_model.pkl` | Install the scikit-learn version pinned in `requirements.txt`, or retrain the model. |

## Dataset

Crop Recommendation dataset: soil nutrients (N, P, K), temperature, humidity, pH and rainfall, labelled with the most suitable of 22 crops.
