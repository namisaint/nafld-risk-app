# app.py — NAFLD Lifestyle Risk Predictor (Gradio, polished + robust)

import os, json, time
import pandas as pd
import joblib
import gradio as gr

# ---------- (Optional) Mongo logging via HF Secrets ----------
USE_MONGO = bool(os.getenv("MONGODB_URI"))
if USE_MONGO:
    try:
        from pymongo import MongoClient
        mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        db = mongo_client.get_database(os.getenv("MONGODB_DB", "nafld"))
        logs = db.get_collection(os.getenv("MONGODB_COLLECT", "predictions"))
    except Exception as e:
        print("⚠️ Mongo init failed:", e)
        USE_MONGO = False

# ---------- Load artifacts ----------
MODEL_PATH = "pipeline.pkl"
FEATURES_PATH = "feature_order.json"

# Fail fast if files are missing
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("pipeline.pkl not found in app root.")
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("feature_order.json not found in app root.")

pipeline = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

# ---------- UI presets ----------
CATEGORICAL_CHOICES = {
    "Gender": ["Male", "Female"],
    "Race/Ethnicity": ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
    "Smoking status": ["Never", "Former", "Current"],
    "Sleep Disorder Status": ["No", "Yes"],
}

RANGE_PRESETS = {
    # Demographics / lifestyle
    "Age in years": dict(minimum=18, maximum=85, value=40, step=1, label="Age (years)"),
    "BMI": dict(minimum=15.0, maximum=50.0, value=28.0, step=0.1, label="BMI"),
    "Sleep duration (hours/day)": dict(minimum=3.0, maximum=12.0, value=7.0, step=0.5),
    "Work schedule duration (hours)": dict(minimum=0.0, maximum=16.0, value=8.0, step=0.5),
    "Physical activity (minutes/day)": dict(minimum=0, maximum=300, value=30, step=5),

    # Alcohol
    "Alcohol consumption (days/week)": dict(minimum=0.0, maximum=7.0, value=0.0, step=0.5),
    "Alcohol drinks per day": dict(minimum=0.0, maximum=10.0, value=0.0, step=0.5),
    "Number of days drank in the past year": dict(minimum=0, maximum=365, value=0, step=1),
    "Max number of drinks on any single day": dict(minimum=0, maximum=20, value=0, step=1),
    "Alcohol intake frequency (drinks/day)": dict(minimum=0.0, maximum=10.0, value=0.0, step=0.1),

    # Diet
    "Total calorie intake (kcal)": dict(minimum=800, maximum=5000, value=2200, step=50),
    "Total fat (g)": dict(minimum=10, maximum=200, value=70, step=1),
    "Saturated fat (g)": dict(minimum=0, maximum=80, value=25, step=1),
    "Added sugar (g)": dict(minimum=0, maximum=150, value=30, step=1),
    "Dietary fiber (g)": dict(minimum=0, maximum=80, value=20, step=1),
    "Fruit/veg servings per day": dict(minimum=0.0, maximum=10.0, value=3.0, step=0.5),
    "Sugary drinks per week": dict(minimum=0.0, maximum=50.0, value=0.0, step=1.0),
}

def widget_for(name: str):
    """Return a Gradio component suited for this feature."""
    if name in CATEGORICAL_CHOICES:
        return gr.Dropdown(CATEGORICAL_CHOICES[name], label=name)
    if name in RANGE_PRESETS:
        return gr.Slider(**RANGE_PRESETS[name])
    # Fallback for any extra numeric features in your 21:
    return gr.Number(label=name)

def predict_fn(*values):
    """Build a single-row DataFrame in the exact training order and predict."""
    try:
        X = pd.DataFrame([values], columns=FEATURES)
        proba = float(pipeline.predict_proba(X)[:, 1][0])
        label = int(pipeline.predict(X)[0])
        risk_pct = round(proba * 100, 1)
        risk_band = "High risk" if risk_pct >= 50 else "Low/moderate risk"

        # Optional logging
        if USE_MONGO:
            try:
                logs.insert_one({
                    "ts": time.time(),
                    "inputs": dict(zip(FEATURES, values)),
                    "probability": proba,
                    "label": label,
                    "risk_pct": risk_pct,
                    "risk_band": risk_band,
                    "app_version": "v1",
                })
            except Exception as e:
                print("⚠️ Mongo log failed:", e)

        # Nicely formatted result
        return (
            f"**Predicted NAFLD risk: {risk_pct}% — {risk_band}**",
            proba
        )
    except Exception as e:
        return (f"❌ Inference error: {e}", None)

# ---------- Build the polished UI ----------
with gr.Blocks(title="NAFLD Lifestyle Risk Predictor") as demo:
    gr.Markdown("# NAFLD Lifestyle Risk Predictor")
    gr.Markdown(
        "Estimate NAFLD probability from lifestyle & demographic inputs.  \n"
        "**Note:** Educational use only — not medical advice."
    )

    # Inputs in a clean two-column layout
    inputs = []
    half = (len(FEATURES) + 1) // 2
    with gr.Row():
        with gr.Column():
            for name in FEATURES[:half]:
                inputs.append(widget_for(name))
        with gr.Column():
            for name in FEATURES[half:]:
                inputs.append(widget_for(name))

    # Outputs
    out_text = gr.Markdown("")
    out_bar = gr.Slider(0.0, 1.0, value=0.0, step=0.001, interactive=False, label="Predicted probability")

    # Action
    gr.Button("Predict", variant="primary").click(
        predict_fn,
        inputs=inputs,
        outputs=[out_text, out_bar]
    )

if __name__ == "__main__":
    demo.launch()
