import os, json, time
import numpy as np
import pandas as pd
import gradio as gr
import joblib

# ---------- Optional Mongo logging ----------
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

# ---------- Load model pipeline + schema ----------
MODEL_PATH = "pipeline.pkl"
FEATURES_PATH = "feature_order.json"  # optional but recommended

_err = None
pipeline = None
FEATURES = None

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    _err = f"Could not load model pipeline at '{MODEL_PATH}': {e}"

if os.path.exists(FEATURES_PATH):
    try:
        with open(FEATURES_PATH, "r") as f:
            FEATURES = json.load(f)
    except Exception as e:
        print("⚠️ feature_order.json could not be read:", e)

# If you don't provide feature_order.json, we define a sensible default set.
# *** Make sure these EXACT names match what your pipeline expects. ***
if FEATURES is None:
    FEATURES = [
        "Gender", "Age in years", "Race/Ethnicity", "Family income ratio",
        "Smoking status", "Sleep Disorder Status", "Sleep duration (hours/day)",
        "Work schedule duration (hours)", "Physical activity (minutes/day)", "BMI",
        "Alcohol consumption (days/week)", "Alcohol drinks per day",
        "Number of days drank in the past year", "Max number of drinks on any single day",
        "Alcohol intake frequency (drinks/day)", "Total calorie intake (kcal)",
        "Total fat (g)", "Saturated fat (g)", "Added sugar (g)", "Dietary fiber (g)",
        "Fruit/veg servings per day", "Sugary drinks per week"
    ]

# ---------- Input widgets ----------
# Map categorical fields to dropdown choices (must match your training preprocessing!)
gender_choices = ["Male", "Female"]
race_choices = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
smoke_choices = ["Never", "Former", "Current"]
sleep_disorder_choices = ["No", "Yes"]

# Build the Gradio inputs in the same order as FEATURES
# For any cat feature, we use Dropdown; numeric → Slider/Number
INPUTS_SPEC = {
    "Gender": gr.Dropdown(gender_choices, label="Gender"),
    "Age in years": gr.Slider(18, 85, 40, step=1, label="Age (years)"),
    "Race/Ethnicity": gr.Dropdown(race_choices, label="Race/Ethnicity"),
    "Family income ratio": gr.Slider(0.0, 5.0, 2.0, step=0.1, label="Family Poverty Income Ratio"),
    "Smoking status": gr.Dropdown(smoke_choices, label="Smoking status"),
    "Sleep Disorder Status": gr.Dropdown(sleep_disorder_choices, label="Sleep disorder diagnosed?"),
    "Sleep duration (hours/day)": gr.Slider(3.0, 12.0, 7.0, step=0.5, label="Sleep duration (h/day)"),
    "Work schedule duration (hours)": gr.Slider(0.0, 16.0, 8.0, step=0.5, label="Work hours/day"),
    "Physical activity (minutes/day)": gr.Slider(0, 300, 30, step=5, label="Physical activity (min/day)"),
    "BMI": gr.Slider(15.0, 50.0, 28.0, step=0.1, label="BMI"),
    "Alcohol consumption (days/week)": gr.Slider(0.0, 7.0, 0.0, step=0.5, label="Alcohol days/week"),
    "Alcohol drinks per day": gr.Slider(0.0, 10.0, 0.0, step=0.5, label="Drinks per drinking day"),
    "Number of days drank in the past year": gr.Slider(0, 365, 0, step=1, label="Days drank in past year"),
    "Max number of drinks on any single day": gr.Slider(0, 20, 0, step=1, label="Max drinks in a day"),
    "Alcohol intake frequency (drinks/day)": gr.Slider(0.0, 10.0, 0.0, step=0.1, label="Avg drinks/day"),
    "Total calorie intake (kcal)": gr.Slider(800, 4000, 2200, step=50, label="Calories (kcal)"),
    "Total fat (g)": gr.Slider(10, 200, 70, step=1, label="Total fat (g)"),
    "Saturated fat (g)": gr.Slider(0, 80, 25, step=1, label="Saturated fat (g)"),
    "Added sugar (g)": gr.Slider(0, 150, 30, step=1, label="Added sugar (g)"),
    "Dietary fiber (g)": gr.Slider(0, 80, 20, step=1, label="Fiber (g)"),
    "Fruit/veg servings per day": gr.Slider(0.0, 10.0, 3.0, step=0.5, label="Fruit/veg servings/day"),
    "Sugary drinks per week": gr.Slider(0.0, 50.0, 0.0, step=1.0, label="Sugary drinks/week"),
}

gr_inputs = [INPUTS_SPEC[f] for f in FEATURES]

def make_df_from_inputs(values):
    """Create a one-row DataFrame with columns in exact FEATURES order."""
    return pd.DataFrame([values], columns=FEATURES)

def predict_fn(*values):
    if _err:
        return f"❌ Model not loaded: {_err}", None
    try:
        X = make_df_from_inputs(list(values))
        proba = float(pipeline.predict_proba(X)[:, 1][0])
        label = int(pipeline.predict(X)[0])
        risk_pct = round(proba * 100, 1)
        risk_band = "High risk" if risk_pct >= 50 else "Low/moderate risk"

        # Optional Mongo logging
        if USE_MONGO:
            try:
                logs.insert_one({
                    "ts": time.time(),
                    "inputs": {k: v for k, v in zip(FEATURES, values)},
                    "probability": proba,
                    "label": label,
                    "risk_pct": risk_pct,
                    "risk_band": risk_band,
                    "app_version": "v1"
                })
            except Exception as e:
                print("⚠️ Mongo log failed:", e)

        msg = f"Predicted NAFLD risk: **{risk_pct}%** — **{risk_band}**"
        return msg, proba
    except Exception as e:
        return f"❌ Inference error: {e}", None

with gr.Blocks(title="NAFLD Lifestyle Risk Predictor") as demo:
    gr.Markdown("# NAFLD Lifestyle Risk Predictor")
    gr.Markdown(
        "Enter your lifestyle & demographic info below. "
        "This tool estimates the probability of fatty liver (for education only; not medical advice)."
    )

    with gr.Row():
        with gr.Column():
            out_text = gr.Markdown("")
            out_bar = gr.Slider(0.0, 1.0, value=0.0, step=0.001, interactive=False, label="Predicted probability")

    with gr.Row():
        form = gr.Column()
        inputs = []
        for w in gr_inputs:
            inputs.append(form.append(w))

    btn = gr.Button("Predict", variant="primary")

    btn.click(fn=predict_fn, inputs=gr_inputs, outputs=[out_text, out_bar])

if __name__ == "__main__":
    demo.launch()
