import os, json
import pandas as pd
import joblib
import gradio as gr

# ---------- Load artifacts ----------
pipeline = joblib.load("pipeline.pkl")
with open("feature_order.json") as f:
    FEATURES = [s.strip() for s in json.load(f) if str(s).strip()]

# ---------- Choices & ranges ----------
CATEGORICAL_CHOICES = {
    "Gender": ["Male", "Female"],
    "Race/Ethnicity": ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
    "Smoking status": ["Never", "Former", "Current"],
    "Sleep Disorder Status": ["No", "Yes"],
}

RANGE_PRESETS = {
    "Age in years": dict(minimum=18, maximum=85, value=40, step=1),
    "Family income ratio": dict(minimum=0.0, maximum=5.0, value=2.0, step=0.1),
    "Sleep duration (hours/day)": dict(minimum=3.0, maximum=12.0, value=7.0, step=0.5),
    "Work schedule duration (hours)": dict(minimum=0.0, maximum=16.0, value=8.0, step=0.5),
    "Physical activity (minutes/day)": dict(minimum=0, maximum=300, value=30, step=5),
    "BMI": dict(minimum=15.0, maximum=50.0, value=28.0, step=0.1),
    "Number of days drank in the past year": dict(minimum=0, maximum=365, value=0, step=1),
    "Alcohol consumption (days/week)": dict(minimum=0.0, maximum=7.0, value=0.0, step=0.5),
    "Alcohol drinks per day": dict(minimum=0.0, maximum=10.0, value=0.0, step=0.5),
    "Max number of drinks on any single day": dict(minimum=0, maximum=20, value=0, step=1),
    "Alcohol intake frequency (drinks/day)": dict(minimum=0.0, maximum=10.0, value=0.0, step=0.1),
    "Total calorie intake (kcal)": dict(minimum=800, maximum=5000, value=2200, step=50),
    "Total fat (g)": dict(minimum=10, maximum=200, value=70, step=1),
    "Saturated fat (g)": dict(minimum=0, maximum=80, value=25, step=1),
    "Added sugar (g)": dict(minimum=0, maximum=150, value=30, step=1),
    "Dietary fiber (g)": dict(minimum=0, maximum=80, value=20, step=1),
    "Fruit/veg servings per day": dict(minimum=0.0, maximum=10.0, value=3.0, step=0.5),
    "Sugary drinks per week": dict(minimum=0.0, maximum=50.0, value=0.0, step=1.0),
}

GROUPS = {
    "Demographics": [
        "Gender","Age in years","Race/Ethnicity","Family income ratio","Smoking status"
    ],
    "Sleep & Activity": [
        "Sleep Disorder Status","Sleep duration (hours/day)",
        "Work schedule duration (hours)","Physical activity (minutes/day)"
    ],
    "Anthropometrics": ["BMI"],
    "Alcohol": [
        "Number of days drank in the past year","Alcohol consumption (days/week)",
        "Alcohol drinks per day","Max number of drinks on any single day",
        "Alcohol intake frequency (drinks/day)"
    ],
    "Diet": [
        "Total calorie intake (kcal)","Total fat (g)","Saturated fat (g)",
        "Added sugar (g)","Dietary fiber (g)","Fruit/veg servings per day","Sugary drinks per week"
    ],
}

def widget_for(name: str):
    if name in CATEGORICAL_CHOICES:
        return gr.Dropdown(CATEGORICAL_CHOICES[name], label=name)
    if name in RANGE_PRESETS:
        return gr.Slider(label=name, **RANGE_PRESETS[name])
    return gr.Number(label=name)

def predict_fn(*values):
    X = pd.DataFrame([values], columns=FEATURES)
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    pct = round(proba * 100, 1)
    band = "High risk" if pct >= 50 else "Low/moderate risk"
    return f"**Predicted NAFLD risk: {pct}% — {band}**", pct

theme = gr.themes.Soft()
with gr.Blocks(title="NAFLD Lifestyle Risk Predictor", theme=theme) as demo:
    gr.Markdown(
        "# NAFLD Lifestyle Risk Predictor\n"
        "This app estimates your likelihood of NAFLD using lifestyle and demographic inputs. "
        "It is not a medical diagnosis and should not replace professional medical advice."
    )

    # Render inputs in model order, grouped for layout
    inputs = []
    order_groups = ["Demographics","Sleep & Activity","Anthropometrics","Alcohol","Diet"]
    grouped = {g: [] for g in order_groups}
    other = []
    for name in FEATURES:
        placed = False
        for g, flist in GROUPS.items():
            if name in flist:
                grouped[g].append(name); placed = True; break
        if not placed:
            other.append(name)

    for g in order_groups:
        if grouped[g]:
            with gr.Accordion(g, open=True):
                with gr.Row():
                    half = (len(grouped[g]) + 1)//2
                    with gr.Column():
                        for fname in grouped[g][:half]: inputs.append(widget_for(fname))
                    with gr.Column():
                        for fname in grouped[g][half:]: inputs.append(widget_for(fname))

    if other:
        with gr.Accordion("Other", open=False):
            for fname in other:
                inputs.append(widget_for(fname))

    out_text = gr.Markdown("")
    out_bar  = gr.Slider(0, 100, value=0.0, step=0.1, interactive=False, label="Predicted probability (%)")
    gr.Button("Predict", variant="primary").click(predict_fn, inputs=inputs, outputs=[out_text, out_bar])

if __name__ == "__main__":
    # Works locally and on Hugging Face (Docker)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
