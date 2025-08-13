# app.py — Streamlit NAFLD Lifestyle Risk Predictor
import json
import joblib
import pandas as pd
import streamlit as st

# -------------------- Page config --------------------
st.set_page_config(
    page_title="NAFLD Lifestyle Risk Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("NAFLD Lifestyle Risk Predictor")
st.caption(
    "This app estimates your likelihood of NAFLD using lifestyle and demographic inputs. "
    "It is not a medical diagnosis and should not replace professional medical advice."
)

# -------------------- Load artifacts --------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("pipeline.pkl")

@st.cache_resource
def load_features():
    with open("feature_order.json") as f:
        feats = json.load(f)
    # sanitize: strip blanks / dups, keep order
    clean, seen = [], set()
    for x in feats:
        name = str(x).strip()
        if name and name not in seen:
            clean.append(name); seen.add(name)
    return clean

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Could not load pipeline.pkl: {e}")
    st.stop()

try:
    FEATURES = load_features()
except Exception as e:
    st.error(f"Could not read feature_order.json: {e}")
    st.stop()

# -------------------- UI dictionaries --------------------
CATEGORICAL_CHOICES = {
    "Gender": ["Male", "Female"],
    "Race/Ethnicity": ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
    "Smoking status": ["Never", "Former", "Current"],
    "Sleep Disorder Status": ["No", "Yes"],
}

RANGE_PRESETS = {
    # Demographics / socioeconomics
    "Age in years": dict(min_value=18, max_value=85, value=40, step=1.0),
    "Family income ratio": dict(min_value=0.0, max_value=5.0, value=2.0, step=0.1),

    # Sleep / activity
    "Sleep duration (hours/day)": dict(min_value=3.0, max_value=12.0, value=7.0, step=0.5),
    "Work schedule duration (hours)": dict(min_value=0.0, max_value=16.0, value=8.0, step=0.5),
    "Physical activity (minutes/day)": dict(min_value=0, max_value=300, value=30, step=5),

    # Anthropometrics
    "BMI": dict(min_value=15.0, max_value=50.0, value=28.0, step=0.1),

    # Alcohol
    "Number of days drank in the past year": dict(min_value=0, max_value=365, value=0, step=1),
    "Alcohol consumption (days/week)": dict(min_value=0.0, max_value=7.0, value=0.0, step=0.5),
    "Alcohol drinks per day": dict(min_value=0.0, max_value=10.0, value=0.0, step=0.5),
    "Max number of drinks on any single day": dict(min_value=0, max_value=20, value=0, step=1),
    "Alcohol intake frequency (drinks/day)": dict(min_value=0.0, max_value=10.0, value=0.0, step=0.1),

    # Diet
    "Total calorie intake (kcal)": dict(min_value=800, max_value=5000, value=2200, step=50),
    "Total fat (g)": dict(min_value=10, max_value=200, value=70, step=1),
    "Saturated fat (g)": dict(min_value=0, max_value=80, value=25, step=1),
    "Added sugar (g)": dict(min_value=0, max_value=150, value=30, step=1),
    "Dietary fiber (g)": dict(min_value=0, max_value=80, value=20, step=1),
    "Fruit/veg servings per day": dict(min_value=0.0, max_value=10.0, value=3.0, step=0.5),
    "Sugary drinks per week": dict(min_value=0.0, max_value=50.0, value=0.0, step=1.0),
}

HELP_TEXT = {
    "Family income ratio": "NHANES Poverty Income Ratio (0–~5, higher ≈ higher income).",
    "Alcohol intake frequency (drinks/day)": "Average daily number of drinks.",
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
    help_txt = HELP_TEXT.get(name)
    if name in CATEGORICAL_CHOICES:
        return st.selectbox(name, CATEGORICAL_CHOICES[name], help=help_txt, key=name)
    if name in RANGE_PRESETS:
        p = RANGE_PRESETS[name]
        # slider for moderate ranges, number_input otherwise
        if (p["max_value"] - p["min_value"]) <= 200:
            return st.slider(name, help=help_txt, **p)
        else:
            return st.number_input(name, help=help_txt, **p)
    # fallback numeric if we don't have a preset
    return st.number_input(name, value=0.0, step=0.1, help=help_txt, key=name)

# -------------------- Layout & inputs --------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Inputs")
    values_dict = {}

    # Render by groups, only for features present in FEATURES (order preserved)
    for group_name, group_feats in GROUPS.items():
        feats = [f for f in FEATURES if f in group_feats]
        if feats:
            with st.expander(group_name, expanded=True):
                cols = st.columns(2)
                for i, fname in enumerate(feats):
                    with cols[i % 2]:
                        values_dict[fname] = widget_for(fname)

    # Any remaining features not listed above
    remaining = [f for f in FEATURES if all(f not in g for g in GROUPS.values())]
    if remaining:
        with st.expander("Other", expanded=False):
            cols = st.columns(2)
            for i, fname in enumerate(remaining):
                with cols[i % 2]:
                    values_dict[fname] = widget_for(fname)

# Keep exact model order
values = [values_dict[f] for f in FEATURES]

# -------------------- Predict --------------------
with right:
    st.subheader("Prediction")
    go = st.button("Predict", type="primary", use_container_width=True)
    prog = st.empty()
    msg_box = st.empty()
    metric_box = st.empty()

    if go:
        try:
            X = pd.DataFrame([values], columns=FEATURES)
            proba = float(pipeline.predict_proba(X)[:, 1][0])  # 0..1
            pct = round(proba * 100, 1)
            band = "High risk" if pct >= 50 else "Low/moderate risk"

            prog.progress(proba, text="Predicted probability")
            msg_box.success(f"Predicted NAFLD risk: **{pct}%** — **{band}**")
            metric_box.metric("Risk (%)", pct)
        except Exception as e:
            msg_box.error(f"Inference error: {e}")
