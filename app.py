# app.py — NAFLD Risk Self-Screening (self-contained, no pickle / sklearn)
# Minimal dependencies: just streamlit. Everything else is plain Python.

import math
import streamlit as st

st.set_page_config(page_title="NAFLD Risk Self-Screening Tool", page_icon="🧪", layout="wide")
st.title("NAFLD Risk Self-Screening Tool")
st.write("Enter your data below to receive a **non-diagnostic** risk estimate. No data is stored.")

# --- Choices (match your previous UI) ---
CHOICES = {
    "RIAGENDR": ["Male","Female"],
    "RIDRETH3": [
        "Mexican American","Other Hispanic","Non-Hispanic White",
        "Non-Hispanic Black","Non-Hispanic Asian","Other/Multi"
    ],
    "ALQ111": ["Yes","No"],
    "ALQ151": ["Yes","No"],
    "SLQ120": ["Yes","No"],
    "SLQ050": ["Never","Rarely","Sometimes","Often","Almost always"],
    "Is_Smoker_Cat": ["Never","Former","Current"],
}

# --- Helper to compute a transparent, rule-based risk score (no ML) ---
def compute_risk_and_contribs(
    RIAGENDR, RIDRETH3, RIDAGEYR, INDFMPIR,
    ALQ111, ALQ142, Is_Smoker_Cat, ALQ121, ALQ170, ALQ151,
    SLQ050, SLD012, SLQ120,
    DR1TKCAL, DR1TPROT, DR1TCARB, DR1TSUGR, DR1TFIBE, DR1TTFAT,
    PAQ620, BMXBMI
):
    contribs = []

    # Intercept keeps baseline low
    s = -4.0

    # Age (centered at 45): +0.03 per year
    age_c = (RIDAGEYR - 45) * 0.03
    s += age_c; contribs.append(("Age", age_c))

    # BMI (centered at 25): +0.12 per BMI point
    bmi_c = (BMXBMI - 25.0) * 0.12
    s += bmi_c; contribs.append(("BMI", bmi_c))

    # Alcohol
    drinks_day_c = float(ALQ142) * 0.15
    s += drinks_day_c; contribs.append(("Drinks per drinking day", drinks_day_c))

    binge_days_c = float(ALQ170) * 0.01
    s += binge_days_c; contribs.append(("Days 5+/4+ drinks (year)", binge_days_c))

    ever_binge_c = 0.4 if ALQ151 == "Yes" else 0.0
    s += ever_binge_c; contribs.append(("Ever 5+/4+ drinks in day", ever_binge_c))

    freq_year_c = float(ALQ121) * 0.003
    s += freq_year_c; contribs.append(("How often drink (days/year)", freq_year_c))

    # Smoking
    smoker_map = {"Never": 0.0, "Former": 0.2, "Current": 0.5}
    smoke_c = smoker_map.get(Is_Smoker_Cat, 0.0)
    s += smoke_c; contribs.append(("Smoking status", smoke_c))

    # Sleep
    trouble_map = {"Never": 0.0,"Rarely": 0.1,"Sometimes": 0.2,"Often": 0.3,"Almost always": 0.4}
    sleep_trouble_c = trouble_map.get(SLQ050, 0.0)
    s += sleep_trouble_c; contribs.append(("Sleep trouble", sleep_trouble_c))

    sleep_dx_c = 0.2 if SLQ120 == "Yes" else 0.0
    s += sleep_dx_c; contribs.append(("Sleep diagnosis", sleep_dx_c))

    sleep_hours_c = 0.08 * abs(float(SLD012) - 7.0)  # farther from ~7h => higher risk
    s += sleep_hours_c; contribs.append(("Sleep hours (deviation from 7h)", sleep_hours_c))

    # Diet (simple heuristics)
    kcal_c = max(0.0, (float(DR1TKCAL) - 2000.0)) * 0.0002
    s += kcal_c; contribs.append(("Total kcal (above 2000)", kcal_c))

    sugar_c = float(DR1TSUGR) * 0.0005
    s += sugar_c; contribs.append(("Sugar (g)", sugar_c))

    carbs_c = max(0.0, float(DR1TCARB) - 200.0) * 0.0008
    s += carbs_c; contribs.append(("Carbs (g above 200)", carbs_c))

    fiber_c = -0.002 * float(DR1TFIBE)  # protective
    s += fiber_c; contribs.append(("Fiber (g)", fiber_c))

    fat_c = float(DR1TTFAT) * 0.001
    s += fat_c; contribs.append(("Total fat (g)", fat_c))

    # Physical activity (protective)
    pa_c = -0.05 * float(PAQ620)
    s += pa_c; contribs.append(("Moderate activity days/week", pa_c))

    # Socio-economic (protective in this simple score)
    fpl_c = -0.10 * max(0.0, float(INDFMPIR) - 1.0)
    s += fpl_c; contribs.append(("Income-to-poverty ratio (>1)", fpl_c))

    # Small bumps for male sex; race/ethnicity left neutral for fairness
    gender_c = 0.10 if RIAGENDR == "Male" else 0.0
    s += gender_c; contribs.append(("Male sex", gender_c))

    # Having ≥12 drinks last year (tiny bump; mostly captured above)
    any_alcohol_c = 0.05 if ALQ111 == "Yes" else 0.0
    s += any_alcohol_c; contribs.append(("Had ≥12 drinks last year", any_alcohol_c))

    # Convert to probability (logistic)
    proba = 1.0 / (1.0 + math.exp(-s))
    # Sort contributions by absolute impact, top 8 for display
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return proba, contribs[:8]

with st.form("risk_form"):
    st.subheader("Sociodemographic & Lifestyle Data")

    st.markdown("### Sociodemographic Data")
    c1, c2 = st.columns(2)
    with c1:
        RIAGENDR = st.selectbox("Gender (RIAGENDR)", CHOICES["RIAGENDR"])
        RIDRETH3 = st.selectbox("Race/Ethnicity (RIDRETH3)", CHOICES["RIDRETH3"])
    with c2:
        RIDAGEYR = st.slider("Age in Years (RIDAGEYR)", 18, 99, 45)
        INDFMPIR = st.number_input("Family Income-to-Poverty Ratio (INDFMPIR)", min_value=0.0, value=1.5, step=0.1)

    st.markdown("### Alcohol and Smoking Data")
    c3, c4 = st.columns(2)
    with c3:
        ALQ111 = st.selectbox("Had at least 12 alcohol drinks/1 yr? (ALQ111)", CHOICES["ALQ111"])
        ALQ142 = st.number_input("Average number of drinks on days consumed (ALQ142)", min_value=0.0, value=2.0, step=0.5)
        Is_Smoker_Cat = st.selectbox("Smoking Status (Is_Smoker_Cat)", CHOICES["Is_Smoker_Cat"])
    with c4:
        ALQ121 = st.number_input("How often do you drink in the last year? (ALQ121, days)", min_value=0.0, value=100.0, step=1.0)
        ALQ170 = st.number_input("Number of days had 5+/4+ drinks? (ALQ170)", min_value=0.0, value=0.0, step=1.0)
        ALQ151 = st.selectbox("Ever had 5+/4+ drinks in a day? (ALQ151)", CHOICES["ALQ151"])

    st.markdown("### Sleep Data")
    c5, c6 = st.columns(2)
    with c5:
        SLQ050 = st.selectbox("How often have trouble sleeping? (SLQ050)", CHOICES["SLQ050"])
    with c6:
        SLD012 = st.slider("Average sleep hours per day (SLD012)", 1, 12, 7)
        SLQ120 = st.selectbox("Had a medical sleep diagnosis? (SLQ120)", CHOICES["SLQ120"])

    st.markdown("### Dietary Intake (Last 24 Hours)")
    c7, c8, c9 = st.columns(3)
    with c7:
        DR1TKCAL = st.number_input("Total Kilocalories (DR1TKCAL)", min_value=0.0, value=2000.0, step=50.0)
        DR1TPROT = st.number_input("Total Protein (DR1TPROT)", min_value=0.0, value=75.0, step=5.0)
    with c8:
        DR1TCARB = st.number_input("Total Carbohydrates (DR1TCARB)", min_value=0.0, value=250.0, step=5.0)
        DR1TSUGR = st.number_input("Total Sugar (DR1TSUGR)", min_value=0.0, value=90.0, step=5.0)
    with c9:
        DR1TFIBE = st.number_input("Total Fiber (DR1TFIBE)", min_value=0.0, value=25.0, step=1.0)
        DR1TTFAT = st.number_input("Total Fat (DR1TTFAT)", min_value=0.0, value=65.0, step=2.0)

    st.markdown("### Physical & Anthropometric Data")
    c10, c11 = st.columns(2)
    with c10:
        PAQ620 = st.slider("Days of moderate activity per week (PAQ620)", 0, 7, 3)
    with c11:
        BMXBMI = st.number_input("BMI (BMXBMI)", min_value=10.0, max_value=80.0, value=28.0, step=0.1)

    submit = st.form_submit_button("Get Risk Assessment")

if submit:
    proba, top_contribs = compute_risk_and_contribs(
        RIAGENDR, RIDRETH3, RIDAGEYR, INDFMPIR,
        ALQ111, ALQ142, Is_Smoker_Cat, ALQ121, ALQ170, ALQ151,
        SLQ050, SLD012, SLQ120,
        DR1TKCAL, DR1TPROT, DR1TCARB, DR1TSUGR, DR1TFIBE, DR1TTFAT,
        PAQ620, BMXBMI
    )

    st.subheader("Your Results")
    st.metric("Estimated probability", f"{proba:.3f}")
    st.progress(min(1.0, proba))

    if proba >= 0.5:
        st.error("Based on your data, you may be at **higher risk** for NAFLD.")
    else:
        st.success("Based on your data, you may be at **lower risk** (threshold 0.5).")

    st.caption("This is a screening aid based on a transparent heuristic—not a medical diagnosis.")

    st.markdown("#### Top contributing factors")
    for name, val in top_contribs:
        sign = "▲" if val >= 0 else "▼"
        st.write(f"- {sign} **{name}**: {val:+.3f}")

st.divider()
st.caption("No external model or data files required. This app uses a transparent rule-based score so it runs anywhere.")
