"""
Smart Outcome Predictor — Streamlit app
---------------------------------------
Predicts (1) whether a student will complete an online course and
(2) the student's final performance score, using the best ensemble models
selected in Smart_Outcome_Predictor.ipynb.

Run locally:   streamlit run app.py
"""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"
SHOTS = BASE / "screenshots"
DATA_FILE = BASE / "data" / "Smart_Outcome_Predictor_Dataset_5200.xlsx"

INT_FEATURES = ["age", "week_of_year", "sessions", "videos_watched",
                "quiz_attempts", "assignments_submitted", "forum_posts"]
LABELS = {
    "age": "Age (years)", "week_of_year": "Course start week (1-52)", "sessions": "Number of sessions",
    "time_spent_hours": "Time spent (hours)", "videos_watched": "Videos watched",
    "quiz_attempts": "Quiz attempts", "assignments_submitted": "Assignments submitted",
    "forum_posts": "Forum posts", "avg_quiz_score": "Average quiz score (0-100)",
    "attendance_rate": "Attendance rate (0-1)", "country_region": "Country / region",
    "device_type": "Device type", "education_background": "Education background",
    "course_level": "Course level", "course_category": "Course category",
}

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts():
    try:
        clf = joblib.load(MODELS / "best_classifier.joblib")
        reg = joblib.load(MODELS / "best_regressor.joblib")
        meta = json.loads((MODELS / "metadata.json").read_text())
        return clf, reg, meta, None
    except Exception as exc:  # missing files or scikit-learn / xgboost version mismatch
        return None, None, None, exc


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_excel(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_csv(name):
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


clf_model, reg_model, meta, load_error = load_artifacts()

st.title("🎓 Smart Outcome Predictor")
st.caption("Ensemble-based machine learning (Bagging · Boosting · Voting · Stacking) for EdTech analytics — "
           "Red & White Skill Education project")

if load_error is not None:
    st.error("Trained models could not be loaded. Run all cells of `Smart_Outcome_Predictor.ipynb` "
             "(it regenerates the `models/` folder) and make sure the package versions in "
             "`requirements.txt` are installed.")
    st.exception(load_error)
    st.stop()

FEATURES = meta["features"]
NUM = meta["numeric_features"]
CAT = meta["categorical_features"]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Deployed models")
    st.markdown(f"**Classifier:** {meta['classifier_name']}")
    cm = meta["classification_test_metrics"]
    st.caption(f"Test ROC-AUC {cm['ROC-AUC']:.3f} · F1 {cm['F1']:.3f} · Accuracy {cm['Accuracy']:.3f}")
    st.markdown(f"**Regressor:** {meta['regressor_name']}")
    rm = meta["regression_test_metrics"]
    st.caption(f"Test RMSE {rm['RMSE']:.2f} · MAE {rm['MAE']:.2f} · R² {rm['R²']:.3f}")
    st.divider()
    st.caption(f"Trained on {meta['n_students']:,} students. Metrics are from a held-out 20 % test set.")
    st.caption("Predictions are probabilistic estimates for risk ranking and early intervention — "
               "not guarantees about an individual student.")

tab_pred, tab_batch, tab_models, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📂 Batch prediction", "📊 Model comparison", "🔍 Data explorer", "ℹ️ About"])


# ----------------------------------------------------------------------------- helpers
def grade_band(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 65:
        return "Average"
    return "Needs support"


def advice(row: dict, prob: float):
    tips = []
    if row["attendance_rate"] < 0.75:
        tips.append("Attendance is low (< 75 %) — send attendance reminders / schedule a check-in.")
    if row["avg_quiz_score"] < 55:
        tips.append("Quiz performance is weak (< 55) — recommend revision material or mentoring.")
    if row["sessions"] < 10:
        tips.append("Few learning sessions so far — encourage a regular study routine.")
    if row["assignments_submitted"] < 2:
        tips.append("Very few assignments submitted — follow up on pending work.")
    if row["videos_watched"] < 10:
        tips.append("Low video consumption — nudge the student towards the core lectures.")
    if not tips and prob >= 0.5:
        tips.append("Engagement looks healthy — keep the student on track with regular encouragement.")
    if not tips:
        tips.append("No single red flag stands out; the model sees a combination of moderate signals. "
                    "A light-touch check-in is recommended.")
    return tips


def to_frame(row: dict) -> pd.DataFrame:
    return pd.DataFrame([row])[FEATURES]


# ----------------------------------------------------------------------------- tab 1: predict
with tab_pred:
    st.subheader("Enter student information")
    ranges = meta["numeric_ranges"]
    opts = meta["category_options"]
    values = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Profile**")
        values["age"] = st.number_input(LABELS["age"], 10, 80, int(ranges["age"]["median"]))
        values["country_region"] = st.selectbox(LABELS["country_region"], opts["country_region"])
        values["education_background"] = st.selectbox(LABELS["education_background"], opts["education_background"])
        values["device_type"] = st.selectbox(LABELS["device_type"], opts["device_type"])
        values["course_level"] = st.selectbox(LABELS["course_level"], opts["course_level"])
        values["course_category"] = st.selectbox(LABELS["course_category"], opts["course_category"])
        values["week_of_year"] = st.number_input(LABELS["week_of_year"], 1, 52, int(ranges["week_of_year"]["median"]))
    with c2:
        st.markdown("**Engagement**")
        values["sessions"] = st.number_input(LABELS["sessions"], 0, 300, int(ranges["sessions"]["median"]))
        values["time_spent_hours"] = st.number_input(LABELS["time_spent_hours"], 0.0, 500.0,
                                                     float(ranges["time_spent_hours"]["median"]), step=0.5)
        values["videos_watched"] = st.number_input(LABELS["videos_watched"], 0, 500, int(ranges["videos_watched"]["median"]))
        values["forum_posts"] = st.number_input(LABELS["forum_posts"], 0, 100, int(ranges["forum_posts"]["median"]))
    with c3:
        st.markdown("**Assessment & attendance**")
        values["quiz_attempts"] = st.number_input(LABELS["quiz_attempts"], 0, 100, int(ranges["quiz_attempts"]["median"]))
        values["assignments_submitted"] = st.number_input(LABELS["assignments_submitted"], 0, 100,
                                                          int(ranges["assignments_submitted"]["median"]))
        values["avg_quiz_score"] = st.slider(LABELS["avg_quiz_score"], 0.0, 100.0,
                                             float(ranges["avg_quiz_score"]["median"]), 0.5)
        values["attendance_rate"] = st.slider(LABELS["attendance_rate"], 0.0, 1.0,
                                              float(ranges["attendance_rate"]["median"]), 0.01)
        threshold = st.slider("Decision threshold for 'Completed'", 0.10, 0.90, 0.50, 0.05,
                              help="Lower it to catch more likely completers (higher recall); raise it for higher precision.")

    if st.button("🔮 Predict outcome", type="primary", width="stretch"):
        frame = to_frame(values)
        prob = float(clf_model.predict_proba(frame)[0, 1])
        score = float(min(100.0, max(0.0, reg_model.predict(frame)[0])))
        completed = prob >= threshold

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Completion probability", f"{prob:.1%}")
        r2.metric("Predicted outcome", "✅ Likely to complete" if completed else "⚠️ At risk of not completing")
        r3.metric("Predicted final score", f"{score:.1f} / 100", grade_band(score))
        st.progress(min(max(prob, 0.0), 1.0), text="Probability of completing the course")

        base_rate = float(load_data()["completion_status"].mean())
        st.caption(f"Reference: {base_rate:.1%} of students in the training data completed their course "
                   f"(average final score {load_data()['final_score'].mean():.1f}).")
        st.markdown("#### Suggested actions")
        for t in advice(values, prob):
            st.markdown(f"- {t}")

# ----------------------------------------------------------------------------- tab 2: batch
with tab_batch:
    st.subheader("Predict for many students at once")
    st.write("Upload a **CSV or Excel** file that contains these columns (extra columns are ignored, "
             "missing numeric values are imputed automatically):")
    st.code(", ".join(FEATURES))
    template = pd.DataFrame([{**{c: meta["numeric_ranges"][c]["median"] for c in NUM},
                              **{c: meta["category_options"][c][0] for c in CAT}}])[FEATURES]
    st.download_button("⬇️ Download input template (CSV)", template.to_csv(index=False).encode(),
                       "input_template.csv", "text/csv")
    up = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if up is not None:
        try:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            missing_cols = [c for c in FEATURES if c not in raw.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                out = raw.copy()
                out["pred_completion_probability"] = clf_model.predict_proba(raw[FEATURES])[:, 1].round(4)
                out["pred_completion_status"] = (out["pred_completion_probability"] >= 0.5).astype(int)
                out["pred_final_score"] = reg_model.predict(raw[FEATURES]).clip(0, 100).round(1)
                st.success(f"Predicted {len(out):,} students.")
                st.dataframe(out.head(200), width="stretch")
                st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

# ----------------------------------------------------------------------------- tab 3: models
with tab_models:
    st.subheader("Ensemble model comparison (held-out test set)")
    ct, rt = load_csv("classification_results.csv"), load_csv("regression_results.csv")
    if ct is not None:
        st.markdown("**Classification — completion_status**")
        st.dataframe(ct.drop(columns=["Fit time (s)"], errors="ignore"), width="stretch", hide_index=True)
    if rt is not None:
        st.markdown("**Regression — final_score**")
        st.dataframe(rt.drop(columns=["Fit time (s)"], errors="ignore"), width="stretch", hide_index=True)
    st.markdown("**Plots**")
    for fname, cap in [("10_classification_metrics_comparison.png", "Classification metrics"),
                       ("11_roc_curves.png", "ROC curves"),
                       ("13_regression_metrics_comparison.png", "Regression metrics"),
                       ("17_feature_importance.png", "Feature importance of the deployed models")]:
        if (SHOTS / fname).exists():
            st.image(str(SHOTS / fname), caption=cap, width="stretch")

# ----------------------------------------------------------------------------- tab 4: data
with tab_data:
    st.subheader("Dataset explorer")
    data = load_data()
    a, b, c, d = st.columns(4)
    a.metric("Students", f"{len(data):,}")
    b.metric("Completion rate", f"{data['completion_status'].mean():.1%}")
    c.metric("Mean final score", f"{data['final_score'].mean():.1f}")
    d.metric("Features", len(FEATURES))
    col_cat = st.selectbox("Group by", CAT)
    g = data.groupby(col_cat).agg(students=("student_id", "count"),
                                  completion_rate=("completion_status", "mean"),
                                  mean_final_score=("final_score", "mean")).round(3)
    x1, x2 = st.columns(2)
    x1.markdown("**Completion rate**"); x1.bar_chart(g["completion_rate"])
    x2.markdown("**Mean final score**"); x2.bar_chart(g["mean_final_score"])
    st.markdown("**Sample of the data**")
    st.dataframe(data.head(100), width="stretch")

# ----------------------------------------------------------------------------- tab 5: about
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Smart Outcome Predictor** combines ensemble learning methods to solve two supervised tasks on student data:

* **Classification** — will a student complete the course? (`completion_status`)
* **Regression** — what will the student's final score be? (`final_score`)

**Methods compared:** Bagging, AdaBoost, Gradient Boosting, XGBoost (default and tuned), Hard/Soft Voting and
Stacking, against a single decision tree and linear baselines. The best models were selected by 5-fold
cross-validation and refit on all data for this app:

* Classifier → **{meta['classifier_name']}**
* Regressor → **{meta['regressor_name']}**

**Inputs used:** 10 numeric and 5 categorical features. The other target is never used as an input (no leakage).

**Important:** predictions are statistical estimates. Use them to prioritise support, not to make final
decisions about individuals. See `README.md` and the notebook for methodology and limitations.
""")
