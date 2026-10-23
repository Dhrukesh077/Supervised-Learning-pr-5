<div align="center">

# 🎓 Smart Outcome Predictor

**Ensemble learning to predict online-course completion and final student score**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-3.4-green) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

*Red & White Skill Education · Supervised Learning project*

</div>

| Resource | Link |
|---|---|
| 🌐 Live Streamlit app | `<ADD STREAMLIT APP URL>` |
| 🎥 Video explanation | `<ADD GOOGLE DRIVE / YOUTUBE UNLISTED LINK>` |
| 📓 Notebook | [`Smart_Outcome_Predictor.ipynb`](Smart_Outcome_Predictor.ipynb) |
| 📄 Theory document | [`docs/Theory_Concepts.pdf`](docs/Theory_Concepts.pdf) |
| 📊 Final report | [`docs/Final_Report.pdf`](docs/Final_Report.pdf) |

---

## Table of contents
[Overview](#overview) · [Results at a glance](#results-at-a-glance) · [Project structure](#project-structure) · [Dataset](#dataset) · [Methodology](#methodology) · [Detailed results](#detailed-results) · [Streamlit app](#streamlit-app) · [Getting started](#getting-started) · [Deployment](#deployment) · [Task coverage](#task-coverage) · [Limitations & future work](#limitations--future-work) · [Troubleshooting](#troubleshooting)

---

## Overview

An EdTech analytics platform wants to know, from a student's behaviour and background:

| Task | Target | Type |
|---|---|---|
| Will the student **complete** the course? | `completion_status` (0 / 1) | Classification |
| What **final score** (0–100) will the student achieve? | `final_score` | Regression |

Single models gave unstable predictions, so this project builds and compares **ensemble** solutions — **Bagging, Boosting (AdaBoost, Gradient Boosting, XGBoost), Voting and Stacking** — evaluates them with held-out tests and cross-validation, selects the best model per task, and serves it through a **Streamlit web app**.

---

## Results at a glance

| | Classification | Regression |
|---|---|---|
| **Selected model** (best 5-fold CV among ensembles) | **XGBoost (tuned)** | **XGBoost (tuned)** |
| **Test performance** | Accuracy **0.742** · F1 **0.629** · ROC-AUC **0.788** | MAE **8.13** · RMSE **10.12** · R² **0.437** |
| Single decision tree | ROC-AUC 0.712 · F1 0.549 | RMSE 11.57 · R² 0.265 |
| Linear benchmark (LogReg / Ridge) | ROC-AUC 0.794 · F1 0.634 | RMSE 10.04 · R² 0.446 |

**Key findings**

1. **Ensembles beat a single decision tree** on both tasks (ROC-AUC 0.71 → 0.79; RMSE 11.6 → 10.1). Boosting also shrinks the train–test gap dramatically; bagging does so only modestly.
2. **The top ensembles are practically tied** — their differences are about one cross-validation standard deviation.
3. **Soft voting beats hard voting**; **stacking** matches the best boosters by learning how much to trust each base model.
4. **A regularised linear model is statistically tied with the best ensemble** (see [note](#a-note-on-the-linear-benchmark)) because the signal is largely linear and the data are noisy.
5. **Drivers:** `avg_quiz_score`, `attendance_rate` and engagement features (`time_spent_hours`, `sessions`, `videos_watched`) matter most; demographics add little.

---

## Project structure

```
Smart_Outcome_Predictor/
├── README.md
├── Smart_Outcome_Predictor.ipynb     # full solution (Parts A–G), executed
├── app.py                            # Streamlit web app
├── requirements.txt                  # runtime dependencies (pinned)
├── requirements-dev.txt              # + Jupyter & PDF tooling
├── VIDEO_SCRIPT.md                   # script/checklist for the recorded explanation
├── data/
│   └── Smart_Outcome_Predictor_Dataset_5200.xlsx
├── models/                           # created by the notebook, used by the app
│   ├── best_classifier.joblib        # preprocessing + XGBoost (tuned)
│   ├── best_regressor.joblib         # preprocessing + XGBoost (tuned)
│   └── metadata.json                 # features, categories, ranges, metrics
├── outputs/                          # evaluation tables (CSV)
├── screenshots/                      # 17 labelled plots (01 … 17)
├── docs/
│   ├── Theory_Concepts.pdf
│   └── Final_Report.pdf
└── scripts/make_pdfs.py              # regenerates the two PDFs
```

---

## Dataset

`data/Smart_Outcome_Predictor_Dataset_5200.xlsx` — **5,200 students × 19 columns**, no duplicate students, **37.5 % completed**.

| Group | Columns |
|---|---|
| Demographics & context | `age`, `country_region`, `device_type`, `education_background`, `course_level`, `course_category`, `week_of_year` |
| Engagement | `sessions`, `time_spent_hours`, `videos_watched`, `quiz_attempts`, `assignments_submitted`, `forum_posts`, `attendance_rate` |
| Assessment | `avg_quiz_score` |
| Targets | `completion_status` (classification), `final_score` (regression, 35–100, mean 74.8) |
| Not used | `student_id` (identifier), `course_start_date` (seasonality already in `week_of_year`) |

Missing values: `time_spent_hours` (112), `avg_quiz_score` (81), `attendance_rate` (80).

---

## Methodology

**Features and leakage.** 15 features (10 numeric, 5 categorical). `completion_status` and `final_score` correlate at r ≈ 0.69, so **neither is ever used as a feature for the other task** — at prediction time both are unknown.

**Split.** One stratified 80/20 split (`random_state=42`) shared by both tasks: 4,160 training and 1,040 test students.

**Preprocessing** (inside every model's `Pipeline`, so it is fitted on training data only): numeric → median imputation → `StandardScaler`; categorical → `OneHotEncoder(handle_unknown="ignore")`.

**Models**

| Family | Classification | Regression |
|---|---|---|
| Single baseline | Decision tree (depth 8) | Decision tree (depth 8) |
| Bagging | 200 × depth-8 trees, bootstrap, OOB score | same |
| AdaBoost | 200 × depth-2 trees, lr 0.1 | 200 × depth-3 trees, lr 0.1 |
| Gradient Boosting | sklearn defaults | sklearn defaults |
| XGBoost | 300 trees, depth 3, lr 0.05, subsample / colsample 0.8 | same |
| XGBoost (tuned) | `RandomizedSearchCV` (20 iterations, 3-fold, training data only) | same |
| Voting | LogReg + KNN + Tree + SVC — **hard vs soft** | Ridge + KNN + Tree + SVR |
| Stacking | RF + XGBoost + AdaBoost + KNN → LogReg meta-learner | same base → Ridge meta-learner |

**Evaluation.** Classification: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves. Regression: MAE, RMSE, R², actual-vs-predicted and residual plots. Robustness: 5-fold cross-validation for every model and train–test gaps. Explainability: permutation importance.

**Model selection.** The deployed model is the ensemble with the best **cross-validated** score (ROC-AUC / RMSE) — not the best test score. The test set is used only as the final unbiased check. The chosen pipelines are then refit on all 5,200 students and saved.

---

## Detailed results

### Classification — `completion_status` (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.745 | 0.687 | 0.590 | 0.634 | 0.794 | 0.785 ± 0.009 |
| AdaBoost | 0.733 | 0.685 | 0.531 | 0.598 | 0.789 | 0.774 ± 0.009 |
| **XGBoost (tuned)** | 0.742 | 0.684 | 0.582 | 0.629 | 0.788 | 0.780 ± 0.009 |
| Gradient Boosting | 0.739 | 0.676 | 0.582 | 0.625 | 0.788 | 0.771 ± 0.008 |
| Stacking | 0.739 | 0.685 | 0.559 | 0.616 | 0.787 | 0.779 ± 0.011 |
| Voting (soft) | 0.729 | 0.682 | 0.518 | 0.589 | 0.785 | 0.776 ± 0.010 |
| XGBoost | 0.731 | 0.660 | 0.582 | 0.619 | 0.780 | 0.771 ± 0.007 |
| Bagging | 0.725 | 0.657 | 0.559 | 0.604 | 0.777 | 0.770 ± 0.012 |
| Decision Tree (single) | 0.679 | 0.580 | 0.520 | 0.549 | 0.712 | 0.696 ± 0.015 |
| Voting (hard) | 0.728 | 0.712 | 0.462 | 0.560 | 0.675 | – |

*Hard voting has no probabilities, so its ROC-AUC (from labels) is not comparable and it has no CV score.*

![ROC curves](screenshots/11_roc_curves.png)

### Regression — `final_score` (test set)

| Model | MAE | RMSE | R² | CV RMSE |
|---|---|---|---|---|
| Ridge | 8.084 | 10.045 | 0.446 | 9.536 ± 0.124 |
| Stacking | 8.124 | 10.121 | 0.437 | 9.654 ± 0.119 |
| **XGBoost (tuned)** | 8.131 | 10.123 | 0.437 | 9.626 ± 0.118 |
| Voting | 8.165 | 10.124 | 0.437 | 9.736 ± 0.132 |
| XGBoost | 8.155 | 10.170 | 0.432 | 9.708 ± 0.146 |
| Gradient Boosting | 8.207 | 10.200 | 0.428 | 9.701 ± 0.120 |
| Bagging | 8.229 | 10.241 | 0.424 | 9.846 ± 0.082 |
| AdaBoost | 8.606 | 10.572 | 0.386 | 10.268 ± 0.115 |
| Decision Tree (single) | 9.239 | 11.566 | 0.265 | 11.469 ± 0.166 |

![Regression metrics](screenshots/13_regression_metrics_comparison.png)

### What the experiments show

| Question | Finding |
|---|---|
| Does bagging help? | Yes: with the same depth-8 tree, ROC-AUC 0.712 → 0.777 and RMSE 11.57 → 10.24. Gains flatten after ~25–50 trees. |
| How do weak learners improve in AdaBoost? | One weak learner reaches ROC-AUC ≈ 0.62; after 200 rounds ≈ 0.79. Each learner's weighted error rises from ≈ 0.30 toward 0.5 and its voting weight shrinks. |
| XGBoost vs other boosting? | XGBoost, tuned XGBoost and Gradient Boosting are close and stable across folds; AdaBoost is clearly weakest on regression (RMSE 10.57). XGBoost trains in ≈ 0.2 s vs ≈ 3 s for bagging. |
| Hard vs soft voting? | Soft wins (F1 0.589 vs 0.560); it does not beat its best member because member errors are correlated. |
| Is stacking worth it? | It matches the best boosters (ROC-AUC 0.787, RMSE 10.12) at a higher training cost. |

![AdaBoost sequential learning](screenshots/06_adaboost_sequential_learning.png)
![Feature importance](screenshots/17_feature_importance.png)

All 17 plots are in [`screenshots/`](screenshots/); the full comparison tables are in [`outputs/`](outputs/).

### A note on the linear benchmark

Logistic Regression (ROC-AUC 0.794) and Ridge (RMSE 10.04) are marginally better than the selected ensemble (0.788; 10.12). The relationship is largely linear and the targets are noisy (many true scores are capped at 100), so ensembling gives a large gain over a single tree but a negligible one over a well-regularised linear model. The ensemble is deployed because the project calls for an ensemble solution and it is more robust to non-linear patterns and outliers; if interpretability is the priority, the linear model is a valid alternative.

### Business interpretation

- Rank students by **completion probability** and trigger early support for low quiz scores or attendance in the first weeks.
- Use the **predicted score** to flag likely under-performers and offer targeted practice.
- Choose the **decision threshold** from the cost of missing an at-risk student versus an unnecessary intervention (the app has a slider).
- Expect risk ranking, not certainty: ROC-AUC ≈ 0.79, R² ≈ 0.44.

---

## Streamlit app

`app.py` loads the saved pipelines and provides five tabs:

| Tab | Purpose |
|---|---|
| **Predict** | Enter one student → completion probability, outcome at an adjustable threshold, predicted final score with grade band, suggested actions |
| **Batch prediction** | Upload CSV/Excel with the 15 feature columns → predictions for all rows + download (template provided) |
| **Model comparison** | Test-set tables and plots for all models |
| **Data explorer** | Dataset KPIs and completion / score by category |
| **About** | Methods, selected models, caveats |

---

## Getting started

**Requirements:** Python 3.11+ (tested on 3.12).

```bash
git clone <YOUR-REPO-URL>
cd Smart_Outcome_Predictor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Re-run the analysis** (≈ 8–10 minutes on one CPU; regenerates plots, tables and models). Run from the project root:

```bash
jupyter lab                        # open the notebook → Kernel ▸ Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute Smart_Outcome_Predictor.ipynb --inplace
```

**Launch the app:**

```bash
streamlit run app.py               # http://localhost:8501
```

**Regenerate the PDFs (optional):** `python scripts/make_pdfs.py`

All randomness uses `random_state=42`, so results are reproducible.

---

## Deployment

1. Create a public GitHub repository and push the project. Commit the `models/` folder — the app needs it.
   ```bash
   git init && git add . && git commit -m "Smart Outcome Predictor"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
2. Open <https://share.streamlit.io> → **New app** → choose the repo, branch `main`, main file `app.py`, Python 3.12 → **Deploy**.
3. Add the app URL and video link to the table at the top of this README, then commit and push.

---

## Task coverage

| Part | Tasks | Location |
|---|---|---|
| A — Conceptual foundation | 1–5 | Notebook Part A · `docs/Theory_Concepts.pdf` |
| B — Dataset understanding & preparation | 6–8 | Notebook Part B (plots 01–04) |
| C — Bagging | 9–11 | Notebook Part C (plot 05) |
| D — Boosting (AdaBoost, XGBoost) | 12–17 | Notebook Part D (plots 06–07) |
| E — Voting & Stacking | 18–21 | Notebook Part E (plots 08–09) |
| F — Evaluation & comparison | 22, 29, 30 | Notebook Part F (plots 10–17) |
| G — Final analysis & reporting | 31–32 | Notebook Part G · `docs/Final_Report.pdf` · app · this README |

---

## Limitations & future work

- **Limitations:** moderate signal (ROC-AUC ≈ 0.79, R² ≈ 0.44); ceiling effect at `final_score = 100`; the dataset appears synthetic; no fairness audit; moderate recall for completers at the default 0.5 threshold; column meanings are inferred from their names.
- **Future work:** week-by-week activity features for early-warning models; probability calibration and threshold optimisation; censored/quantile regression for the score cap; SHAP explanations; fairness analysis across regions and education levels; drift monitoring.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App says the models could not be loaded | Install the exact versions in `requirements.txt` (the scikit-learn / XGBoost pins matter for `.joblib` files), or re-run the notebook to regenerate `models/` |
| `FileNotFoundError` for the dataset | Start Jupyter from the project root and keep the file in `data/` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Notebook is slow | It fits about 25 models with 5-fold CV; lower `n_estimators` or the number of folds for a quick run |
| Streamlit Cloud build fails | Select Python 3.11 or 3.12 and keep `requirements.txt` in the repo root |

---

<div align="center"><sub>Built for the Red & White Skill Education "Smart Outcome Predictor" project.</sub></div>
