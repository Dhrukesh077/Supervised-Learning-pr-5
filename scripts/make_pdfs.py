"""Generates docs/Theory_Concepts.pdf and docs/Final_Report.pdf from the notebook outputs.
Run after executing the notebook:  python scripts/make_pdfs.py
"""
from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
DOCS, OUT, SHOTS, MODELS = ROOT / "docs", ROOT / "outputs", ROOT / "screenshots", ROOT / "models"
DOCS.mkdir(exist_ok=True)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

RED, DARK, GREY = colors.HexColor("#C00000"), colors.HexColor("#1f2937"), colors.HexColor("#f3f4f6")
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=13.2, textColor=DARK, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=11, textColor=colors.HexColor("#4b5563"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=19, leading=24, textColor=RED, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=12.5, leading=16, textColor=RED, spaceBefore=10, spaceAfter=4, keepWithNext=1)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=10, leading=13, spaceBefore=6, spaceAfter=2, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10.5, spaceAfter=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="DV-B", textColor=colors.white)
FORM = ParagraphStyle("form", parent=BODY, fontName="DV-I", alignment=1, backColor=GREY, borderPadding=5, spaceBefore=4, spaceAfter=8)

def P(t, s=BODY): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]

def table(rows, widths=None, header=True, zebra=True):
    data = [[Paragraph(str(c), CELLH if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), RED))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), GREY))
    t.setStyle(TableStyle(st)); return t

def df_table(df, fmt=3, widths=None):
    def f(v):
        if isinstance(v, float):
            return "–" if pd.isna(v) else f"{v:.{fmt}f}"
        return v
    rows = [list(df.columns)] + [[f(v) for v in r] for r in df.itertuples(index=False)]
    return KeepTogether([table(rows, widths)])

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("DV", 7.5); canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(2 * cm, 1.2 * cm, "Smart Outcome Predictor — Ensemble Learning Project")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(RED); canvas.setLineWidth(1.2); canvas.line(2 * cm, A4[1] - 1.5 * cm, A4[0] - 2 * cm, A4[1] - 1.5 * cm)
    canvas.restoreState()

def build(path, story, title):
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=1.8 * cm, title=title, author="Smart Outcome Predictor project").build(
        story, onFirstPage=footer, onLaterPages=footer)

def fig(name, width_cm=16, caption=None):
    p = SHOTS / name
    if not p.exists(): return []
    from PIL import Image as PI
    w, h = PI.open(p).size
    items = [Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)]
    if caption: items.append(P(caption, SMALL))
    return [KeepTogether(items), Spacer(1, 6)]

# =============================================================================================== THEORY
def theory():
    s = [P("Ensemble Learning — Theory Concepts", H1),
         P("Smart Outcome Predictor · Part A: Conceptual Foundation · Supervised Learning", SMALL), Spacer(1, 6)]

    s += [P("1. Ensemble Learning Methods", H2),
          P("<b>Definition.</b> An <b>ensemble</b> combines the predictions of several models (<i>base learners</i> or <i>weak learners</i>) "
            "to produce one final prediction that is, on average, more accurate and more stable than any individual model."),
          P("<b>Why they work.</b> (a) <b>Statistical:</b> many different hypotheses fit the training data equally well; averaging them reduces the risk of "
            "choosing a poor one. (b) <b>Computational:</b> local-search learners (trees, neural networks) get stuck in different optima; combining them "
            "approximates the best one. (c) <b>Representational:</b> a weighted combination can represent functions that no single base learner can."),
          P("For an average of B models with individual variance σ² and pairwise correlation ρ, the variance of the average is:"),
          P("Var(average) = ρ·σ² + (1 − ρ)·σ² / B", FORM),
          P("Hence ensembles help most when members are <b>accurate</b> and <b>diverse</b> (low ρ). Diversity is created by resampling the data (bagging), "
            "re-weighting it (boosting), using different algorithms (voting/stacking) or random feature subsets."),
          ]

    s += [P("2. Bagging vs Boosting vs Voting", H2),
          table([["", "Bagging", "Boosting", "Voting"],
                 ["Training scheme", "Parallel; each model sees a bootstrap sample (≈ 63.2 % unique rows)", "Sequential; model t+1 focuses on errors of models 1…t", "Parallel; every model sees the same data"],
                 ["Base learners", "Same algorithm, usually deep (high-variance) trees", "Same algorithm, usually shallow (high-bias) trees", "Different algorithms (LR, KNN, tree, SVM…)"],
                 ["Combination", "Majority vote / mean", "Weighted sum of learners", "Hard: majority of labels. Soft: mean of probabilities"],
                 ["Main benefit", "Lower <b>variance</b>", "Lower <b>bias</b> (and some variance)", "Cancels out diverse errors"],
                 ["Risk", "Little effect on bias; correlated trees", "Over-fits noisy labels / outliers", "Weak or correlated members add little"],
                 ["Examples", "Bagging, Random Forest", "AdaBoost, Gradient Boosting, XGBoost", "VotingClassifier / VotingRegressor"]],
                [2.7 * cm, 4.8 * cm, 4.8 * cm, 4.7 * cm]),
          Spacer(1, 4),
          P("<b>Out-of-bag (OOB) estimate.</b> Each bootstrap sample leaves out ≈ 36.8 % of the rows; predicting those rows with the trees that never saw them gives a free, "
            "nearly unbiased validation score for bagging.")]

    s += [P("3. Bias–Variance Trade-off", H2),
          P("For squared error, the expected prediction error at a point decomposes as:"),
          P("E[(y − ŷ)²] = Bias² + Variance + σ²(noise)", FORM),
          *bullets(["<b>Bias</b> – error from wrong/too-simple assumptions (under-fitting).",
                    "<b>Variance</b> – sensitivity to the particular training sample (over-fitting).",
                    "<b>Irreducible noise σ²</b> – cannot be removed by any model; it sets the ceiling on accuracy."]),
          P("<b>How ensembles address it.</b> <b>Bagging</b> keeps the low bias of deep trees and shrinks variance by averaging. <b>Boosting</b> starts with high-bias weak learners "
            "and reduces bias by adding learners that fit the remaining error (shrinkage and sub-sampling keep variance in check). <b>Voting</b> and <b>stacking</b> blend "
            "heterogeneous models so that their individual errors partly offset each other.")]

    s += [P("4. Voting Classifier and Stacking Ensemble", H2),
          P("<b>Voting Classifier.</b> Independent classifiers are trained on the same data and their outputs are merged by a fixed rule."),
          *bullets(["<b>Hard voting:</b> predicted class = the class receiving most votes. Uses only labels, so no probability ranking (no meaningful ROC-AUC).",
                    "<b>Soft voting:</b> predicted class = argmax of the (weighted) average of predicted class probabilities. Uses model confidence and usually performs at least as well when "
                    "probabilities are reasonably calibrated."]),
          P("<b>Stacking (stacked generalisation).</b> Level-0 base models are trained with K-fold cross-validation; their <i>out-of-fold</i> predictions form a new feature matrix on which a "
            "<b>meta-learner</b> (level-1 model such as Logistic Regression or Ridge) is trained. The meta-learner <i>learns</i> how much to trust each base model. "
            "Using out-of-fold predictions prevents the meta-learner from seeing over-fitted base predictions (data leakage)."),
          table([["", "Voting", "Stacking"],
                 ["Combination rule", "Fixed (majority / average)", "Learned by a meta-model"],
                 ["Extra training", "None", "K-fold base predictions + meta-model"],
                 ["Strength", "Simple, robust, fast", "Can exploit complementary strengths; usually the most accurate blend"],
                 ["Weakness", "Equal trust in all members", "More complex; risk of over-fitting the meta-level on small data"]],
                [3.5 * cm, 6.5 * cm, 7 * cm])]

    s += [P("5. AdaBoost vs XGBoost (conceptual)", H2),
          P("<b>AdaBoost</b> (Adaptive Boosting) maintains a weight w<sub>i</sub> for every training sample. In round t a weak learner h<sub>t</sub> is fitted to the weighted data, its weighted "
            "error ε<sub>t</sub> is measured, and it receives a vote weight:"),
          P("α<sub>t</sub> = ½ · ln((1 − ε<sub>t</sub>) / ε<sub>t</sub>)        w<sub>i</sub> ← w<sub>i</sub> · exp(α<sub>t</sub> · 1[h<sub>t</sub>(x<sub>i</sub>) ≠ y<sub>i</sub>])   (then re-normalise)", FORM),
          P("Misclassified samples therefore get heavier weights, forcing the next learner to concentrate on them. Final prediction: H(x) = sign(Σ α<sub>t</sub>·h<sub>t</sub>(x)). "
            "(scikit-learn uses the multi-class SAMME variant, and a learning-rate parameter that shrinks α<sub>t</sub>; AdaBoost.R2 handles regression.)"),
          P("<b>XGBoost</b> (eXtreme Gradient Boosting) fits trees to the <b>gradient</b> of a loss function and uses a <b>second-order</b> (Newton) approximation. Its regularised objective is:"),
          P("Obj = Σ<sub>i</sub> l(y<sub>i</sub>, ŷ<sub>i</sub>) + Σ<sub>k</sub> Ω(f<sub>k</sub>),     Ω(f) = γ·T + ½·λ·Σ w<sub>j</sub>²", FORM),
          P("where T is the number of leaves and w<sub>j</sub> are leaf weights. Together with shrinkage (learning rate), row/column sub-sampling, built-in missing-value handling and "
            "cache-aware/parallel split finding, this makes XGBoost faster and better regularised."),
          table([["", "AdaBoost", "XGBoost"],
                 ["Core idea", "Re-weight samples; weight learners by accuracy", "Fit gradient/Hessian of a differentiable loss"],
                 ["Typical learner", "Stumps / depth 1-3 trees", "Regularised trees, depth 3-8"],
                 ["Loss", "Exponential-type (classification), AdaBoost.R2 (regression)", "Any differentiable loss (log-loss, squared error, …)"],
                 ["Regularisation", "Learning rate, number of learners", "λ (L2), α (L1), γ, min_child_weight, sub-sampling, learning rate, early stopping"],
                 ["Missing values", "Must be imputed", "Handled natively"],
                 ["Speed", "Slower, mostly sequential", "Fast (parallel, histogram-based)"],
                 ["Noise sensitivity", "High (noisy samples gain weight)", "Lower, thanks to regularisation"]],
                [3.2 * cm, 6.8 * cm, 7 * cm])]

    s += [P("Glossary", H2),
          table([["Term", "Meaning"],
                 ["Weak learner", "A model only slightly better than random guessing (e.g. a decision stump)."],
                 ["Bootstrap", "Sampling n rows <i>with replacement</i> from n rows."],
                 ["Shrinkage / learning rate", "Factor (0–1] that scales each new learner's contribution to slow down fitting and reduce over-fitting."],
                 ["Meta-learner", "The model that combines base-model predictions in stacking."],
                 ["ROC-AUC", "Probability that a random positive is ranked above a random negative; threshold-independent ranking quality."],
                 ["RMSE / MAE / R²", "Root-mean-squared error / mean absolute error / proportion of variance explained."]],
                [4 * cm, 13 * cm])]
    build(DOCS / "Theory_Concepts.pdf", s, "Ensemble Learning — Theory Concepts")

# =============================================================================================== REPORT
def report():
    clf = pd.read_csv(OUT / "classification_results.csv"); reg = pd.read_csv(OUT / "regression_results.csv")
    best = pd.read_csv(OUT / "best_models_summary.csv"); meta = json.loads((MODELS / "metadata.json").read_text())
    bc, br = pd.read_csv(OUT / "bagging_vs_boosting_classification.csv"), pd.read_csv(OUT / "bagging_vs_boosting_regression.csv")
    ic, ir = pd.read_csv(OUT / "feature_importance_classification.csv"), pd.read_csv(OUT / "feature_importance_regression.csv")
    c, r = clf.set_index("Model"), reg.set_index("Model")
    bcn, brn = meta["classifier_name"], meta["regressor_name"]

    s = [P("Smart Outcome Predictor — Final Report", H1),
         P("Ensemble-based prediction of course completion (classification) and final score (regression)", SMALL), Spacer(1, 6)]

    s += [P("1. Executive summary", H2),
          *bullets([f"Ensemble methods improve markedly over a single decision tree: ROC-AUC {c.loc['Decision Tree (single)','ROC-AUC']:.3f} → "
                    f"{c.loc[bcn,'ROC-AUC']:.3f} (classification) and RMSE {r.loc['Decision Tree (single)','RMSE']:.2f} → {r.loc[brn,'RMSE']:.2f} (regression).",
                    f"Best ensemble by 5-fold cross-validation: <b>{bcn}</b> (classification) and <b>{brn}</b> (regression); both refit on all {meta['n_students']:,} students and deployed in the Streamlit app.",
                    f"Test performance of the deployed models — Accuracy {c.loc[bcn,'Accuracy']:.3f}, F1 {c.loc[bcn,'F1']:.3f}, ROC-AUC {c.loc[bcn,'ROC-AUC']:.3f}; "
                    f"MAE {r.loc[brn,'MAE']:.2f}, RMSE {r.loc[brn,'RMSE']:.2f}, R² {r.loc[brn,'R²']:.3f}.",
                    f"A regularised linear benchmark (Logistic Regression ROC-AUC {c.loc['Logistic Regression','ROC-AUC']:.3f}; Ridge RMSE {r.loc['Ridge','RMSE']:.2f}) is statistically tied with the best ensemble, "
                    "because the signal is mostly linear and the data are noisy.",
                    f"Predictions are driven by assessment and engagement: top features are {', '.join(ic['Unnamed: 0'].head(3))} (classification) and {', '.join(ir['Unnamed: 0'].head(3))} (regression)."])]

    s += [P("2. Data and method", H2),
          P("5,200 students, 15 predictors (10 numeric, 5 categorical) after dropping <i>student_id</i>, <i>course_start_date</i> and the opposite target (to prevent leakage). "
            "80/20 stratified split (random_state = 42). Median imputation + standard scaling for numeric and one-hot encoding for categorical features, wrapped in a scikit-learn "
            "<i>Pipeline</i> with every model. Models were compared on the test set and with 5-fold cross-validation (robustness); the deployed model was chosen by CV score, "
            "never by test score.")]

    s += [P("3. Results", H2), P("<b>Classification (test set, sorted by ROC-AUC)</b>", H3)]
    ct = clf[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC mean", "CV ROC-AUC std"]].copy()
    ct.columns = ["Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "CV AUC", "CV std"]
    s += [df_table(ct, 3, [4.2 * cm] + [2.1 * cm] * 7), P("Hard voting has no probabilities; its ROC-AUC is computed from class labels and is not comparable.", SMALL),
          P("<b>Regression (test set, sorted by RMSE)</b>", H3)]
    rt = reg[["Model", "MAE", "RMSE", "R²", "CV RMSE mean", "CV RMSE std"]].copy(); rt.columns = ["Model", "MAE", "RMSE", "R²", "CV RMSE", "CV std"]
    s += [df_table(rt, 3, [4.8 * cm] + [2.5 * cm] * 5), Spacer(1, 6)]
    s += fig("11_roc_curves.png", 10.5, "Figure 1 — ROC curves of the main ensembles and baselines.")
    s += fig("13_regression_metrics_comparison.png", 16.5, "Figure 2 — Regression error metrics for all models.")

    s += [P("4. Impact of Bagging vs Boosting", H2),
          df_table(bc.rename(columns={"Unnamed: 0": "Model"}).drop(columns=["Fit time (s)"]).rename(columns={"Overfit gap (train−test AUC)": "Gap (train−test)"}), 3),
          Spacer(1, 4),
          df_table(br.rename(columns={"Unnamed: 0": "Model"}).drop(columns=["Fit time (s)"]).rename(columns={"Overfit gap (test−train RMSE)": "Gap (test−train)"}), 3),
          Spacer(1, 4),
          *bullets(["<b>Bagging</b> lifts the single tree by reducing variance (ROC-AUC and RMSE improve markedly) but its deep trees still over-fit the training data (large train–test gap).",
                    "<b>Boosting</b> (AdaBoost, XGBoost) builds shallow learners sequentially, achieving similar or better test scores with a much smaller over-fitting gap; XGBoost is also the fastest to train.",
                    "Gains are real but bounded by irreducible noise in the data."])]
    s += fig("05_bagging_vs_single.png", 16.5, "Figure 3 — Single tree vs bagging, and effect of the number of bagged trees.")
    s += fig("06_adaboost_sequential_learning.png", 14.5, "Figure 4 — AdaBoost: sequential improvement of the weak learners.")

    s += [P("5. Comparison of tree-based boosting algorithms", H2),
          table([["Algorithm", "Observed behaviour"],
                 ["AdaBoost", f"Test ROC-AUC {c.loc['AdaBoost','ROC-AUC']:.3f} / RMSE {r.loc['AdaBoost','RMSE']:.2f}. Very small over-fitting gap; clearly weakest on regression; sensitive to noise."],
                 ["Gradient Boosting", f"ROC-AUC {c.loc['Gradient Boosting','ROC-AUC']:.3f} / RMSE {r.loc['Gradient Boosting','RMSE']:.2f}. Accurate and stable, slower to train."],
                 ["XGBoost", f"ROC-AUC {c.loc['XGBoost','ROC-AUC']:.3f} / RMSE {r.loc['XGBoost','RMSE']:.2f}. Fast, regularised, stable across CV folds."],
                 ["XGBoost (tuned)", f"ROC-AUC {c.loc['XGBoost (tuned)','ROC-AUC']:.3f} / RMSE {r.loc['XGBoost (tuned)','RMSE']:.2f}. Randomised-search tuning gave a modest improvement in cross-validated score."]],
                [3.6 * cm, 13.4 * cm])]
    s += fig("07_boosting_robustness_cv.png", 15, "Figure 5 — Robustness of the boosting models across 5 CV folds.")

    s += [P("6. Advantages of Voting and Stacking", H2),
          *bullets([f"<b>Soft voting</b> (ROC-AUC {c.loc['Voting (soft)','ROC-AUC']:.3f}) clearly outperforms <b>hard voting</b> (F1 {c.loc['Voting (hard)','F1']:.3f} vs {c.loc['Voting (soft)','F1']:.3f}) because it uses model confidence — "
                    "but it did not beat its best member here, because the members' errors are strongly correlated.",
                    f"<b>Stacking</b> (ROC-AUC {c.loc['Stacking','ROC-AUC']:.3f}; RMSE {r.loc['Stacking','RMSE']:.2f}) learns how to weight its base models and is among the top performers, at the price of heavier training/inference.",
                    "Both are most useful when the base models are diverse (linear, tree-based, instance-based, boosted)."])]
    s += fig("09_stacking_meta_learner_weights.png", 13, "Figure 6 — Coefficients learned by the stacking meta-learners.")

    s += [P("7. Final model recommendation for deployment", H2),
          P(f"<b>Classifier: {bcn}. Regressor: {brn}.</b> Selected by best 5-fold cross-validated ROC-AUC / RMSE among all ensemble techniques, then refit on the full dataset. "
            "They offer top-tier accuracy, small variance across folds, sub-second training, native handling of non-linearities and missing values, and a simple deployment "
            "(single scikit-learn pipeline saved with joblib)."),
          P(f"<b>Honest note:</b> the linear benchmark (Logistic Regression / Ridge) performs essentially the same on this dataset. If interpretability or maximal simplicity "
            "is the priority, it is a perfectly valid production alternative; the ensemble is recommended here because it is robust to non-linear patterns, outliers and future data drift.")]
    s += fig("17_feature_importance.png", 16.5, "Figure 7 — Permutation importance of the deployed models.")

    s += [P("8. Business interpretation, limitations and next steps", H2),
          *bullets(["Use the <b>completion probability</b> to rank students by drop-out risk and trigger early support (reminders, mentoring) — especially for low quiz scores and low attendance in the first weeks.",
                    "Use the <b>predicted final score</b> to flag likely under-performers and offer targeted practice.",
                    "Recall for completers at the default 0.5 threshold is moderate; set the threshold from the relative cost of missing an at-risk student vs an unnecessary intervention (the app has a threshold slider).",
                    "<b>Limitations:</b> modest signal (ROC-AUC ≈ 0.79, R² ≈ 0.44); many true scores are capped at 100; data appear synthetic; single hold-out split + CV; no fairness audit yet.",
                    "<b>Next steps:</b> week-by-week activity features, probability calibration, threshold optimisation, fairness analysis across regions/education levels, drift monitoring after deployment."])]
    build(DOCS / "Final_Report.pdf", s, "Smart Outcome Predictor — Final Report")

if __name__ == "__main__":
    theory(); report(); print("PDFs written to", DOCS)
