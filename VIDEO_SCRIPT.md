# 🎥 Recorded Video Explanation — Script & Checklist (Component B, 3 marks)

**Required:** 5–10 minute MP4/MOV, **face (webcam picture-in-picture) + screen**, explaining *verbally* while you scroll through the Jupyter Notebook. Do not just run cells silently.
**Tool:** OBS Studio (Sources → *Display Capture* for the notebook + *Video Capture Device* for your webcam; drag the webcam to a corner).
**Submit:** upload to **Google Drive (sharable link, "Anyone with the link can view")** or **YouTube (Unlisted)** and paste the link in `README.md` (section 12) and in your submission form.

> Tip: before recording, run **Kernel ▸ Restart & Run All** so that every output is visible, and keep Streamlit running in another browser tab.

| Time | What to show | What to say (talking points — use your own words) |
|---|---|---|
| 0:00–0:30 | Face + title cell | Your name; project title *Smart Outcome Predictor*; the business problem: an EdTech company wants to predict **course completion (classification)** and **final score (regression)**; single models are unstable → ensembles. |
| 0:30–1:45 | **Part A** (theory) | In your own words: ensembles combine weak/strong learners; **bagging** = parallel + bootstrap → lowers variance; **boosting** = sequential, fixes previous errors → lowers bias; **voting** = different models, hard vs soft; **stacking** = meta-learner; **AdaBoost** re-weights samples, **XGBoost** uses gradients + regularisation; bias–variance trade-off. |
| 1:45–3:00 | **Part B** (data + EDA + preprocessing) | 5,200 rows, 19 columns; missing values in 3 columns → median imputation; targets and **why `final_score` / `completion_status` are not used as features for each other (leakage, r = 0.69)**; stratified 80/20 split; pipeline = impute + scale + one-hot. Point at the correlation heatmap: `avg_quiz_score` and `attendance_rate` matter most. |
| 3:00–4:15 | **Part C** (Bagging) | Same depth-8 tree as single baseline vs 200 bagged trees; improvement in ROC-AUC/RMSE; train–test gap; OOB score; the "number of trees" curve. |
| 4:15–5:45 | **Part D** (AdaBoost + XGBoost) | Show the **sequential-learning plot** (round 1 weak learner AUC ≈ 0.62 → ≈ 0.79); learner errors rise towards 0.5, weights shrink; XGBoost classifier/regressor; Gradient Boosting + tuned XGBoost; CV boxplot = robustness. |
| 5:45–7:00 | **Part E** (Voting & Stacking) | Hard vs soft voting table (soft is better because it uses confidence); stacking with a logistic-regression / ridge meta-learner; show the meta-learner weights plot. |
| 7:00–8:30 | **Part F & G** (evaluation) | Walk through the comparison tables, ROC curves, confusion matrices, actual-vs-predicted; explain **how the best model was chosen (5-fold CV, not the test set)** → *XGBoost (tuned)*; be honest that the linear baseline is statistically tied; top features; business recommendations. |
| 8:30–9:30 | **Streamlit app** | Open the live app: enter a student, click *Predict*, explain probability / predicted score / suggested actions; show the batch upload and model comparison tabs. |
| 9:30–10:00 | Wrap-up | Key conclusions, limitations (noisy data, ROC-AUC ≈ 0.79, R² ≈ 0.44), next steps; thank you. |

### Before you hit "Stop recording"
- [ ] Face clearly visible the whole time
- [ ] Notebook screen readable (zoom to 125 %)
- [ ] You explained verbally (did not only execute cells)
- [ ] Length between 5 and 10 minutes
- [ ] Link works in an incognito window
