"""
train_model.py
---------------
AI course project: Predicting Insurance Claim Fraud
Case study: University of Ibadan (Jaja Clinic / NHIS-linked claims)

Pipeline:
1. Load dataset
2. Exploratory Data Analysis (class balance, fraud rate by red flag)
3. Encode categorical features, split into train/test (stratified)
4. Train and compare 4 classification models:
     - Logistic Regression (baseline)
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier
5. Evaluate with Accuracy, Precision, Recall, F1 and ROC-AUC
6. Save comparison table, confusion matrix, ROC curves and
   feature importance plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

DATA_PATH = "/home/claude/fraud_ui/data/ui_health_insurance_claims.csv"
PLOTS_DIR = "/home/claude/fraud_ui/plots"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
target = "fraud_reported"

# ---------------------------------------------------------------
# 2. EDA plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df[target].value_counts().sort_index()
ax.bar(["Genuine (0)", "Fraud (1)"], counts.values, color=["#3B7A57", "#C1440E"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 40, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=10)
ax.set_ylabel("Number of claims")
ax.set_title("Class Balance: Genuine vs Fraudulent Claims\n(UI / Jaja Clinic case study)")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/class_balance.png", dpi=150)
plt.close()

flags = {
    "Diagnosis/treatment mismatch": df.diagnosis_treatment_mismatch == 1,
    "No referral letter": df.referral_letter_provided == 0,
    "Multiple providers, same week": df.multiple_providers_same_week == 1,
    "Weekend/holiday claim": df.weekend_or_holiday_claim == 1,
    "Claim > ceiling": df.claim_to_ceiling_ratio > 1,
    "Re-claim within 3 days": df.days_since_last_claim < 3,
}
overall = df[target].mean()
labels, rates = [], []
for label, mask in flags.items():
    labels.append(label)
    rates.append(df.loc[mask, target].mean())

fig, ax = plt.subplots(figsize=(8, 5))
order = np.argsort(rates)
ax.barh(np.array(labels)[order], np.array(rates)[order] * 100, color="#C1440E")
ax.axvline(overall * 100, color="#2E5EAA", linestyle="--",
           label=f"Overall fraud rate ({overall:.1%})")
ax.set_xlabel("Fraud rate (%)")
ax.set_title("Fraud Rate by Red-Flag Condition")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/fraud_rate_by_flag.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Feature encoding and split
# ---------------------------------------------------------------
X = df.drop(columns=["claim_id", target])
X = pd.get_dummies(X, columns=["beneficiary_type", "provider", "visit_type",
                                "diagnosis_category"], drop_first=True)
y = df[target]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

results = []
fitted, probas = {}, {}

for name, model in models.items():
    if name == "Gradient Boosting":
        sw = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
    })
    fitted[name] = model
    probas[name] = proba

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = fitted[best_name]
print(f"\nBest model (by F1): {best_name}")

with open("/home/claude/fraud_ui/data/model_comparison.json", "w") as f:
    json.dump({"results": results, "best_model": best_name,
               "fraud_rate": float(y.mean())}, f, indent=2)

# ---------------------------------------------------------------
# 5. ROC curves
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    auc = roc_auc_score(y_test, probas[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Fraud Detection Models (UI case study)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------
best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Predicted\nGenuine", "Predicted\nFraud"])
ax.set_yticks([0, 1], ["Actual\nGenuine", "Actual\nFraud"])
labels_cm = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    xlabel = "Feature Importance"
else:
    imp = np.abs(best_model.named_steps["clf"].coef_[0])
    xlabel = "Absolute Coefficient"
order = np.argsort(imp)[-12:]
ax.barh(np.array(feature_names)[order], imp[order], color="#3B7A57")
ax.set_xlabel(xlabel)
ax.set_title(f"Top Predictors of Fraud — {best_name}")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
plt.close()

print("\nPlots saved to", PLOTS_DIR)

