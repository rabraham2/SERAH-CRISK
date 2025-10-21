# crisk_train.py — Train CRISK model, report metrics, save artifacts

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib


# Paths
BASE = Path("Dataset")
IN = BASE / "crisk_clean"
OUT = BASE / "crisk_models"
OUT.mkdir(parents=True, exist_ok=True)


# Load data
train = pd.read_parquet(IN / "train.parquet")
valid = pd.read_parquet(IN / "valid.parquet")
test = pd.read_parquet(IN / "test.parquet")

y_tr = train["target"].astype(int).to_numpy()
y_va = valid["target"].astype(int).to_numpy()
y_te = test["target"].astype(int).to_numpy()

X_tr = train.drop(columns=["target"])
X_va = valid.drop(columns=["target"])
X_te = test.drop(columns=["target"])

# Feature groups
pay_cols = [f"pay_{i}" for i in [1, 2, 3, 4, 5, 6]]
bill_cols = [f"bill_amt{i}" for i in [1, 2, 3, 4, 5, 6]]
payamt_cols = [f"pay_amt{i}" for i in [1, 2, 3, 4, 5, 6]]

num_cols = bill_cols + payamt_cols + ["age", "limit_bal"]
ord_cat_cols = pay_cols + ["sex", "education", "marriage"]

# Preprocessor (robust to sklearn version)
ver_major, ver_minor = map(int, skl_version.split(".")[:2])
ohe_kwargs = {"handle_unknown": "ignore"}
if (ver_major, ver_minor) >= (1, 2):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False
ohe = OneHotEncoder(**ohe_kwargs)

pre = ColumnTransformer(
    transformers=[
        ("cat", ohe, ord_cat_cols),
        ("num", RobustScaler(), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Model
clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=7,
    max_leaf_nodes=None,
    min_samples_leaf=25,
    l2_regularization=0.01,
    early_stopping=True,
    random_state=42,
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf),
])


# Train
pipe.fit(X_tr, y_tr)

# Helpers to save curves robustly
def _pad_to_len(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a).ravel()
    if len(a) >= n:
        return a[:n]
    out = np.empty(n, dtype=float)
    out[:len(a)] = a
    out[len(a):] = np.nan
    return out

def safe_save_roc_pr(y_true, proba, name: str):
    fpr, tpr, thr = roc_curve(y_true, proba)
    n = max(len(fpr), len(tpr), len(thr) + 1)  # thresholds is one shorter
    roc_df = pd.DataFrame({
        "fpr": _pad_to_len(fpr, n),
        "tpr": _pad_to_len(tpr, n),
        "thr": _pad_to_len(thr, n),  # padded with NaN on the last row
    })
    roc_df.to_csv(OUT / f"roc_{name}.csv", index=False)

    pr, rc, _ = precision_recall_curve(y_true, proba)
    pd.DataFrame({"precision": pr, "recall": rc}).to_csv(OUT / f"pr_{name}.csv", index=False)


# Evaluate
def eval_split(X, y, name: str) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred50 = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)

    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    thr_star = float(thr[j_idx]) if j_idx < len(thr) else 0.5
    pred_star = (proba >= thr_star).astype(int)

    rep50 = classification_report(y, pred50, output_dict=True, zero_division=0)
    rep_star = classification_report(y, pred_star, output_dict=True, zero_division=0)
    cm50 = confusion_matrix(y, pred50).tolist()
    cm_star = confusion_matrix(y, pred_star).tolist()

    safe_save_roc_pr(y, proba, name)

    return {
        "split": name,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "AUC": float(auc),
        "AP": float(ap),
        "Brier": float(brier),
        "threshold_star": thr_star,
        "report_at_0.50": rep50,
        "report_at_star": rep_star,
        "confusion_at_0.50": cm50,
        "confusion_at_star": cm_star,
    }

metrics = {
    "valid": eval_split(X_va, y_va, "valid"),
    "test":  eval_split(X_te, y_te, "test"),
}

# Permutation importance on TRANSFORMED X and FINAL ESTIMATOR
# (fixes the length mismatch)
Xva_tr = pipe.named_steps["pre"].transform(X_va)                   # transformed matrix
feat_names = pipe.named_steps["pre"].get_feature_names_out()       # expanded names match columns
perm = permutation_importance(
    estimator=pipe.named_steps["clf"],                              # final classifier only
    X=Xva_tr,
    y=y_va,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
)
imp = (pd.DataFrame({
    "feature": feat_names,
    "importance_mean": perm.importances_mean,
    "importance_std":  perm.importances_std,
})
       .sort_values("importance_mean", ascending=False))
imp.to_csv(OUT / "feature_importance_validation.csv", index=False)


# Persist model + metrics
joblib.dump(pipe, OUT / "crisk_model.pkl")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("Saved model to:", OUT / "crisk_model.pkl")
print("Validation AUC:", round(metrics["valid"]["AUC"], 4),
      "| Test AUC:", round(metrics["test"]["AUC"], 4))
print("Validation threshold* (Youden J):", round(metrics["valid"]["threshold_star"], 4))
print("Top 10 features by permutation importance:")
print(imp.head(10))
