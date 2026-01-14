# ===============================
# plots.py – Final Analysis & Plots
# ===============================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Paths
# ===============================
RES_DIR = "results"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# Load ground truth
# ===============================
y_true_full = np.load(os.path.join(RES_DIR, "y_true.npy"))

# ===============================
# Models to compare
# ===============================
MODELS = {
    "One-hot + RF": "y_pred_onehot_rf.npy",
    "ProtBERT + RF": "y_pred_protbert_rf.npy",
    "ESM2 + RF": "y_pred_esm2_rf.npy",
    "ESM2 + MLP": "y_pred_esm2_mlp.npy",
}

# ===============================
# Metric helpers
# ===============================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def bootstrap_ci(y_true, y_pred, n_boot=500, seed=42):
    """Manual bootstrap for RMSE (95% CI)"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmses = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rmses.append(rmse(y_true[idx], y_pred[idx]))

    low = np.percentile(rmses, 2.5)
    high = np.percentile(rmses, 97.5)
    return max(0, low), max(0, high)


# ===============================
# Collect results
# ===============================
results = {}

for name, file in MODELS.items():
    y_pred = np.load(os.path.join(RES_DIR, file))

    # Align sizes (test set only)
    y_true = y_true_full[: len(y_pred)]

    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    ci_low, ci_high = bootstrap_ci(y_true, y_pred)

    results[name] = {
        "rmse": rmse_val,
        "r2": r2_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_true": y_true,
        "y_pred": y_pred,
    }

# ===============================
# 1️⃣ RMSE BAR PLOT + CI
# ===============================
labels = list(results.keys())
rmse_vals = [results[m]["rmse"] for m in labels]

yerr = [
    [
        results[m]["rmse"] - results[m]["ci_low"],
        results[m]["ci_high"] - results[m]["rmse"],
    ]
    for m in labels
]

yerr = np.array(yerr).T  # (2, N)

plt.figure(figsize=(10, 6))
plt.bar(labels, rmse_vals, yerr=yerr, capsize=6)
plt.ylabel("RMSE (IC50)")
plt.title("RMSE Comparison with 95% Confidence Intervals")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rmse_comparison_ci.png"), dpi=300)
plt.close()

# ===============================
# 2️⃣ LOG-SCALE TRUE vs PRED
# ===============================
for name, res in results.items():
    plt.figure(figsize=(6, 6))
    plt.scatter(res["y_true"], res["y_pred"], s=5, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True IC50")
    plt.ylabel("Predicted IC50")
    plt.title(name)

    min_v = min(res["y_true"].min(), res["y_pred"].min())
    max_v = max(res["y_true"].max(), res["y_pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], "--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"scatter_log_{name.replace(' ', '_')}.png"),
        dpi=300,
    )
    plt.close()

# ===============================
# 3️⃣ ERROR DISTRIBUTIONS
# ===============================
plt.figure(figsize=(8, 6))
for name, res in results.items():
    errors = np.abs(res["y_true"] - res["y_pred"])
    plt.hist(errors, bins=100, density=True, alpha=0.5, label=name)

plt.xlabel("|Prediction Error| (IC50)")
plt.ylabel("Density")
plt.title("Absolute Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "error_distribution.png"), dpi=300)
plt.close()

# ===============================
# 4️⃣ R² COMPARISON
# ===============================
plt.figure(figsize=(8, 5))
plt.bar(labels, [results[m]["r2"] for m in labels])
plt.ylabel("R² Score")
plt.title("R² Comparison Across Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "r2_comparison.png"), dpi=300)
plt.close()

print("✅ All plots generated successfully in results/plots/")

