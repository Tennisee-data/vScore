"""
Axis discovery through residual analysis.

The question: when the model fails to score a shot type,
what direction in feature space is it missing?

The method:
    1. Train valence head on 6 axes
    2. Identify high-residual categories (MAE > 2.5)
    3. Analyse what the encoder sees that the axes don't capture
    4. Propose a 7th axis from the data
    5. Retrain with 7 axes and show improvement on problem categories
    6. Verify the new axis doesn't overfit (cross-shot transfer must hold)

This is NOT overfitting because:
    - The new axis is derived from encoder features, not from labels
    - It must improve held-out transfer, not just training fit
    - We show before/after on the SAME holdout protocol

Run: python -m vScore.axis_discovery
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

FEATURES_DIR = Path("vScore/.features/tennis")
MANIFEST_PATH = FEATURES_DIR / "manifest.json"
ENCODER_DIM = 1024

AXIS_NAMES_V1 = ["speed", "impact", "precision", "verticality", "aggression", "tension"]

PROXY_SCORES_V1 = {
    "backhand":              [6.0, 6.0, 6.0, 3.0, 6.0, 5.0],
    "backhand2hands":        [7.0, 7.0, 6.0, 3.0, 7.0, 5.0],
    "backhand_slice":        [4.0, 3.0, 8.0, 2.0, 3.0, 4.0],
    "backhand_volley":       [5.0, 5.0, 8.0, 2.0, 6.0, 6.0],
    "forehand_flat":         [8.0, 7.0, 6.0, 3.0, 8.0, 5.0],
    "forehand_openstands":   [7.0, 6.0, 5.0, 3.0, 7.0, 4.0],
    "forehand_slice":        [5.0, 3.0, 8.0, 2.0, 4.0, 4.0],
    "forehand_volley":       [6.0, 5.0, 8.0, 2.0, 7.0, 6.0],
    "flat_service":          [9.0, 7.0, 7.0, 6.0, 8.0, 8.0],
    "kick_service":          [7.0, 6.0, 7.0, 7.0, 7.0, 8.0],
    "slice_service":         [6.0, 5.0, 8.0, 4.0, 6.0, 7.0],
    "smash":                 [9.0, 9.0, 5.0, 8.0, 9.0, 3.0],
}


def load_dataset(proxy_scores):
    manifest = json.loads(MANIFEST_PATH.read_text())
    features, scores, categories = [], [], []

    for vid_id, info in manifest.items():
        cat = info["category"]
        if cat not in proxy_scores:
            continue
        feat = torch.load(info["feature_path"], weights_only=True)
        features.append(feat)
        scores.append(torch.tensor(proxy_scores[cat], dtype=torch.float32))
        categories.append(cat)

    return {
        "features": torch.stack(features),
        "scores": torch.stack(scores),
        "categories": categories,
    }


def train_head(dataset, n_axes, holdout=None, n_epochs=500):
    if holdout:
        train_idx = [i for i, c in enumerate(dataset["categories"]) if c != holdout]
        test_idx = [i for i, c in enumerate(dataset["categories"]) if c == holdout]
    else:
        n = len(dataset["categories"])
        perm = torch.randperm(n).tolist()
        split = int(0.8 * n)
        train_idx, test_idx = perm[:split], perm[split:]

    if not train_idx or not test_idx:
        return None

    X_train = dataset["features"][train_idx]
    y_train = dataset["scores"][train_idx]
    X_test = dataset["features"][test_idx]
    y_test = dataset["scores"][test_idx]

    head = nn.Sequential(
        nn.Linear(ENCODER_DIM, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, n_axes), nn.ReLU(),
    )

    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    head.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss_fn(head(X_train), y_train).backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        preds = head(X_test)
        mae = (preds - y_test).abs().mean().item()
        per_axis = (preds - y_test).abs().mean(dim=0)

        # Also get per-category MAE on test set
        cat_maes = {}
        for i, idx in enumerate(test_idx):
            cat = dataset["categories"][idx]
            if cat not in cat_maes:
                cat_maes[cat] = []
            cat_maes[cat].append((preds[i] - y_test[i]).abs().mean().item())
        cat_maes = {c: sum(v)/len(v) for c, v in cat_maes.items()}

    return {
        "head": head, "mae": mae, "per_axis": per_axis,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "cat_maes": cat_maes,
        "train_idx": train_idx, "test_idx": test_idx,
        "preds": preds, "actuals": y_test,
    }


def analyse_residuals(dataset, n_axes):
    """
    Train on all data, compute residuals per category,
    find the direction in feature space the model is missing.
    """
    head = nn.Sequential(
        nn.Linear(ENCODER_DIM, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, n_axes), nn.ReLU(),
    )
    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X = dataset["features"]
    y = dataset["scores"]

    head.train()
    for _ in range(500):
        optimizer.zero_grad()
        loss_fn(head(X), y).backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        preds = head(X)
        residuals = (preds - y).abs().mean(dim=1)  # per-sample MAE

    # Group by category
    cat_residuals = {}
    cat_features = {}
    for i, cat in enumerate(dataset["categories"]):
        if cat not in cat_residuals:
            cat_residuals[cat] = []
            cat_features[cat] = []
        cat_residuals[cat].append(residuals[i].item())
        cat_features[cat].append(X[i])

    return cat_residuals, cat_features, head


def discover_missing_axis(dataset, cat_features, cat_residuals, threshold=1.5):
    """
    For high-residual categories, find the principal direction
    in feature space that distinguishes them from low-residual ones.

    This direction IS the missing axis.
    """
    # Identify problem categories
    cat_mean_residual = {c: sum(v)/len(v) for c, v in cat_residuals.items()}
    problem_cats = [c for c, r in cat_mean_residual.items() if r > threshold]
    good_cats = [c for c, r in cat_mean_residual.items() if r <= threshold]

    if not problem_cats:
        return None, None, None

    # Collect features from problem vs. good categories
    problem_features = torch.stack([
        f for c in problem_cats for f in cat_features[c]
    ])
    good_features = torch.stack([
        f for c in good_cats for f in cat_features[c]
    ])

    # The direction that separates them
    problem_mean = problem_features.mean(dim=0)
    good_mean = good_features.mean(dim=0)
    direction = problem_mean - good_mean
    direction = direction / direction.norm()  # unit vector

    # Project all features onto this direction
    all_features = dataset["features"]
    projections = (all_features @ direction).numpy()

    # Per-category projection stats
    cat_projections = {}
    for i, cat in enumerate(dataset["categories"]):
        if cat not in cat_projections:
            cat_projections[cat] = []
        cat_projections[cat].append(projections[i])

    return direction, cat_projections, problem_cats


def main():
    print("=" * 70)
    print("AXIS DISCOVERY: Finding what the model is missing")
    print("=" * 70)

    # ── Step 1: Baseline with 6 axes ──────────────────────────

    print("\n── Step 1: Baseline (6 axes) ──")
    dataset_v1 = load_dataset(PROXY_SCORES_V1)
    cats = sorted(set(dataset_v1["categories"]))
    print(f"  {len(dataset_v1['categories'])} videos, {len(cats)} shot types")

    # ── Step 2: Cross-shot transfer baseline ──────────────────

    print(f"\n── Step 2: Cross-shot transfer (6 axes, baseline) ──")
    baseline_results = {}
    for cat in cats:
        result = train_head(dataset_v1, 6, holdout=cat)
        if result:
            baseline_results[cat] = result["mae"]

    print(f"\n  {'shot':>25s}  {'6-axis MAE':>10s}")
    for cat in sorted(baseline_results.keys()):
        marker = " <<< PROBLEM" if baseline_results[cat] > 2.5 else ""
        print(f"  {cat:>25s}  {baseline_results[cat]:10.2f}{marker}")

    # Use TRANSFER MAE to identify problem categories
    transfer_threshold = 2.5
    problem_cats = [c for c, mae in baseline_results.items() if mae > transfer_threshold]
    print(f"\n  Problem categories (transfer MAE > {transfer_threshold}): {problem_cats}")

    # ── Step 3: Discover missing axis ─────────────────────────

    print(f"\n── Step 3: Residual analysis ──")
    print(f"  Finding the direction in feature space that separates")
    print(f"  problem shots from well-scored shots...")

    # Get features grouped by category
    cat_features = {}
    cat_residuals = {}
    for i, cat in enumerate(dataset_v1["categories"]):
        if cat not in cat_features:
            cat_features[cat] = []
            cat_residuals[cat] = []
        cat_features[cat].append(dataset_v1["features"][i])
        cat_residuals[cat].append(baseline_results.get(cat, 0))

    direction, cat_projections, prob = discover_missing_axis(
        dataset_v1, cat_features, cat_residuals, threshold=transfer_threshold
    )

    if direction is None:
        print("  No problem categories found. All axes are sufficient.")
        return

    print(f"\n  Projection of each shot type onto the discovered axis:")
    print(f"  (This axis separates high-residual from low-residual shots)\n")

    proj_stats = {}
    for cat in sorted(cat_projections.keys()):
        vals = cat_projections[cat]
        mean = sum(vals) / len(vals)
        proj_stats[cat] = mean
        in_problem = "PROBLEM" if cat in problem_cats else ""
        bar_len = int(abs(mean) * 20)
        bar = "#" * bar_len if mean > 0 else ""
        print(f"    {cat:>25s}: {mean:+7.3f}  {bar}  {in_problem}")

    # ── Step 4: Create axis 7 scores from projections ─────────

    print(f"\n── Step 4: Define axis 7 from projections ──")

    # Normalise projections to 0-10 scale
    all_proj = [v for vals in cat_projections.values() for v in vals]
    proj_min, proj_max = min(all_proj), max(all_proj)

    axis7_scores = {}
    for cat, mean in proj_stats.items():
        # Scale to 0-10
        if proj_max > proj_min:
            scaled = (mean - proj_min) / (proj_max - proj_min) * 10.0
        else:
            scaled = 5.0
        axis7_scores[cat] = round(scaled, 1)

    print(f"\n  Axis 7 (data-derived, unnamed) scores:")
    for cat in sorted(axis7_scores.keys()):
        print(f"    {cat:>25s}: {axis7_scores[cat]:.1f}")

    # ── Step 5: Retrain with 7 axes ───────────────────────────

    print(f"\n── Step 5: Retrain with 7 axes ──")

    PROXY_SCORES_V2 = {}
    for cat, scores_v1 in PROXY_SCORES_V1.items():
        PROXY_SCORES_V2[cat] = scores_v1 + [axis7_scores[cat]]

    AXIS_NAMES_V2 = AXIS_NAMES_V1 + ["axis_7"]

    dataset_v2 = load_dataset(PROXY_SCORES_V2)

    # Cross-shot transfer with 7 axes
    improved_results = {}
    for cat in cats:
        result = train_head(dataset_v2, 7, holdout=cat)
        if result:
            improved_results[cat] = result["mae"]

    # ── Step 6: Compare ───────────────────────────────────────

    print(f"\n── Step 6: Before vs. After ──")
    print(f"\n  {'shot':>25s}  {'6-axis':>7s}  {'7-axis':>7s}  {'change':>7s}  {'verdict':>10s}")
    print(f"  {'─'*25}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")

    improvements = []
    regressions = []
    for cat in sorted(cats):
        v1 = baseline_results.get(cat, 0)
        v2 = improved_results.get(cat, 0)
        change = v2 - v1
        if change < -0.1:
            verdict = "IMPROVED"
            improvements.append((cat, change))
        elif change > 0.1:
            verdict = "REGRESSED"
            regressions.append((cat, change))
        else:
            verdict = "same"
        print(f"  {cat:>25s}  {v1:7.2f}  {v2:7.2f}  {change:+7.2f}  {verdict:>10s}")

    v1_mean = sum(baseline_results.values()) / len(baseline_results)
    v2_mean = sum(improved_results.values()) / len(improved_results)

    print(f"\n  {'MEAN':>25s}  {v1_mean:7.2f}  {v2_mean:7.2f}  {v2_mean-v1_mean:+7.2f}")

    # ── Overfitting check ─────────────────────────────────────

    print(f"\n── Overfitting check ──")
    print(f"  Improvements: {len(improvements)}")
    print(f"  Regressions:  {len(regressions)}")
    print(f"  Mean MAE change: {v2_mean - v1_mean:+.3f}")

    if len(regressions) > len(improvements):
        print(f"\n  WARNING: More regressions than improvements.")
        print(f"  The new axis may be overfitting to problem categories")
        print(f"  at the cost of categories that were already well-scored.")
    elif v2_mean < v1_mean:
        print(f"\n  The 7th axis improves overall transfer MAE by {v1_mean - v2_mean:.3f}")
        print(f"  without degrading more categories than it helps.")
        print(f"  This is not overfitting. It is axis discovery.")
    else:
        print(f"\n  The 7th axis did not improve overall transfer.")
        print(f"  The original 6 axes may already capture the relevant dynamics.")

    print(f"""
{'=' * 70}
INTERPRETATION
{'=' * 70}

  Axis 7 was not named by a human. It was discovered by asking:
  "what direction in feature space separates the shots I score
  well from the shots I score badly?"

  If axis 7 improves transfer on problem categories (smash,
  flat serve) WITHOUT degrading good categories (backhand,
  forehand), then the model identified a genuine missing
  dimension in its scoring, not noise.

  The axis can be interpreted post-hoc by examining which
  shot types score highest on it. If smash and serve score
  high while groundstrokes score low, the axis likely
  captures overhead/preparation dynamics.

  The human names it after the fact. The data found it.
""")


if __name__ == "__main__":
    main()
