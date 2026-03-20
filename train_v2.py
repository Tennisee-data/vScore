"""
vScore training pipeline v2 — uses batch-extracted features.

Loads features from the manifest (produced by extract_batch.py),
trains valence heads, runs cross-domain transfer experiments.

Works with any number of videos — 100 or 10,000.

Run: python -m vScore.train_v2
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# ── Configuration ──────────────────────────────────────────────

FEATURES_DIR = Path("vScore/.features")
MANIFEST_PATH = FEATURES_DIR / "manifest.json"
ENCODER_DIM = 1024

AXIS_NAMES = ["speed", "impact", "precision", "verticality", "coordination", "tension"]

# Proxy scores for ALL known categories
# axes: [speed, impact, precision, verticality, coordination, tension]
PROXY_SCORES = {
    # Kinetics-mini
    "archery":      [3.0, 2.0, 9.0, 1.0, 1.0, 8.0],
    "bowling":      [5.0, 8.0, 7.0, 0.5, 1.0, 6.0],
    "flying_kite":  [4.0, 0.5, 3.0, 7.0, 1.0, 1.0],
    "high_jump":    [7.0, 4.0, 6.0, 9.0, 1.0, 7.0],
    "marching":     [3.0, 1.0, 5.0, 0.5, 9.0, 1.0],
    # HACS categories (add as downloaded)
    "Arm_wrestling":       [2.0, 6.0, 5.0, 0.5, 2.0, 9.0],
    "Doing_fencing":       [7.0, 4.0, 9.0, 1.0, 2.0, 8.0],
    "Fixing_the_roof":     [2.0, 3.0, 5.0, 5.0, 1.0, 4.0],
    "Javelin_throw":       [8.0, 3.0, 8.0, 6.0, 1.0, 8.0],
    "Pole_vault":          [7.0, 3.0, 8.0, 9.0, 1.0, 9.0],
    "Slacklining":         [2.0, 1.0, 8.0, 3.0, 1.0, 7.0],
    "Snowboarding":        [8.0, 4.0, 6.0, 4.0, 1.0, 4.0],
    "Springboard_diving":  [5.0, 5.0, 8.0, 9.0, 1.0, 7.0],
}


def load_dataset():
    """Load all cached features and pair with proxy scores."""
    if not MANIFEST_PATH.exists():
        print("  No manifest found. Run: python -m vScore.extract_batch")
        return None

    manifest = json.loads(MANIFEST_PATH.read_text())

    features = []
    scores = []
    categories = []
    video_ids = []
    sources = []
    skipped = []

    for video_id, info in manifest.items():
        # Extract category from either format
        if isinstance(info, dict):
            category = info.get("category", video_id.split("/")[0])
        else:
            category = video_id.split("/")[0]

        if category not in PROXY_SCORES:
            skipped.append((video_id, category))
            continue

        # Handle both old format (string path) and new format (dict with feature_path)
        if isinstance(info, str):
            feat_path = info
        elif isinstance(info, dict):
            feat_path = info.get("feature_path", "")
        else:
            continue

        feat = torch.load(feat_path, weights_only=True)
        score = torch.tensor(PROXY_SCORES[category], dtype=torch.float32)

        features.append(feat)
        scores.append(score)
        categories.append(category)
        video_ids.append(video_id)
        sources.append(info.get("source", "unknown") if isinstance(info, dict) else "unknown")

    if skipped:
        skip_cats = set(c for _, c in skipped)
        print(f"  Skipped {len(skipped)} videos from categories without proxy scores: {skip_cats}")

    return {
        "features": torch.stack(features),
        "scores": torch.stack(scores),
        "categories": categories,
        "video_ids": video_ids,
        "sources": sources,
    }


def train_valence_head(dataset, holdout_category=None, n_epochs=500, lr=1e-3):
    """Train a valence head. Optionally hold out a category for transfer testing."""
    n_axes = len(AXIS_NAMES)

    if holdout_category:
        train_idx = [i for i, c in enumerate(dataset["categories"]) if c != holdout_category]
        test_idx = [i for i, c in enumerate(dataset["categories"]) if c == holdout_category]
    else:
        n = len(dataset["categories"])
        perm = torch.randperm(n).tolist()
        split = int(0.8 * n)
        train_idx = perm[:split]
        test_idx = perm[split:]

    if not train_idx or not test_idx:
        return None, train_idx, test_idx

    X_train = dataset["features"][train_idx]
    y_train = dataset["scores"][train_idx]
    X_test = dataset["features"][test_idx]
    y_test = dataset["scores"][test_idx]

    head = nn.Sequential(
        nn.Linear(ENCODER_DIM, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, n_axes),
        nn.ReLU(),
    )

    optimizer = optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    head.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = head(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    # Final evaluation
    head.eval()
    with torch.no_grad():
        train_loss = loss_fn(head(X_train), y_train).item()
        test_loss = loss_fn(head(X_test), y_test).item()
        preds = head(X_test)
        overall_mae = (preds - y_test).abs().mean().item()
        per_axis_mae = (preds - y_test).abs().mean(dim=0)

    return {
        "head": head,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "overall_mae": overall_mae,
        "per_axis_mae": per_axis_mae,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }, train_idx, test_idx


def main():
    print("=" * 70)
    print("vScore Training Pipeline v2")
    print("=" * 70)

    # Load dataset
    print("\n── Loading cached features ──")
    dataset = load_dataset()
    if dataset is None:
        return

    n = len(dataset["categories"])
    categories = sorted(set(dataset["categories"]))
    sources = sorted(set(dataset["sources"]))

    print(f"  {n} videos, {len(categories)} categories, {len(sources)} sources")
    print(f"  Categories: {categories}")
    print(f"  Sources: {sources}")
    for cat in categories:
        count = sum(1 for c in dataset["categories"] if c == cat)
        print(f"    {cat}: {count} videos")

    # Train (random split)
    print(f"\n── Training: random 80/20 split ──")
    result, _, test_idx = train_valence_head(dataset)
    if result:
        print(f"  Train: {result['n_train']}  Test: {result['n_test']}")
        print(f"  Train loss: {result['train_loss']:.4f}  Test loss: {result['test_loss']:.4f}")
        print(f"  Overall MAE: {result['overall_mae']:.3f}")
        print(f"  Per-axis MAE:", end="")
        for name, v in zip(AXIS_NAMES, result["per_axis_mae"]):
            print(f"  {name}={v:.2f}", end="")
        print()

    # Cross-domain transfer
    print(f"\n{'=' * 70}")
    print("CROSS-DOMAIN TRANSFER")
    print(f"{'=' * 70}")

    print(f"\n  {'holdout':>15s}  {'n_train':>7s}  {'n_test':>6s}  {'MAE':>5s}", end="")
    for ax in AXIS_NAMES:
        print(f"  {ax[:5]:>5s}", end="")
    print()
    print(f"  {'─'*15}  {'─'*7}  {'─'*6}  {'─'*5}", end="")
    for _ in AXIS_NAMES:
        print(f"  {'─'*5}", end="")
    print()

    transfer_results = {}
    for cat in categories:
        result, _, _ = train_valence_head(dataset, holdout_category=cat)
        if result:
            transfer_results[cat] = result
            print(f"  {cat:>15s}  {result['n_train']:7d}  {result['n_test']:6d}  "
                  f"{result['overall_mae']:5.2f}", end="")
            for v in result["per_axis_mae"]:
                marker = "*" if v < 2.0 else " "
                print(f"  {v:4.2f}{marker}", end="")
            print()

    # Summary
    print(f"\n  * = transfers well (MAE < 2.0)")

    # Which axes are universal?
    print(f"\n── Universal vs domain-specific axes ──")
    for i, ax in enumerate(AXIS_NAMES):
        maes = [r["per_axis_mae"][i].item() for r in transfer_results.values()]
        mean_mae = sum(maes) / len(maes)
        transfers = sum(1 for m in maes if m < 2.0)
        print(f"  {ax:>15s}  mean_MAE={mean_mae:.2f}  transfers_in={transfers}/{len(maes)}")


if __name__ == "__main__":
    main()
