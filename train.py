"""
vScore training pipeline.

Step 1: Extract V-JEPA 2 features from all videos (cached to disk)
Step 2: Assign proxy valence scores based on visual dynamics
Step 3: Train valence heads
Step 4: Test cross-domain transfer

Domain: "dynamics" — universal motion/outcome axes
    speed        : how fast things move in the scene
    impact       : collision / force transfer patterns
    precision    : controlled vs chaotic motion
    verticality  : up/down movement dominance
    coordination : synchronized multi-agent motion
    tension      : buildup before release/outcome

These axes are pre-linguistic. A baby recognizes speed and impact
before it knows the words. The model should too.

Run: python -m vScore.train
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Configuration ──────────────────────────────────────────────

CACHE_DIR = Path("vScore/.cache")
FEATURES_DIR = Path("vScore/.features")
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES = 64
ENCODER_DIM = 1024

BASE_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val"

VIDEOS = {
    "archery": [
        "-Qz25rXdMjE_000014_000024.mp4",
        "-UJgyiWe500_000029_000039.mp4",
        "0S-P4lr_c7s_000022_000032.mp4",
        "27jJKXiB9Y8_000009_000019.mp4",
        "2x1lIrgKxYo_000589_000599.mp4",
        "36E29x22tnQ_000018_000028.mp4",
        "3and4vWkW4s_000011_000021.mp4",
        "3hoSk280ndk_000025_000035.mp4",
        "4hxnP0stMN0_000003_000013.mp4",
        "4wXnaqEDh3c_000014_000024.mp4",
    ],
    "bowling": [
        "--dVV4_CSvw_000033_000043.mp4",
        "-WH-lxmGJVY_000005_000015.mp4",
        "-jOClYqKtE8_000003_000013.mp4",
        "1W7HNDBA4pA_000002_000012.mp4",
        "4JxH3S5JwMs_000003_000013.mp4",
        "5NLQMrXzCQA_000003_000013.mp4",
        "5P0Szq9VDYg_000021_000031.mp4",
        "5Vu8HJ__eMg_000277_000287.mp4",
        "6V6DgC9u7Y0_000000_000010.mp4",
        "8TiYnWFX-ow_000042_000052.mp4",
    ],
    "flying_kite": [
        "07xOT83TIG4_000040_000050.mp4",
        "0QH8uFjXiW4_000003_000013.mp4",
        "0huAN2gC6Pc_000143_000153.mp4",
        "0yNXOIqJLtA_000012_000022.mp4",
        "1K1b8Zphi3U_000009_000019.mp4",
        "1elGm6gzpwI_000000_000010.mp4",
        "29aSgwRL_7Q_000005_000015.mp4",
        "3dnRiA5pSwU_000024_000034.mp4",
        "3nvWUCaMHIs_000024_000034.mp4",
        "5BWqkXc8r90_000007_000017.mp4",
    ],
    "high_jump": [
        "01fAWEHzudA_000002_000012.mp4",
        "0oL36GHlSXw_000022_000032.mp4",
        "3sBYgcb4bEY_000003_000013.mp4",
        "4DP5vsyAg1c_000003_000013.mp4",
        "4Zcjoek-1-4_000003_000013.mp4",
        "5RIe5niLskU_000004_000014.mp4",
        "5gVK5JsNRSc_000005_000015.mp4",
        "5jgye7jxJtY_000027_000037.mp4",
        "6VTvoRhQCxU_000003_000013.mp4",
        "8K6x4rDOJEc_000001_000011.mp4",
    ],
    "marching": [
        "-0IErS_cisg_000017_000027.mp4",
        "1m-Kdky1y84_000022_000032.mp4",
        "5EVgOrjJjuM_000160_000170.mp4",
        "5SjV5j8f6rw_000009_000019.mp4",
        "6ofkyLo6dns_000102_000112.mp4",
        "8CqVi5Eb0cw_000130_000140.mp4",
        "8HlGfWcUWlY_000075_000085.mp4",
        "9Xc_Kf9gYQU_000451_000461.mp4",
        "9qbiWvlGeSQ_000023_000033.mp4",
        "AYslYdNVsMU_000272_000282.mp4",
    ],
}

# ── Proxy valence scores ──────────────────────────────────────
# These approximate what a human annotator would score.
# axes: [speed, impact, precision, verticality, coordination, tension]
# All 0-10, zero = nothing happening.

PROXY_SCORES = {
    #            speed  impact  prec  vert  coord  tension
    "archery":     [3.0,   2.0,  9.0,  1.0,  1.0,   8.0],  # Slow, precise, high tension before release
    "bowling":     [5.0,   8.0,  7.0,  0.5,  1.0,   6.0],  # Medium speed, high impact, precise
    "flying_kite": [4.0,   0.5,  3.0,  7.0,  1.0,   1.0],  # Aerial, vertical, low tension
    "high_jump":   [7.0,   4.0,  6.0,  9.0,  1.0,   7.0],  # Fast, very vertical, tension buildup
    "marching":    [3.0,   1.0,  5.0,  0.5,  9.0,   1.0],  # Slow, synchronized, no impact
}

AXIS_NAMES = ["speed", "impact", "precision", "verticality", "coordination", "tension"]


# ── Step 1: Feature extraction ────────────────────────────────

def download_video(category: str, filename: str) -> Path:
    import urllib.request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / filename
    if not local_path.exists():
        url = f"{BASE_URL}/{category}/{filename}"
        urllib.request.urlretrieve(url, local_path)
    return local_path


def decode_video(path: Path, n_frames: int = 64) -> np.ndarray:
    import av
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total = len(frames)
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.pad(indices, (0, n_frames - total), mode="wrap")

    return np.stack([frames[i] for i in indices])  # (T, H, W, 3)


def extract_all_features():
    """Extract and cache V-JEPA 2 features for all videos."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already done
    manifest_path = FEATURES_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) == sum(len(v) for v in VIDEOS.values()):
            print(f"Features already cached ({len(manifest)} videos)")
            return manifest

    from transformers import AutoVideoProcessor, AutoModel

    print(f"Loading encoder: {HF_REPO}")
    model = AutoModel.from_pretrained(HF_REPO)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model.eval()
    print(f"  Loaded on {model.device}")

    manifest = {}
    total = sum(len(v) for v in VIDEOS.values())
    done = 0

    for category, filenames in VIDEOS.items():
        for fname in filenames:
            video_id = f"{category}/{fname}"
            feat_path = FEATURES_DIR / f"{category}_{fname}.pt"

            if feat_path.exists():
                manifest[video_id] = str(feat_path)
                done += 1
                print(f"  [{done}/{total}] cached: {video_id}")
                continue

            # Download
            local = download_video(category, fname)

            # Decode
            video_array = decode_video(local, N_FRAMES)

            # Process
            inputs = processor(list(video_array), return_tensors="pt")

            # Extract features
            with torch.no_grad():
                features = model.get_vision_features(**inputs)

            # Pool to global feature: (1, 8192, 1024) → (1024,)
            pooled = features.squeeze(0).mean(dim=0)

            # Save
            torch.save(pooled, feat_path)
            manifest[video_id] = str(feat_path)
            done += 1
            print(f"  [{done}/{total}] extracted: {video_id}  shape={list(pooled.shape)}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nAll features cached: {len(manifest)} videos")
    return manifest


# ── Step 2: Build dataset ─────────────────────────────────────

def build_dataset(manifest: dict):
    """Load cached features and pair with proxy valence scores."""
    features = []
    scores = []
    categories = []
    video_ids = []

    for video_id, feat_path in manifest.items():
        category = video_id.split("/")[0]
        feat = torch.load(feat_path, weights_only=True)
        score = torch.tensor(PROXY_SCORES[category], dtype=torch.float32)

        features.append(feat)
        scores.append(score)
        categories.append(category)
        video_ids.append(video_id)

    return {
        "features": torch.stack(features),       # (N, 1024)
        "scores": torch.stack(scores),            # (N, 6)
        "categories": categories,
        "video_ids": video_ids,
    }


# ── Step 3: Train ─────────────────────────────────────────────

def train_valence_head(dataset: dict, holdout_category: str = None):
    """
    Train a valence head to predict proxy scores from V-JEPA features.

    If holdout_category is set, exclude it from training and use it
    for cross-domain transfer testing.
    """
    n_axes = len(AXIS_NAMES)

    # Split by category
    if holdout_category:
        train_mask = [c != holdout_category for c in dataset["categories"]]
        test_mask = [c == holdout_category for c in dataset["categories"]]
        train_idx = [i for i, m in enumerate(train_mask) if m]
        test_idx = [i for i, m in enumerate(test_mask) if m]
    else:
        # Random 80/20 split
        n = len(dataset["categories"])
        perm = torch.randperm(n).tolist()
        split = int(0.8 * n)
        train_idx = perm[:split]
        test_idx = perm[split:]

    X_train = dataset["features"][train_idx]
    y_train = dataset["scores"][train_idx]
    X_test = dataset["features"][test_idx]
    y_test = dataset["scores"][test_idx]

    # Simple valence head: Linear → GELU → Linear → GELU → Linear → ReLU
    head = nn.Sequential(
        nn.Linear(ENCODER_DIM, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, n_axes),
        nn.ReLU(),
    )

    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    head.train()
    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = head(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            head.eval()
            with torch.no_grad():
                train_loss = loss_fn(head(X_train), y_train).item()
                test_loss = loss_fn(head(X_test), y_test).item()
            head.train()
            print(f"  epoch {epoch+1:4d}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

    return head, train_idx, test_idx


def evaluate(head, dataset, test_idx, label=""):
    """Show predicted vs actual scores — no words, just numbers."""
    head.eval()

    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")

    header = f"  {'category':>14s} │"
    for ax in AXIS_NAMES:
        header += f" {ax[:5]:>5s}"
    header += "  │"
    for ax in AXIS_NAMES:
        header += f" {ax[:5]:>5s}"
    header += "  │  MAE"
    print(header)
    print(f"  {'':>14s} │{'  ACTUAL':^31s}  │{'  PREDICTED':^31s}  │")
    print(f"  {'─' * 14}─┼{'─' * 32}─┼{'─' * 32}─┼{'─' * 6}")

    with torch.no_grad():
        for i in test_idx:
            feat = dataset["features"][i].unsqueeze(0)
            pred = head(feat).squeeze(0)
            actual = dataset["scores"][i]
            cat = dataset["categories"][i]
            mae = (pred - actual).abs().mean().item()

            row = f"  {cat:>14s} │"
            for v in actual:
                row += f" {v.item():5.1f}"
            row += "  │"
            for v in pred:
                row += f" {v.item():5.1f}"
            row += f"  │ {mae:5.2f}"
            print(row)

    # Overall MAE
    X_test = dataset["features"][test_idx]
    y_test = dataset["scores"][test_idx]
    with torch.no_grad():
        preds = head(X_test)
        overall_mae = (preds - y_test).abs().mean().item()
        per_axis_mae = (preds - y_test).abs().mean(dim=0)

    print(f"\n  Overall MAE: {overall_mae:.3f}")
    print(f"  Per-axis MAE:", end="")
    for name, v in zip(AXIS_NAMES, per_axis_mae):
        print(f"  {name}={v:.2f}", end="")
    print()


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("vScore Training Pipeline")
    print("=" * 70)
    print(f"Domain: dynamics")
    print(f"Axes:   {AXIS_NAMES}")
    print(f"Videos: {sum(len(v) for v in VIDEOS.values())} across {len(VIDEOS)} categories")
    print()

    # Step 1: Extract features
    print("── Step 1: Feature extraction (V-JEPA 2) ──")
    t0 = time.time()
    manifest = extract_all_features()
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Step 2: Build dataset
    print("── Step 2: Build dataset ──")
    dataset = build_dataset(manifest)
    print(f"  {len(dataset['categories'])} samples, {len(AXIS_NAMES)} axes")
    print(f"  Feature dim: {dataset['features'].shape[1]}")
    print()

    # Step 3: Train (random split)
    print("── Step 3: Train valence head (random 80/20 split) ──")
    head, train_idx, test_idx = train_valence_head(dataset)
    evaluate(head, dataset, test_idx, "Random split — test set")

    # Step 4: Cross-domain transfer
    # Train on 4 categories, test on the held-out one
    print("\n\n" + "=" * 70)
    print("CROSS-DOMAIN TRANSFER TEST")
    print("Can the model score a domain it has NEVER seen?")
    print("=" * 70)

    for holdout in VIDEOS.keys():
        print(f"\n── Holdout: {holdout} (train on everything else) ──")
        head_xd, train_idx_xd, test_idx_xd = train_valence_head(
            dataset, holdout_category=holdout
        )
        evaluate(head_xd, dataset, test_idx_xd,
                 f"Transfer: trained WITHOUT {holdout}, tested ON {holdout}")

    # Final summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  The valence head maps 1024-dim V-JEPA features to 6 valence axes.
  No words were used anywhere in the pipeline.

  Cross-domain transfer works if the held-out category scores
  have low MAE — meaning the model learned visual dynamics
  (speed, impact, tension) not category labels.

  If MAE is high for a held-out category, that category has
  unique visual primitives not shared with the training set.
  That itself is information: it identifies truly novel domains.

  Next: replace proxy scores with real annotations.
  The architecture is domain-agnostic. Only the scores change.
""")


if __name__ == "__main__":
    main()
