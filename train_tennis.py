"""
Tennis experiment — THETIS dataset.

Extracts V-JEPA 2 features from THETIS tennis shots, trains a
valence head, and tests whether visually similar shots cluster
together in valence space without being told they're related.

The test:
    - Do forehand variants cluster together?
    - Do serves cluster separately from groundstrokes?
    - Do volleys form their own group?
    - Can we hold out an entire shot type and predict its scores?

Run: python -m vScore.train_tennis

Requires: THETIS dataset cloned to ../THETIS-dataset/
"""

import json
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Configuration ──────────────────────────────────────────────

THETIS_DIR = Path("THETIS-dataset")
CACHE_DIR = Path("vScore/.cache/tennis")
FEATURES_DIR = Path("vScore/.features/tennis")
MANIFEST_PATH = FEATURES_DIR / "manifest.json"
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES = 64
ENCODER_DIM = 1024

AXIS_NAMES = ["speed", "impact", "precision", "verticality", "aggression", "tension"]

# Map THETIS directory names to proxy scores
# axes: [speed, impact, precision, verticality, aggression, tension]
# THETIS actual directory names → proxy scores
# axes: [speed, impact, precision, verticality, aggression, tension]
PROXY_SCORES = {
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

# Shot families for clustering analysis
SHOT_FAMILIES = {
    "forehands": ["forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley"],
    "backhands": ["backhand", "backhand2hands", "backhand_slice", "backhand_volley"],
    "serves":    ["flat_service", "kick_service", "slice_service"],
    "net_play":  ["forehand_volley", "backhand_volley", "smash"],
}


# ── Discovery ─────────────────────────────────────────────────

def discover_thetis_videos(max_per_category: int = 15) -> list[dict]:
    """Find RGB video files in THETIS dataset."""
    # THETIS structure: VIDEO_RGB/{action}/*.avi
    rgb_dir = THETIS_DIR / "VIDEO_RGB"

    if not rgb_dir.exists():
        print(f"  VIDEO_RGB not found at {rgb_dir}")
        print(f"  Contents of {THETIS_DIR}:")
        if THETIS_DIR.exists():
            for p in sorted(THETIS_DIR.iterdir())[:20]:
                print(f"    {p.name} ({'dir' if p.is_dir() else 'file'})")
        return []

    videos = []
    for category in PROXY_SCORES.keys():
        cat_dir = rgb_dir / category
        if not cat_dir.exists():
            print(f"  Warning: {cat_dir} not found")
            continue

        cat_videos = sorted(cat_dir.glob("*.avi"))

        for v in cat_videos[:max_per_category]:
            videos.append({
                "source": "thetis",
                "category": category,
                "filename": v.name,
                "path": str(v),
                "video_id": f"tennis/{category}/{v.name}",
            })

    return videos


def decode_video(path: str, n_frames: int = 64) -> np.ndarray:
    """Decode video to numpy array."""
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames in {path}")

    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.pad(indices, (0, n_frames - total), mode="wrap")

    return np.stack([frames[i] for i in indices])


def extract_features(videos: list[dict]) -> dict:
    """Extract and cache V-JEPA 2 features."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    to_process = [v for v in videos if v["video_id"] not in manifest]

    if not to_process:
        print(f"  All {len(videos)} videos cached.")
        return manifest

    print(f"  {len(to_process)} new videos to extract ({len(manifest)} cached)")
    print(f"  Estimated: ~{len(to_process) * 9 // 60}min {len(to_process) * 9 % 60}s")

    from transformers import AutoVideoProcessor, AutoModel

    print(f"  Loading {HF_REPO}...")
    model = AutoModel.from_pretrained(HF_REPO)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model.eval()

    t0 = time.time()
    errors = []

    for i, video in enumerate(to_process):
        safe_name = video["video_id"].replace("/", "_")
        feat_path = FEATURES_DIR / f"{safe_name}.pt"

        try:
            video_array = decode_video(video["path"], N_FRAMES)
            inputs = processor(list(video_array), return_tensors="pt")

            with torch.no_grad():
                features = model.get_vision_features(**inputs)

            pooled = features.squeeze(0).mean(dim=0)
            torch.save(pooled, feat_path)

            manifest[video["video_id"]] = {
                "feature_path": str(feat_path),
                "category": video["category"],
                "source": "thetis",
            }

            elapsed = time.time() - t0
            remaining = (elapsed / (i + 1)) * (len(to_process) - i - 1)
            print(f"  [{i+1}/{len(to_process)}] {video['category']}/{video['filename']}"
                  f"  ({remaining:.0f}s left)")

        except Exception as e:
            errors.append((video["video_id"], str(e)))
            print(f"  [{i+1}/{len(to_process)}] ERROR: {e}")

        if (i + 1) % 10 == 0:
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  Done: {len(to_process) - len(errors)} extracted, {len(errors)} errors")
    return manifest


def load_dataset(manifest: dict) -> dict:
    """Load features and pair with proxy scores."""
    features, scores, categories, video_ids = [], [], [], []

    for vid_id, info in manifest.items():
        cat = info["category"]
        if cat not in PROXY_SCORES:
            continue
        feat = torch.load(info["feature_path"], weights_only=True)
        features.append(feat)
        scores.append(torch.tensor(PROXY_SCORES[cat], dtype=torch.float32))
        categories.append(cat)
        video_ids.append(vid_id)

    return {
        "features": torch.stack(features),
        "scores": torch.stack(scores),
        "categories": categories,
        "video_ids": video_ids,
    }


def train_head(dataset, holdout=None, n_epochs=500):
    """Train valence head, optionally holding out a category."""
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
        nn.Linear(128, len(AXIS_NAMES)), nn.ReLU(),
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

    return {
        "mae": mae, "per_axis": per_axis,
        "n_train": len(train_idx), "n_test": len(test_idx),
    }


def clustering_analysis(dataset):
    """
    Do visually similar shots cluster in feature space?
    No labels, no valence head — pure encoder features.
    """
    print(f"\n{'=' * 70}")
    print("CLUSTERING: Do similar shots cluster WITHOUT labels?")
    print(f"{'=' * 70}")

    # Compute mean feature vector per category
    cat_features = {}
    for i, cat in enumerate(dataset["categories"]):
        if cat not in cat_features:
            cat_features[cat] = []
        cat_features[cat].append(dataset["features"][i])

    cat_means = {}
    for cat, feats in cat_features.items():
        cat_means[cat] = torch.stack(feats).mean(dim=0)

    # Cosine similarity between all category pairs
    cats = sorted(cat_means.keys())
    print(f"\n  Cosine similarity between shot types (encoder features, no training):\n")

    # Header
    short = {c: c[:6] for c in cats}
    print(f"  {'':>18s}", end="")
    for c in cats:
        print(f" {short[c]:>6s}", end="")
    print()

    sim_matrix = {}
    for c1 in cats:
        print(f"  {c1:>18s}", end="")
        for c2 in cats:
            sim = torch.nn.functional.cosine_similarity(
                cat_means[c1].unsqueeze(0), cat_means[c2].unsqueeze(0)
            ).item()
            sim_matrix[(c1, c2)] = sim
            print(f" {sim:6.3f}", end="")
        print()

    # Family analysis
    print(f"\n  Within-family vs. across-family similarity:\n")
    for family_name, members in SHOT_FAMILIES.items():
        present = [m for m in members if m in cat_means]
        if len(present) < 2:
            continue

        # Within-family similarity
        within = []
        for i, m1 in enumerate(present):
            for m2 in present[i+1:]:
                within.append(sim_matrix[(m1, m2)])

        # Across-family similarity (to non-members)
        across = []
        non_members = [c for c in cats if c not in members]
        for m in present:
            for nm in non_members:
                across.append(sim_matrix[(m, nm)])

        within_mean = sum(within) / len(within) if within else 0
        across_mean = sum(across) / len(across) if across else 0
        ratio = within_mean / across_mean if across_mean > 0 else 0

        print(f"  {family_name:>12s}:  within={within_mean:.3f}  across={across_mean:.3f}"
              f"  ratio={ratio:.2f}  {'CLUSTERS' if ratio > 1.1 else 'mixed'}")


def main():
    print("=" * 70)
    print("vScore Tennis Experiment — THETIS Dataset")
    print("=" * 70)

    # Discover
    print("\n── Discovering THETIS videos ──")
    videos = discover_thetis_videos(max_per_category=15)
    if not videos:
        print("  No videos found. Clone THETIS first:")
        print("  git clone https://github.com/THETIS-dataset/dataset.git THETIS-dataset")
        return

    by_cat = {}
    for v in videos:
        by_cat[v["category"]] = by_cat.get(v["category"], 0) + 1
    print(f"  Found {len(videos)} videos:")
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat}: {n}")

    # Extract
    print(f"\n── Extracting features ──")
    manifest = extract_features(videos)

    # Load
    dataset = load_dataset(manifest)
    n = len(dataset["categories"])
    cats = sorted(set(dataset["categories"]))
    print(f"\n  Dataset: {n} videos, {len(cats)} shot types")

    # Clustering (pure features, no training)
    clustering_analysis(dataset)

    # Within-domain training
    print(f"\n{'=' * 70}")
    print("WITHIN-DOMAIN TRAINING (80/20 split)")
    print(f"{'=' * 70}")

    result = train_head(dataset)
    if result:
        print(f"  MAE: {result['mae']:.3f}")
        print(f"  Per-axis:", end="")
        for name, v in zip(AXIS_NAMES, result["per_axis"]):
            print(f"  {name}={v:.2f}", end="")
        print()

    # Cross-shot transfer
    print(f"\n{'=' * 70}")
    print("CROSS-SHOT TRANSFER")
    print("Train on 11 shot types, test on held-out 12th")
    print(f"{'=' * 70}")

    print(f"\n  {'holdout':>20s}  {'n_tr':>4s}  {'n_te':>4s}  {'MAE':>5s}", end="")
    for ax in AXIS_NAMES:
        print(f"  {ax[:5]:>5s}", end="")
    print()

    for cat in cats:
        result = train_head(dataset, holdout=cat)
        if result:
            print(f"  {cat:>20s}  {result['n_train']:4d}  {result['n_test']:4d}"
                  f"  {result['mae']:5.2f}", end="")
            for v in result["per_axis"]:
                marker = "*" if v < 2.0 else " "
                print(f"  {v:4.2f}{marker}", end="")
            print()

    print(f"""
{'=' * 70}
WHAT THIS PROVES
{'=' * 70}

  If forehands cluster together in encoder space without labels,
  the encoder sees what a coach sees: similar body mechanics,
  similar racquet paths, similar ball trajectories.

  If a held-out shot type (e.g., Smash) can be scored by a model
  trained on the other 11, the visual primitives of tennis
  (speed, impact, verticality) are shared across shot types.

  No words were used. No shot names. Just pixels and scores.
""")


if __name__ == "__main__":
    main()
