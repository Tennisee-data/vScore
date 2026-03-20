"""
Batch feature extraction — extract once, experiment forever.

Processes videos from multiple sources, extracts V-JEPA 2 features,
caches them to disk. All downstream experiments run on cached
1024-dim vectors — instant, zero cost.

Sources:
    1. Kinetics-mini (train + val splits, 100 videos, 5 classes)
    2. HACS (if available)
    3. Any additional HuggingFace video dataset

Run: python -m vScore.extract_batch
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import av
import numpy as np
import torch

# ── Configuration ──────────────────────────────────────────────

CACHE_DIR = Path("vScore/.cache")
FEATURES_DIR = Path("vScore/.features")
MANIFEST_PATH = FEATURES_DIR / "manifest.json"
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES = 64

# ── Video sources ─────────────────────────────────────────────

KINETICS_MINI_BASE = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main"


def discover_kinetics_mini() -> list[dict]:
    """Discover all available videos in kinetics-mini (train + val)."""
    videos = []
    for split in ["train", "val"]:
        url = f"https://huggingface.co/api/datasets/nateraw/kinetics-mini/tree/main/{split}"
        try:
            resp = urllib.request.urlopen(url)
            categories = json.loads(resp.read())
        except Exception as e:
            print(f"  Warning: could not list {split}: {e}")
            continue

        for cat_entry in categories:
            if cat_entry["type"] != "directory":
                continue
            cat = cat_entry["path"].split("/")[-1]
            cat_url = f"https://huggingface.co/api/datasets/nateraw/kinetics-mini/tree/main/{split}/{cat}"
            try:
                resp2 = urllib.request.urlopen(cat_url)
                files = json.loads(resp2.read())
            except Exception:
                continue

            for f in files:
                if f["type"] == "file" and f["path"].endswith(".mp4"):
                    fname = f["path"].split("/")[-1]
                    videos.append({
                        "source": "kinetics-mini",
                        "split": split,
                        "category": cat,
                        "filename": fname,
                        "url": f"{KINETICS_MINI_BASE}/{split}/{cat}/{fname}",
                        "video_id": f"kinetics-mini/{split}/{cat}/{fname}",
                    })

    return videos


def setup_hacs() -> list[dict]:
    """
    HACS setup instructions.

    HACS (Human Action Clips and Segments) has 200 action classes
    and 1.5M clips, but videos are sourced from YouTube — not
    directly downloadable from HuggingFace.

    To use HACS:
        1. Clone: git clone https://github.com/hangzhaomit/HACS-dataset
        2. Install: pip install youtube-dl (or yt-dlp)
        3. Download validation set (smallest):
           python download_videos.py --root_dir vScore/.cache/hacs --dataset segments
        4. Re-run this script — it will detect HACS videos in .cache/hacs/

    Categories include: archery, basketball_dunk, biking, cliff_diving,
    cricket, fencing, golf, javelin_throw, juggling, kayaking, pole_vault,
    skateboarding, skiing, surfing, sword_fighting, and 185 more.
    """
    hacs_dir = CACHE_DIR / "hacs"
    videos = []

    if not hacs_dir.exists():
        print("  HACS not downloaded yet.")
        print("  To add HACS videos:")
        print("    1. pip install yt-dlp")
        print("    2. git clone https://github.com/hangzhaomit/HACS-dataset")
        print("    3. cd HACS-dataset")
        print("    4. python download_videos.py --root_dir ../vScore/.cache/hacs --dataset segments")
        print("    5. Re-run: python -m vScore.extract_batch")
        return videos

    # Scan for downloaded HACS videos
    for category_dir in sorted(hacs_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for video_file in sorted(category_dir.glob("*.mp4")):
            videos.append({
                "source": "hacs",
                "split": "segments",
                "category": category,
                "filename": video_file.name,
                "url": str(video_file),  # Local path, not URL
                "video_id": f"hacs/{category}/{video_file.name}",
                "local": True,
            })

    if videos:
        categories = set(v["category"] for v in videos)
        print(f"  Found {len(videos)} HACS videos in {len(categories)} categories")
    else:
        print("  HACS directory exists but no .mp4 files found.")

    return videos


# ── Core extraction functions ─────────────────────────────────

def download_video(video: dict) -> Path:
    """Download video to local cache if not present, or return local path."""
    if video.get("local"):
        return Path(video["url"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / video["filename"]
    if not local_path.exists():
        urllib.request.urlretrieve(video["url"], local_path)
    return local_path


def decode_video(path: Path, n_frames: int = 64) -> np.ndarray:
    """Decode video to numpy array of frames."""
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames decoded from {path}")

    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = np.arange(total)
        indices = np.pad(indices, (0, n_frames - total), mode="wrap")

    return np.stack([frames[i] for i in indices])


def load_manifest() -> dict:
    """Load existing manifest or create empty."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    """Save manifest to disk."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def extract_features_batch(videos: list[dict]):
    """
    Extract V-JEPA 2 features for all videos.
    Skips already-cached videos. Loads model only if needed.
    """
    manifest = load_manifest()

    # Filter to uncached only
    to_process = [v for v in videos if v["video_id"] not in manifest]

    if not to_process:
        print(f"  All {len(videos)} videos already cached.")
        return manifest

    print(f"  {len(to_process)} new videos to process "
          f"({len(manifest)} already cached)")
    print(f"  Estimated time: ~{len(to_process) * 9 // 60} min {len(to_process) * 9 % 60} sec")

    # Load model only when needed
    from transformers import AutoVideoProcessor, AutoModel

    print(f"\n  Loading encoder: {HF_REPO}")
    model = AutoModel.from_pretrained(HF_REPO)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model.eval()
    print(f"  Loaded on {model.device}\n")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    t0 = time.time()

    for i, video in enumerate(to_process):
        video_id = video["video_id"]
        # Create safe filename for feature cache
        safe_name = video_id.replace("/", "_").replace(" ", "_")
        feat_path = FEATURES_DIR / f"{safe_name}.pt"

        try:
            # Download
            local = download_video(video)

            # Decode
            video_array = decode_video(local, N_FRAMES)

            # Process
            inputs = processor(list(video_array), return_tensors="pt")

            # Extract
            with torch.no_grad():
                features = model.get_vision_features(**inputs)

            # Pool to global feature: (1, 8192, 1024) → (1024,)
            pooled = features.squeeze(0).mean(dim=0)

            # Save
            torch.save(pooled, feat_path)
            manifest[video_id] = {
                "feature_path": str(feat_path),
                "source": video["source"],
                "category": video["category"],
                "split": video.get("split", "unknown"),
            }

            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(to_process) - i - 1)

            print(f"  [{i+1}/{len(to_process)}] {video_id}  "
                  f"({remaining:.0f}s remaining)")

        except Exception as e:
            errors.append((video_id, str(e)))
            print(f"  [{i+1}/{len(to_process)}] ERROR {video_id}: {e}")

        # Save manifest incrementally (crash-safe)
        if (i + 1) % 10 == 0:
            save_manifest(manifest)

    save_manifest(manifest)

    elapsed = time.time() - t0
    print(f"\n  Done: {len(to_process) - len(errors)} extracted, "
          f"{len(errors)} errors, {elapsed:.0f}s total")

    if errors:
        print(f"\n  Errors:")
        for vid, err in errors:
            print(f"    {vid}: {err}")

    return manifest


# ── Proxy scores for all categories ───────────────────────────

# axes: [speed, impact, precision, verticality, coordination, tension]
PROXY_SCORES = {
    # Kinetics-mini
    "archery":      [3.0, 2.0, 9.0, 1.0, 1.0, 8.0],
    "bowling":      [5.0, 8.0, 7.0, 0.5, 1.0, 6.0],
    "flying_kite":  [4.0, 0.5, 3.0, 7.0, 1.0, 1.0],
    "high_jump":    [7.0, 4.0, 6.0, 9.0, 1.0, 7.0],
    "marching":     [3.0, 1.0, 5.0, 0.5, 9.0, 1.0],
}

AXIS_NAMES = ["speed", "impact", "precision", "verticality", "coordination", "tension"]


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("vScore Batch Feature Extraction")
    print("=" * 70)

    all_videos = []

    # Source 1: Kinetics-mini (train + val)
    print("\n── Discovering kinetics-mini videos ──")
    km_videos = discover_kinetics_mini()
    print(f"  Found {len(km_videos)} videos across train+val")
    by_cat = {}
    for v in km_videos:
        key = f"{v['split']}/{v['category']}"
        by_cat[key] = by_cat.get(key, 0) + 1
    for key, count in sorted(by_cat.items()):
        print(f"    {key}: {count}")
    all_videos.extend(km_videos)

    # Source 2: HACS
    print("\n── Checking HACS ──")
    hacs_videos = setup_hacs()
    if hacs_videos:
        all_videos.extend(hacs_videos)

    # Extract
    print(f"\n── Extracting features ──")
    print(f"  Total videos: {len(all_videos)}")
    manifest = extract_features_batch(all_videos)

    # Summary
    print(f"\n── Summary ──")
    print(f"  Total cached features: {len(manifest)}")
    by_source = {}
    by_category = {}
    for vid, info in manifest.items():
        if isinstance(info, dict):
            src = info.get("source", "unknown")
            cat = info.get("category", vid.split("/")[0])
        else:
            src = "unknown"
            cat = vid.split("/")[0]
        by_source[src] = by_source.get(src, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"  By source:")
    for src, count in sorted(by_source.items()):
        print(f"    {src}: {count}")
    print(f"  By category:")
    for cat, count in sorted(by_category.items()):
        scores = PROXY_SCORES.get(cat, None)
        score_str = f"  scores: {scores}" if scores else "  (no proxy scores yet)"
        print(f"    {cat}: {count}{score_str}")

    print(f"\n  Cache size: {sum(1 for _ in FEATURES_DIR.glob('*.pt'))} files, "
          f"{sum(f.stat().st_size for f in FEATURES_DIR.glob('*.pt')) / 1024:.0f} KB")
    print(f"\n  All downstream experiments now run on cached vectors — instant.")


if __name__ == "__main__":
    main()
