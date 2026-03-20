"""
Download a curated HACS subset — 10 videos per category, 10 categories.

100 videos covering visually diverse action types that stress
different valence axes. Downloads from YouTube via yt-dlp,
trims to the annotated segment.

Run: python -m vScore.download_hacs_subset

Requires: pip install yt-dlp
"""

import json
import subprocess
import sys
from pathlib import Path

HACS_ANNOTATION = Path("HACS-dataset/HACS_v1.1.1/HACS_segments_v1.1.1.json")
OUTPUT_DIR = Path("vScore/.cache/hacs")
VIDEOS_PER_CATEGORY = 10

# 10 categories chosen for visual diversity across valence axes
# axes: [speed, impact, precision, verticality, coordination, tension]
TARGET_CATEGORIES = {
    #                          speed  impact  prec  vert  coord  tension
    "Bullfighting":           [7.0,   8.0,    4.0,  1.0,  2.0,   9.0],  # threat + impact
    "Springboard diving":     [5.0,   5.0,    8.0,  9.0,  1.0,   7.0],  # verticality + precision
    "Arm wrestling":          [2.0,   6.0,    5.0,  0.5,  2.0,   9.0],  # tension + impact, static
    "Javelin throw":          [8.0,   3.0,    8.0,  6.0,  1.0,   8.0],  # speed + precision
    "Snowboarding":           [8.0,   4.0,    6.0,  4.0,  1.0,   4.0],  # speed dominant
    "Chopping wood":          [4.0,   9.0,    6.0,  5.0,  1.0,   5.0],  # impact dominant
    "Doing fencing":          [7.0,   4.0,    9.0,  1.0,  2.0,   8.0],  # precision + threat
    "Pole vault":             [7.0,   3.0,    8.0,  9.0,  1.0,   9.0],  # verticality + tension
    "Slacklining":            [2.0,   1.0,    8.0,  3.0,  1.0,   7.0],  # precision + tension, slow
    "Fixing the roof":        [2.0,   3.0,    5.0,  5.0,  1.0,   4.0],  # routine manual work
}


def main():
    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    # Load annotations
    if not HACS_ANNOTATION.exists():
        print(f"Annotation file not found: {HACS_ANNOTATION}")
        print("Run: cd HACS-dataset && unzip HACS_v1.1.1.zip")
        sys.exit(1)

    data = json.load(open(HACS_ANNOTATION))["database"]

    # Collect target videos
    targets = {}  # category → [(youtube_id, start, end), ...]
    for vid_id, info in data.items():
        if info["subset"] != "validation":
            continue
        for ann in info["annotations"]:
            label = ann["label"]
            if label in TARGET_CATEGORIES:
                if label not in targets:
                    targets[label] = []
                if len(targets[label]) < VIDEOS_PER_CATEGORY:
                    targets[label].append({
                        "youtube_id": vid_id,
                        "start": ann["segment"][0],
                        "end": ann["segment"][1],
                    })

    total = sum(len(v) for v in targets.values())
    print(f"Downloading {total} videos across {len(targets)} categories")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    errors = 0

    for category, videos in sorted(targets.items()):
        cat_dir = OUTPUT_DIR / category.replace(" ", "_")
        cat_dir.mkdir(exist_ok=True)

        print(f"── {category} ({len(videos)} videos) ──")

        for i, vid in enumerate(videos):
            yt_id = vid["youtube_id"]
            start = vid["start"]
            end = vid["end"]
            duration = end - start
            output_path = cat_dir / f"{yt_id}_{start:.0f}_{end:.0f}.mp4"

            if output_path.exists():
                print(f"  [{i+1}/{len(videos)}] cached: {output_path.name}")
                success += 1
                continue

            # Download and trim with yt-dlp
            url = f"https://www.youtube.com/watch?v={yt_id}"
            cmd = [
                "yt-dlp",
                "-f", "worst[ext=mp4]",    # Smallest quality — we only need features
                "--download-sections", f"*{start}-{end}",
                "--force-keyframes-at-cuts",
                "-o", str(output_path),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                url,
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if output_path.exists():
                    print(f"  [{i+1}/{len(videos)}] OK: {output_path.name}")
                    success += 1
                else:
                    print(f"  [{i+1}/{len(videos)}] FAIL: {yt_id} (video unavailable)")
                    errors += 1
            except subprocess.TimeoutExpired:
                print(f"  [{i+1}/{len(videos)}] TIMEOUT: {yt_id}")
                errors += 1
            except Exception as e:
                print(f"  [{i+1}/{len(videos)}] ERROR: {yt_id} — {e}")
                errors += 1

    print(f"\n── Summary ──")
    print(f"  Downloaded: {success}")
    print(f"  Errors: {errors} (YouTube videos may be removed)")
    print(f"  Location: {OUTPUT_DIR}")
    print(f"\n  Next: python -m vScore.extract_batch")
    print(f"        python -m vScore.train_v2")

    # Save proxy scores for train_v2
    scores_path = OUTPUT_DIR / "proxy_scores.json"
    # Normalize category names to match directory names
    normalized = {}
    for cat, scores in TARGET_CATEGORIES.items():
        normalized[cat.replace(" ", "_")] = scores
    scores_path.write_text(json.dumps(normalized, indent=2))
    print(f"\n  Proxy scores saved to: {scores_path}")


if __name__ == "__main__":
    main()
