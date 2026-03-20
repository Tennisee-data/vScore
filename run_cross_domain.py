"""
vScore cross-domain experiment.

Downloads 3 sample videos from different domains, runs them through
V-JEPA 2, and compares the raw feature spaces to answer:

    Do different domains share visual primitives?

If a "looming fast object" activates similar features whether it's
a puck, a car, or a fist — then cross-domain transfer is real.

Run: python -m vScore.run_cross_domain
"""

import sys
import torch
import numpy as np
from pathlib import Path

# ── Sample videos (public, short clips from HuggingFace datasets) ──────────

SAMPLES = {
    # Domain: (video_url, description for humans only — model never sees this)
    # 3 visually distinct domains: projectile release, impact/collision, wind dynamics
    "projectile": (
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4",
        "archery — projectile launch, tension-release pattern",
    ),
    "collision": (
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/--dVV4_CSvw_000033_000043.mp4",
        "bowling — object on collision course with targets",
    ),
    "wind": (
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/flying_kite/07xOT83TIG4_000040_000050.mp4",
        "kite flying — wind/weather/aerial dynamics",
    ),
}


def load_model():
    from transformers import AutoVideoProcessor, AutoModel

    hf_repo = "facebook/vjepa2-vitl-fpc64-256"
    print(f"Loading {hf_repo}...")
    model = AutoModel.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    model.eval()
    print(f"  Loaded. Device: {model.device}")
    return model, processor


def download_video(url: str, cache_dir: Path = Path("vScore/.cache")) -> Path:
    """Download video to local cache if not already present."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    local_path = cache_dir / filename

    if not local_path.exists():
        print(f"  Downloading: {filename}")
        urllib.request.urlretrieve(url, local_path)
    else:
        print(f"  Cached: {filename}")

    return local_path


def load_video(url: str, processor, n_frames: int = 64):
    import av

    local_path = download_video(url)
    print(f"  Decoding video: {local_path.name}")

    container = av.open(str(local_path))
    stream = container.streams.video[0]

    # Decode all frames
    all_frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")  # H x W x 3
        all_frames.append(arr)
    container.close()

    total_frames = len(all_frames)

    # Sample n_frames evenly
    if total_frames >= n_frames:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        indices = np.pad(indices, (0, n_frames - total_frames), mode="wrap")

    sampled = [all_frames[i] for i in indices]
    # Stack to (T, H, W, C) — processor expects list of frames or numpy array
    video_array = np.stack(sampled)  # (T, H, W, 3)

    processed = processor(list(video_array), return_tensors="pt")
    return processed


def extract_features(model, video_input) -> torch.Tensor:
    with torch.no_grad():
        features = model.get_vision_features(
            **{k: v.to(model.device) for k, v in video_input.items()}
        )
    return features


def analyze_features(features_by_domain: dict[str, torch.Tensor]):
    """
    Compare feature spaces across domains.
    No words involved — just geometry.
    """
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN FEATURE ANALYSIS")
    print("=" * 60)

    # Basic stats per domain
    print("\n── Per-domain feature statistics ──")
    for name, feat in features_by_domain.items():
        f = feat.squeeze()
        if f.dim() > 1:
            # Has spatial/temporal tokens — report shape and pool
            print(f"  {name:>10s}: raw shape {list(f.shape)}")
            f_pooled = f.mean(dim=list(range(f.dim() - 1)))
        else:
            f_pooled = f
            print(f"  {name:>10s}: shape {list(f.shape)}")

        print(f"             mean={f_pooled.mean():.4f}  std={f_pooled.std():.4f}"
              f"  L2={f_pooled.norm():.2f}"
              f"  sparsity={((f_pooled.abs() < 0.01).sum() / f_pooled.numel()):.1%}")

        features_by_domain[name] = f  # Store squeezed version

    # Cosine similarity matrix between domains
    print("\n── Cosine similarity (global pooled features) ──")
    names = list(features_by_domain.keys())
    pooled = {}
    for name, feat in features_by_domain.items():
        f = feat.float()
        if f.dim() > 1:
            pooled[name] = f.mean(dim=list(range(f.dim() - 1)))
        else:
            pooled[name] = f

    print(f"{'':>10s}", end="")
    for n in names:
        print(f"  {n:>10s}", end="")
    print()

    for i, n1 in enumerate(names):
        print(f"{n1:>10s}", end="")
        for j, n2 in enumerate(names):
            sim = torch.nn.functional.cosine_similarity(
                pooled[n1].unsqueeze(0), pooled[n2].unsqueeze(0)
            ).item()
            print(f"  {sim:10.4f}", end="")
        print()

    # Token-level analysis (if features have spatial/temporal dims)
    print("\n── Token-level analysis ──")
    for name, feat in features_by_domain.items():
        if feat.dim() > 1:
            # Reshape to (n_tokens, dim)
            tokens = feat.reshape(-1, feat.shape[-1]).float()
            n_tokens = tokens.shape[0]

            # Self-similarity: how diverse are the tokens within this domain?
            if n_tokens > 1:
                # Sample pairs for efficiency
                n_pairs = min(500, n_tokens * (n_tokens - 1) // 2)
                idx1 = torch.randint(0, n_tokens, (n_pairs,))
                idx2 = torch.randint(0, n_tokens, (n_pairs,))
                mask = idx1 != idx2
                idx1, idx2 = idx1[mask], idx2[mask]

                sims = torch.nn.functional.cosine_similarity(
                    tokens[idx1], tokens[idx2]
                )
                print(f"  {name:>10s}: {n_tokens} tokens, "
                      f"self-similarity mean={sims.mean():.4f} std={sims.std():.4f}")
        else:
            print(f"  {name:>10s}: single vector (no token structure)")

    # Cross-domain token similarity
    print("\n── Cross-domain token similarity ──")
    print("  (Do tokens from different domains activate similar features?)")
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            f1 = features_by_domain[n1].float()
            f2 = features_by_domain[n2].float()

            if f1.dim() > 1 and f2.dim() > 1:
                t1 = f1.reshape(-1, f1.shape[-1])
                t2 = f2.reshape(-1, f2.shape[-1])

                # Sample cross-domain pairs
                n_pairs = 1000
                idx1 = torch.randint(0, t1.shape[0], (n_pairs,))
                idx2 = torch.randint(0, t2.shape[0], (n_pairs,))

                sims = torch.nn.functional.cosine_similarity(t1[idx1], t2[idx2])
                print(f"  {n1:>10s} x {n2:<10s}: "
                      f"mean={sims.mean():.4f} std={sims.std():.4f} "
                      f"max={sims.max():.4f}")

    # Top shared dimensions: which feature dimensions are highly activated
    # across ALL domains?
    print("\n── Shared high-activation dimensions ──")
    print("  (Feature dims that are active across all 3 domains)")
    means = []
    for name in names:
        f = features_by_domain[name].float()
        if f.dim() > 1:
            means.append(f.mean(dim=list(range(f.dim() - 1))))
        else:
            means.append(f)

    stacked = torch.stack(means)  # (n_domains, dim)
    # Dims where ALL domains have high absolute activation
    min_activation = stacked.abs().min(dim=0).values
    top_shared = min_activation.topk(20)

    print(f"  Top 20 universally active dimensions (out of {stacked.shape[1]}):")
    print(f"  dims:   {top_shared.indices.tolist()}")
    print(f"  scores: {[f'{v:.3f}' for v in top_shared.values.tolist()]}")

    # Dims where domains DIVERGE most
    variance_per_dim = stacked.var(dim=0)
    top_divergent = variance_per_dim.topk(20)
    print(f"\n  Top 20 domain-discriminating dimensions:")
    print(f"  dims:   {top_divergent.indices.tolist()}")
    print(f"  scores: {[f'{v:.3f}' for v in top_divergent.values.tolist()]}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
  High shared activation = universal visual primitives
    (edges, motion, spatial structure — domain-agnostic)

  High divergence = domain-specific features
    (fire texture vs ice surface vs water patterns)

  If cross-domain token similarity is high, the encoder
  already sees shared structure. The valence heads just
  need to learn which activations matter for each domain.

  If it's low, the encoder treats them as different worlds
  and transfer will require fine-tuning the encoder too.
""")


def main():
    model, processor = load_model()

    features = {}
    for domain, (url, desc) in SAMPLES.items():
        print(f"\n── Processing: {domain} ({desc}) ──")
        video_input = load_video(url, processor)
        feat = extract_features(model, video_input)
        features[domain] = feat
        print(f"  Features extracted: {list(feat.shape)}")

    analyze_features(features)


if __name__ == "__main__":
    main()
