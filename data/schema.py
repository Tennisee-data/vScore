"""
Annotation schema for vScore training data.

No labels. No words. Just scores per frame window.

Each annotation is a video segment scored on domain-specific axes.
Overlapping windows so the model learns trajectories, not snapshots.

Annotations can come from:
    - Human domain experts (firefighters scoring fire videos, etc.)
    - Physiological sensors (heart rate, GSR, pupil dilation)
    - Outcome data (did the building collapse? did the team score?)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AnnotationRecord:
    """
    A single scored observation of a video segment.

    No text. No categories. Just:
        - Which video
        - Which frames
        - Which domain
        - What scores on each axis
    """
    video_id: str
    frame_start: int
    frame_end: int
    domain: str
    scores: list[float]         # One per axis, all >= 0, 0 = neutral
    fps: float = 30.0           # For converting frames to time
    annotator: str = "human"    # "human", "physiological", "outcome"
    confidence: float = 1.0     # Annotator confidence in the scores

    def __post_init__(self):
        if any(s < 0 for s in self.scores):
            raise ValueError("All scores must be >= 0. Zero is homeostasis.")

    @property
    def time_start(self) -> float:
        return self.frame_start / self.fps

    @property
    def time_end(self) -> float:
        return self.frame_end / self.fps

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start


@dataclass
class DomainAnnotationSet:
    """
    All annotations for a single domain.

    Provides windowed iteration for trajectory learning.
    """
    domain: str
    axis_names: list[str]
    records: list[AnnotationRecord] = field(default_factory=list)

    def add(self, record: AnnotationRecord):
        if record.domain != self.domain:
            raise ValueError(f"Domain mismatch: {record.domain} vs {self.domain}")
        if len(record.scores) != len(self.axis_names):
            raise ValueError(
                f"Expected {len(self.axis_names)} scores, got {len(record.scores)}"
            )
        self.records.append(record)

    def get_video_trajectory(self, video_id: str) -> list[AnnotationRecord]:
        """Get all annotations for a video, sorted by frame_start."""
        return sorted(
            [r for r in self.records if r.video_id == video_id],
            key=lambda r: r.frame_start,
        )

    @property
    def video_ids(self) -> list[str]:
        return sorted(set(r.video_id for r in self.records))

    def stats(self) -> dict:
        """Summary statistics — no words, just numbers."""
        if not self.records:
            return {"n_records": 0}

        import statistics
        all_scores = [s for r in self.records for s in r.scores]
        return {
            "n_records": len(self.records),
            "n_videos": len(self.video_ids),
            "mean_score": statistics.mean(all_scores),
            "max_score": max(all_scores),
            "pct_zero": sum(1 for s in all_scores if s == 0) / len(all_scores),
        }


def save_annotations(dataset: DomainAnnotationSet, path: str | Path):
    path = Path(path)
    data = {
        "domain": dataset.domain,
        "axis_names": dataset.axis_names,
        "records": [asdict(r) for r in dataset.records],
    }
    path.write_text(json.dumps(data, indent=2))


def load_annotations(path: str | Path) -> DomainAnnotationSet:
    path = Path(path)
    data = json.loads(path.read_text())
    dataset = DomainAnnotationSet(
        domain=data["domain"],
        axis_names=data["axis_names"],
    )
    for r in data["records"]:
        dataset.add(AnnotationRecord(**r))
    return dataset
