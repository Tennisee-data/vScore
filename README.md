# vScore: Pre-Linguistic Visual Intelligence

A framework for visual intelligence that operates below language. Videos are mapped to numerical valence vectors on domain-specific outcome axes. Zero is homeostasis. Everything above zero is cost.

No words anywhere in the pipeline.

## The idea

Current visual AI maps pixels to words. Biology does the opposite: a gazelle flees before it knows the word "lion." A firefighter reads a blaze and acts before articulating why. The evaluation is pre-linguistic, operating on outcome-relevant axes (threat, speed, containment, momentum) through threshold dynamics, not categorization.

vScore formalizes this as:

```
pixels → encoder → valence scores → trajectory projection → threshold trigger → [language, optionally]
```

Built on [V-JEPA 2](https://arxiv.org/abs/2603.14482) (LeCun et al., Meta FAIR) as the frozen visual encoder. If the encoder learns to see without words, what should the evaluation layer look like? Our answer: biological valence scoring.

## Results

213 videos, 13 categories, 2 datasets (Kinetics-mini, HACS). Within-domain MAE: **0.79** on a 0-10 scale.

**Cross-domain transfer** (train on 12 categories, test on held-out 13th):

| Axis | Mean MAE | Transfers in | Status |
|------|----------|-------------|--------|
| coordination | 1.45 | 11/13 (85%) | Universal |
| impact | 1.47 | 9/13 (69%) | Universal |
| speed | 1.74 | 8/13 (62%) | Universal |
| verticality | 2.16 | 7/13 (54%) | Semi-universal |
| precision | 2.56 | 6/13 (46%) | Semi-universal |
| tension | 3.13 | 5/13 (38%) | Domain-specific |

Coordination, impact, and speed are universal visual primitives that transfer to unseen domains. Tension is genuinely domain-specific.

## Quick start

```bash
pip install torch transformers av numpy
git clone https://github.com/Tennisee-data/vScore.git
cd vScore
```

### Run the fire simulation demo (no GPU, no downloads)

```bash
python -m vScore.demo
```

### Run action inference demo

```bash
python -m vScore.demo_actions
```

Shows how four vectors with identical magnitude (~10.0) produce four different actions (flee, fight, bond, freeze) depending on direction. Magnitude is arousal. Direction is meaning.

### Run the multimodal demo

```bash
python -m vScore.demo_multimodal
```

Same scoring mechanism across vision and audio. A locomotive approaching: vision scores fear=4, audio scores fear=9. The word "locomotive" is never needed.

### Run the dual-system demo (vScore + LLM)

```bash
python -m vScore.demo_dual_system
```

vScore gates the LLM. Clear threats are handled in milliseconds without words. Ambiguous situations escalate to linguistic reasoning. "I flinched before I knew why" = vScore fired before the LLM started.

### Train on real video (requires V-JEPA 2 download, ~30 min on CPU)

```bash
# Extract features (one-time, cached forever)
python -m vScore.extract_batch

# Train and run cross-domain transfer
python -m vScore.train_v2
```

## Architecture

```
vScore/
├── core/
│   ├── metaclass.py         # ScoredDomain metaclass
│   ├── valence.py           # ValenceVector, ValenceTrajectory
│   ├── threshold.py         # Dynamic threshold triggering
│   ├── action_space.py      # Geometric action inference
│   ├── memory_bayesian.py   # Posterior-aware experience memory
│   ├── multimodal.py        # Cross-modal fusion in valence space
│   ├── dual_system.py       # vScore + LLM integration
│   └── prosody.py           # Voice intonation scoring
├── domains/
│   ├── survival.py          # Panksepp's 7 primal circuits
│   ├── fire.py              # Firefighting
│   ├── hockey.py            # Sport dynamics
│   ├── trading.py           # Market dynamics
│   ├── weather.py           # Atmospheric conditions
│   ├── sound.py             # Auditory threat/safety
│   ├── industrial.py        # Machine monitoring
│   └── music.py             # Affective musical scoring
├── encoder/
│   └── bridge.py            # V-JEPA 2 to valence head
├── projection/
│   └── temporal.py          # Trajectory projection
└── paper/
    └── vscore_paper.md      # Full paper
```

## Defining a new domain

10 lines of Python:

```python
from vScore.core.metaclass import DomainBase

class Surgery(DomainBase):
    axes = [
        "bleeding",        # Hemorrhage severity
        "tissue_exposure",  # Surgical field visibility
        "instrument_proximity", # Distance to critical structures
        "patient_stability",    # Vital sign deviation
        "time_pressure",        # Urgency of completion
    ]
```

The encoder, valence head, action inference, trajectory projection, threshold triggering, and Bayesian memory all work identically on this new domain without any code changes.

## Key concepts

**Zero is homeostasis.** Every non-zero score is a cost. The system exists to return to zero.

**Direction, not magnitude.** Four vectors with the same energy produce four different actions depending on which axes are activated. A classifier says "high activation" for all four. The valence scorer says flee, fight, nurture, or freeze.

**Trajectory, not snapshot.** The system projects where each axis is heading and acts on the projection. A fire at intensity=5 that is accelerating triggers earlier than a fire at intensity=7 that is stable.

**Bayesian memory.** Experiences are stored, recalled, and selectively forgotten based on their statistical contribution to the posterior. Surprising events are kept. Routine ones are discarded. The prior sharpens over time. Replay recomputes all retention scores against the current posterior to eliminate sequential bias.

**Modality-agnostic.** The valence vector is the universal interface. Vision, audio, and any future modality produce vectors in the same space. Fusion happens in valence space, not feature space. Modality conflict (audio says danger, vision says safe) is itself a diagnostic signal.

**Language is optional.** Words enter at the last layer as a lookup table. The intelligence lives in levels 0-2. Level 3 is serialization for human consumption.

## Built on

- [V-JEPA 2](https://arxiv.org/abs/2603.14482) (Bardes, LeCun et al., Meta FAIR) for visual encoding
- [Panksepp (1998)](https://global.oup.com/academic/product/affective-neuroscience-9780195178050) for the primal affective circuits
- [LeCun (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) for the JEPA architecture and the argument that LLMs are insufficient for world understanding
- [HACS](https://github.com/hangzhaomit/HACS-dataset) (Zhao et al., 2019) for action video data

## Status

Preliminary research. The approach shows promise but more extensive experiments are needed: larger datasets, real expert annotations, audio encoder integration, and temporal sequence testing. See the [paper](paper/vscore_paper.md) for full discussion.

## License

MIT
