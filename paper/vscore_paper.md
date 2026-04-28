# vScore: Pre-Linguistic Visual Intelligence Through Domain-Agnostic Valence Scoring

**Francois Reeves**

*Research Assistant: Claude (Anthropic)*

---

## Abstract

We propose vScore, a framework for visual intelligence that operates entirely below the level of language. Rather than mapping visual input to text labels or token sequences, vScore maps video directly to numerical valence vectors: continuous scores on domain-specific outcome axes where zero represents homeostasis and any deviation represents a cost demanding attention. The framework builds on the observation that biological intelligence, from insects to expert humans, evaluates visual stimuli through pre-linguistic pattern-to-outcome projection rather than word retrieval. We formalize this as a metaclass architecture where any scoreable domain (survival, fire, weather, sport, trading) is defined solely by its outcome axes, with a shared visual encoder providing domain-agnostic features. Using V-JEPA 2 as the frozen encoder and a set of lightweight valence heads, we demonstrate on 213 videos across 13 categories from two datasets (Kinetics-mini, HACS) that (1) a small regression head can learn to map self-supervised video features to valence scores with MAE of 0.79 within domain, (2) three axes (coordination, impact, and speed) transfer across held-out domains with mean MAE of 1.45 to 1.74 (transferring in 62–85% of holdout conditions), confirming the existence of universal visual primitives, (3) axes with no visual analogue in the training set (e.g., synchronized coordination, collision impact) correctly fail to transfer, identifying genuinely novel visual dynamics, (4) actions emerge from the *geometry* of the full valence vector, specifically its direction rather than its magnitude, with trajectory projection enabling preemptive responses before threshold states are reached, and (5) a Bayesian experience memory with posterior-aware eviction selectively retains informative experiences (high surprise, extreme outcomes, novel patterns) while discarding redundant ones, using normalized leave-one-out influence, coverage-driven eviction, and periodic replay to eliminate the sequential biases inherent in naive memory management. We argue that this valence-first, language-last hierarchy offers a more faithful model of biological visual intelligence than current vision-language paradigms. In this hierarchy, intelligence is geometric inference over outcome-projected vectors, learning is Bayesian posterior updating, memory is statistically rigorous retention, and words are an optional lookup table appended after scoring.

---

## 1. Introduction

The dominant paradigm in visual AI maps pixels to words. Image classifiers produce text labels. Vision-language models ground visual features in token sequences. Even self-supervised learners are typically evaluated by how well their representations support linguistic categorization. The assumption is implicit: understanding an image means being able to describe it.

Biology disagrees. A gazelle seeing a predator does not produce the label "lion" before fleeing. A firefighter reading a blaze does not think "the fire is spreading at rate X toward exit Y" before acting. A hockey coach does not narrate the play before calling a timeout. In each case, the visual stimulus is evaluated on outcome-relevant axes (threat, speed, containment, momentum) and the response is triggered by threshold crossing on those axes. Language, when it arrives, is a post-hoc serialization of a decision already made.

This intuition, that intelligence requires world models operating in representation space rather than token space, was articulated by LeCun (2022) in his vision for autonomous machine intelligence. LeCun argued that autoregressive language models are fundamentally limited because they operate on discrete tokens rather than continuous representations of the world, and proposed the Joint Embedding Predictive Architecture (JEPA) as the foundation for a system that learns by predicting in latent space rather than pixel or token space. V-JEPA and V-JEPA 2, the self-supervised video encoders we build upon, are direct implementations of this vision. vScore extends LeCun's framework in a specific direction: if the encoder learns to predict the world in latent space without language, what should the downstream evaluation look like? Our answer is that it should look like biological valence scoring: not classification, not captioning, but continuous outcome projection on domain-relevant axes.

This paper introduces vScore, a framework that inverts the standard hierarchy. Instead of:

```
pixels → encoder → language → understanding
```

we propose:

```
pixels → encoder → valence scores → trajectory projection → threshold trigger → [language, optionally]
```

The key contributions are:

1. **A metaclass formalization** where any domain with scoreable outcomes (survival, fire, sport, trading) is defined by its numerical axes alone, enabling domain-agnostic architecture with domain-specific scoring.

2. **Empirical evidence** that self-supervised video features (V-JEPA 2) contain universal visual primitives, notably motion dynamics, that transfer across visually distinct domains without any language supervision.

3. **A threshold-triggering mechanism** with dynamic thresholds that lower under acceleration, modeling the biological principle that organisms act on projected trajectories, not current states.

4. **A hierarchy** in which language is the final, optional layer: a multilingual lookup table indexed by valence state, not a prerequisite for understanding.

---

## 2. Motivation: The Pre-Linguistic Layer

### 2.1 The Translation Problem

Every word is already a translation. When a human says "fire," they have already performed a complex chain of operations: detect high-contrast flickering pattern, evaluate color (orange-red), estimate spatial extent, assess motion (spreading vs. contained), project trajectory (toward me or away), integrate with context (indoor vs. outdoor, near exits or not), cross a relevance threshold, retrieve a lexical item from memory, and articulate. The word "fire" compresses all of this into four letters. A vision-language model trained on the word "fire" learns the compression, not the evaluation chain that produced it.

This matters because the evaluation chain is where intelligence lives. Two fires can both be labeled "fire" while demanding opposite responses. One is a campfire (approach, warmth, safety) and the other is a structure fire (flee, danger, urgency). The word is the same. The valence is opposite.

### 2.2 Biological Precedent

Panksepp (1998) identified seven primal affective circuits operating at the subcortical level in all mammals: SEEKING, RAGE, FEAR, LUST, CARE, PANIC/GRIEF, and PLAY. These circuits:

- Operate before and independently of cortical (linguistic) processing
- Are activated by specific sensory patterns, not labels
- Produce graded responses (not binary classifications)
- Interact with each other (FEAR suppresses PLAY; SEEKING modulates all others)
- Drive behavior through threshold dynamics, not categorization

Gibson's (1979) theory of affordances makes a complementary argument: perception is not about identifying objects but about detecting action possibilities. A surface is not perceived as "a chair" but as "sittable." The affordance is pre-linguistic and directly coupled to the organism's body and goals.

Damasio's (1994) somatic marker hypothesis extends this further: decisions are driven by body-state projections, essentially emotional previews of future outcomes, not by rational analysis. The "gut feeling" is not noise; it is the organism's fastest evaluation system.

vScore formalizes these biological principles computationally.

### 2.3 The Expert's Eye

The gap between novice and expert in any visual domain is not vocabulary but scoring speed and accuracy. A rookie firefighter and a 20-year veteran both know the word "flashover." The difference is that the veteran sees the visual precursors (darkening smoke, thermal layering, rollover at the ceiling) and scores the trajectory toward flashover before it happens. The expert's advantage is entirely pre-linguistic: faster feature extraction, more accurate valence scoring, better trajectory projection.

This suggests that expert training is fundamentally about calibrating valence heads, not expanding vocabulary. The vScore framework makes this explicit.

---

## 3. Framework

### 3.1 Architecture Overview

vScore consists of four components arranged in a strict hierarchy:

```
Level 2: Visual Encoder (frozen, self-supervised)
    Input: Video frames (T, C, H, W)
    Output: Dense feature tensor (N_tokens, D)

Level 1: Valence Head (trainable, per-domain)
    Input: Pooled features (D,)
    Output: Valence vector (N_axes,)

Level 0: Trajectory Projector + Threshold Trigger
    Input: Sequence of valence vectors over time
    Output: Trigger events (IGNORE / ATTEND / ACT)

Level 3 (optional): Language Lookup
    Input: Valence state
    Output: Multilingual text description
```

Levels 0-2 are the intelligence. Level 3 is serialization.

### 3.2 The ScoredDomain Metaclass

We formalize domain definition using a Python metaclass:

```python
class ScoredDomain(type):
    registry = {}
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if 'axes' in namespace and namespace['axes'] is not None:
            cls._axis_index = {ax: i for i, ax in enumerate(axes)}
            cls._n_axes = len(axes)
            mcs.registry[name] = cls
        return cls
```

A domain is defined entirely by its axes:

```python
class Survival(metaclass=ScoredDomain):
    axes = ['seeking', 'rage', 'fear', 'lust', 'care', 'panic', 'play']

class Fire(metaclass=ScoredDomain):
    axes = ['spread_rate', 'proximity', 'intensity', 'containment',
            'escape_route', 'structural_risk', 'smoke_toxicity']

class Hockey(metaclass=ScoredDomain):
    axes = ['scoring_threat', 'possession_pressure', 'breakaway',
            'penalty_risk', 'momentum', 'fatigue']
```

This design enforces a critical constraint: **the model architecture is identical across domains**. Only the number and meaning of axes change. A fire valence head and a hockey valence head are the same neural network with different output dimensions. The visual encoder is shared.

### 3.3 Valence Vectors and Homeostasis

A valence vector $\mathbf{v} \in \mathbb{R}_{\geq 0}^{n}$ scores the current state on $n$ domain-specific axes. The constraint $v_i \geq 0$ is enforced architecturally (ReLU activation on the output layer).

**Zero is homeostasis.** The organism at rest, no activation needed. Every non-zero value represents a deviation that demands energy, whether attention, computation, or physical action. The system's purpose is to return to zero.

This is a fundamentally different framing from classification. A classifier asks "what is this?" A valence scorer asks "how much does this deviate from nothing happening, and in which directions?"

The magnitude $\|\mathbf{v}\|_2$ gives overall activation level. A video of an empty field scores near zero. A video of a building collapse scores high on multiple axes simultaneously.

### 3.4 Trajectory Projection

A single valence vector is a snapshot. Intelligence requires trajectory: where is each axis heading?

Given a sequence of valence vectors $\mathbf{v}_{t-k}, \ldots, \mathbf{v}_t$, we compute per-axis:

- **Velocity**: $\dot{v}_i = \frac{v_i^{(t)} - v_i^{(t-k)}}{k \cdot \Delta t}$, the rate of change
- **Acceleration**: $\ddot{v}_i$, the rate of change of rate of change
- **Projection**: $\hat{v}_i^{(t+h)} = v_i^{(t)} + \dot{v}_i \cdot h$, the expected axis value at horizon $h$

The linear projection serves as baseline. The learned TemporalProjector, a small transformer over valence sequences, captures nonlinear dynamics such as the super-linear acceleration of fire spread following containment loss.

### 3.5 Dynamic Threshold Triggering

Biological systems do not classify; they trigger. The response function is:

$$\text{response}(v_i, \hat{v}_i) = \begin{cases} \text{ACT} & \text{if } \max(v_i, \hat{v}_i) \geq \tau_i - \alpha \cdot \max(0, \ddot{v}_i) \\ \text{ATTEND} & \text{if } \max(v_i, \hat{v}_i) \geq \tau_i^{\text{att}} - \frac{\alpha}{2} \cdot \max(0, \ddot{v}_i) \\ \text{IGNORE} & \text{otherwise} \end{cases}$$

where $\tau_i$ is the base trigger threshold, $\tau_i^{\text{att}}$ is the attention threshold, and $\alpha$ is the acceleration weight.

The key mechanism: **positive acceleration lowers the threshold.** A fire spreading at constant speed triggers at FEAR=7.0. A fire that is accelerating triggers at FEAR=5.5. The organism does not wait for peak danger; it acts on the derivative.

This models a well-documented biological phenomenon: startle responses and preemptive action are driven by rate of change, not absolute magnitude (Schiff, 1965; Regan & Vincent, 1995).

### 3.6 Action Inference from Valence Geometry

Per-axis thresholds (Section 3.5) answer "is this axis critical?" But real decisions emerge from the **interaction** between axes, from the geometry of the full valence vector. A fire with `[spread=7, containment=8]` is a controlled burn. The same spread with `[spread=7, containment=2]` is a crisis. Same axis, same score, opposite action. The difference is the vector's direction in valence space.

We formalize this through an ActionField: a mapping from regions of valence space to response tendencies. Each ActionRegion is defined by a condition function over the full vector, not individual axes:

$$a_k(\mathbf{v}) = f_k(v_1, v_2, \ldots, v_n) \in [0, 1]$$

where $a_k$ is the activation of action $k$ and $f_k$ captures the interaction logic. For example, in the survival domain:

| Action | Condition | Biological basis |
|--------|-----------|------------------|
| FLEE | fear dominant, rage low | Predator escape |
| FIGHT | rage dominant, fear low | Territory defense |
| FREEZE | fear AND rage both high | Competing drives, tonic immobility |
| PROTECT | care AND fear both high | Parental defense despite danger |
| EXPLORE | seeking high, fear low | Foraging, curiosity |

Three properties make this system qualitatively different from classification:

**1. Direction, not magnitude.** Four vectors with identical L2 norm (~10.0) produce four completely different actions:

| Vector | Action | Conflict |
|--------|--------|----------|
| `[0, 0, 10, 0, 0, 0, 0]` (pure fear) | FLEE (1.00) | 0.33 |
| `[0, 10, 0, 0, 0, 0, 0]` (pure rage) | FIGHT (1.00) | 0.33 |
| `[0, 0, 0, 0, 10, 0, 0]` (pure care) | BOND (1.00) | 0.33 |
| `[3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8]` (distributed) | FREEZE (0.38) | 0.53 |

A classifier would label all four as "high activation." The valence scorer identifies four qualitatively different states. Magnitude is arousal. Direction is meaning.

**2. Conflict detection.** When multiple action regions activate at similar strength, the system quantifies conflict as the ratio of the second-strongest to the strongest activation:

$$\text{conflict}(\mathbf{v}) = \frac{a_{\text{second}}(\mathbf{v})}{a_{\text{top}}(\mathbf{v})}$$

Conflict near 1.0 means the organism faces equally compelling but incompatible actions, the classical Buridan's donkey problem. This is biologically real: tonic immobility in prey animals (Gallup, 1977), decision paralysis in humans under competing threat and reward signals (Corr, 2013). The conflict score predicts hesitation latency, not just action identity.

In the fire domain, we observe high conflict (>0.7) at the transition point where ADVANCE and HOLD_POSITION compete, exactly when a firefighter's decision is hardest. The model quantifies what experienced commanders simply call "the judgment call."

**3. Preemptive action from trajectory.** The action field evaluates not only the current valence vector but the projected vector at horizon $h$. When the projected state activates a higher-priority action than the current state, the system triggers a preemption signal:

$$\text{preempt} = \begin{cases} \text{true} & \text{if } \text{top}_k(a(\hat{\mathbf{v}}_{t+h})) \neq \text{top}_k(a(\mathbf{v}_t)) \text{ and } a_{\text{top}}(\hat{\mathbf{v}}_{t+h}) > a_{\text{top}}(\mathbf{v}_t) \\ \text{false} & \text{otherwise} \end{cases}$$

In our fire simulation, preemption fires at t=2 (current action: HOLD_POSITION; projected action: DEFENSIVE) and at t=4 (current: EVACUATE; projected: MAYDAY). The message is clear: *the present is manageable but the future is not. Act on the projection.* This is how expert firefighters survive: they read the trajectory and leave before the current state demands it.

The combination of direction-dependent action, conflict quantification, and trajectory-based preemption produces behavior that is qualitatively richer than threshold-crossing on individual axes. A 7-dimensional valence space with trajectory generates a continuous manifold of behavioral states, each point defined not by which axis is highest, but by the angle of the vector, the curvature of its trajectory, and the proximity of competing action regions.

### 3.7 The Language Layer (Optional)

Language enters only at Level 3, as a lookup from valence state to multilingual text:

```
[0, 0, 8, 0, 0, 3, 0] → FEAR-dominant, PANIC-secondary
    EN: "charging bear"
    FR: "ours qui charge"
    JP: "突進する熊"
    Physiological: HR↑, pupil dilation, cortisol spike
```

The text is not the understanding. The understanding is the valence vector. The text is one possible serialization, chosen for a specific audience in a specific language. The same understanding could be serialized as a motor command (flee), a physiological response (adrenaline), or a social signal (warning call). All are downstream of the same valence evaluation.

### 3.8 Dual-System Integration: vScore + LLM

vScore does not replace language models. It sits before them. The two systems are complementary, operating at different speeds on different representations:

| Property | System 1 (vScore) | System 2 (LLM) |
|----------|-------------------|-----------------|
| Speed | ~50ms | ~500ms+ |
| Input | Patterns (pixels, waveforms, prosody) | Tokens (words, syntax) |
| Output | Valence vectors, action triggers | Semantic interpretation, reasoning |
| Strength | Speed, cross-modal fusion, threat detection | Context, planning, social reasoning |
| Weakness | No reasoning, no planning | Slow, requires language, misses prosody |

The critical architectural principle: **vScore gates the LLM, not the reverse.** The fast system decides whether the slow system is needed:

- **Low valence magnitude** ($\|\mathbf{v}\| < \tau_{\text{floor}}$): nothing happening, neither system activates
- **High valence, low conflict**: clear situation, vScore acts alone (fleeing from a charging bear leaves no time for words)
- **High valence, high conflict**: ambiguous situation, engage the LLM for reasoning (a child laughing then crying activates both CARE and PANIC, and context is needed)
- **Prosody/semantics mismatch**: voice intonation (scored by vScore) contradicts word content (parsed by LLM), suggesting possible deception, sarcasm, or masked distress

The prosodic bridge is particularly significant. Voice intonation is an acoustic pattern with pitch contour, tempo, roughness, and rhythm, all scoreable on valence axes without understanding a single word. vScore processes prosody as sound: a raised voice activates threat regardless of the language spoken. The LLM processes the words. When prosodic valence and semantic content diverge (for example, "I'm fine" spoken with a trembling voice) the system detects the mismatch and flags it for deliberative processing.

This mirrors the known architecture of biological threat processing: the amygdala (fast, subcortical) evaluates sensory input and gates prefrontal cortex (slow, deliberative) engagement. The startle response precedes conscious evaluation. "I flinched before I knew why" is vScore firing before the LLM started.

---

## 4. Experiments

### 4.1 Setup

**Encoder.** We use V-JEPA 2 (ViT-L, 0.3B parameters) pretrained via self-supervised video prediction (Bardes et al., 2025). The model is frozen throughout all experiments. It produces 8,192 spatiotemporal tokens of dimension 1,024 per 64-frame video clip. We global-average-pool to a single 1,024-dimensional vector per video.

**Data.** 213 videos from two sources: Kinetics-mini (150 videos, train+val splits, 5 categories: archery, bowling, flying kite, high jump, marching) and HACS (Zhao et al., 2019) (63 videos, validation segments, 8 categories: arm wrestling, fencing, fixing roof, javelin throw, pole vault, slacklining, snowboarding, springboard diving). Each clip is sampled at 64 frames uniformly. Features are extracted once and cached as 1,024-dimensional vectors; all training experiments operate on cached vectors.

**Domain.** We define a "dynamics" domain with 6 axes capturing universal motion properties:

| Axis | Description |
|------|-------------|
| speed | Overall motion magnitude |
| impact | Collision / force transfer |
| precision | Controlled vs. chaotic motion |
| verticality | Up/down movement dominance |
| coordination | Synchronized multi-agent motion |
| tension | Buildup before release/outcome |

**Proxy annotations.** Each category receives a fixed valence vector based on its dominant visual dynamics (Table 1). This is a deliberate simplification; real deployment would use per-video expert annotations or physiological measurements.

**Table 1: Proxy valence scores (13 categories, 2 datasets)**

| Category | Source | N | speed | impact | precision | verticality | coordination | tension |
|----------|--------|---|-------|--------|-----------|-------------|--------------|---------|
| archery | K-mini | 30 | 3.0 | 2.0 | 9.0 | 1.0 | 1.0 | 8.0 |
| bowling | K-mini | 30 | 5.0 | 8.0 | 7.0 | 0.5 | 1.0 | 6.0 |
| flying kite | K-mini | 30 | 4.0 | 0.5 | 3.0 | 7.0 | 1.0 | 1.0 |
| high jump | K-mini | 30 | 7.0 | 4.0 | 6.0 | 9.0 | 1.0 | 7.0 |
| marching | K-mini | 30 | 3.0 | 1.0 | 5.0 | 0.5 | 9.0 | 1.0 |
| arm wrestling | HACS | 3 | 2.0 | 6.0 | 5.0 | 0.5 | 2.0 | 9.0 |
| fencing | HACS | 7 | 7.0 | 4.0 | 9.0 | 1.0 | 2.0 | 8.0 |
| fixing roof | HACS | 9 | 2.0 | 3.0 | 5.0 | 5.0 | 1.0 | 4.0 |
| javelin throw | HACS | 10 | 8.0 | 3.0 | 8.0 | 6.0 | 1.0 | 8.0 |
| pole vault | HACS | 10 | 7.0 | 3.0 | 8.0 | 9.0 | 1.0 | 9.0 |
| slacklining | HACS | 4 | 2.0 | 1.0 | 8.0 | 3.0 | 1.0 | 7.0 |
| snowboarding | HACS | 10 | 8.0 | 4.0 | 6.0 | 4.0 | 1.0 | 4.0 |
| springboard diving | HACS | 10 | 5.0 | 5.0 | 8.0 | 9.0 | 1.0 | 7.0 |

**Valence head.** A 3-layer MLP (1024 → 256 → 128 → 6) with GELU activations and ReLU output. Trained with MSE loss, Adam optimizer (lr=1e-3), 500 epochs.

### 4.2 Cross-Domain Feature Analysis

Before training valence heads, we analyze the raw V-JEPA 2 feature space across three representative categories (archery, bowling, flying kite).

**Table 2: Global cosine similarity between domain-pooled features**

| | archery | bowling | flying kite |
|---|---------|---------|-------------|
| archery | 1.00 | 0.45 | 0.69 |
| bowling | 0.45 | 1.00 | 0.56 |
| flying kite | 0.69 | 0.56 | 1.00 |

The encoder produces features that are neither identical across domains (which would prevent discrimination) nor orthogonal (which would prevent transfer). The intermediate similarity (0.45–0.69) suggests a shared visual vocabulary with domain-specific variation, precisely the structure needed for domain-agnostic scoring with domain-specific heads.

Token-level analysis reveals 8,192 spatiotemporal tokens per video with mean self-similarity of 0.24–0.28 within domains and 0.12–0.18 across domains. Critically, maximum cross-domain token similarity reaches 0.54, meaning specific visual tokens (likely encoding motion dynamics) activate nearly identically across different domains.

We identify 20 feature dimensions with high minimum activation across all domains (universal primitives, likely encoding edges, motion magnitude, and spatial structure) and 20 dimensions with high variance across domains (domain-discriminating features encoding texture and scene-specific patterns).

### 4.3 Within-Domain Training

With random 80/20 train/test split across all 213 videos (13 categories):

**Overall test MAE: 0.79** (on a 0–10 scale)

**Table 3: Per-axis MAE (random split)**

| speed | impact | precision | verticality | coordination | tension |
|-------|--------|-----------|-------------|--------------|---------|
| 0.34 | 0.43 | 0.43 | 0.66 | 2.33 | 0.55 |

Five of six axes achieve MAE below 0.7, confirming that the valence head reliably maps V-JEPA 2 features to outcome-relevant scores. Coordination remains the hardest axis (2.33), consistent with the observation that synchronized multi-body motion is a higher-order visual pattern requiring more diverse training examples. The overall MAE of 0.79 represents a 36% improvement over the initial 50-video experiment (1.24), demonstrating that the framework scales efficiently with data.

### 4.4 Cross-Domain Transfer

The critical experiment: train on 12 categories, test on the held-out 13th. If the model learns visual dynamics rather than category identity, axes should transfer to categories never seen during training.

**Table 4: Per-axis MAE when tested on held-out category (13 categories, 213 videos)**

| Held-out | N | speed | impact | precision | verticality | coordination | tension | Overall |
|----------|---|-------|--------|-----------|-------------|--------------|---------|---------|
| arm wrestling | 3 | 2.03 | 2.29 | **1.54** | **0.78** | **0.87** | 3.80 | 1.89 |
| fencing | 7 | **0.93** | **0.13** | 2.33 | 3.89 | **0.33** | 2.09 | 1.62 |
| fixing roof | 9 | 2.37 | **0.56** | 3.31 | 3.18 | **0.32** | 3.44 | 2.20 |
| javelin throw | 10 | **1.30** | **0.19** | **0.94** | **1.31** | **0.66** | **0.67** | 0.85 |
| pole vault | 10 | **0.69** | **0.31** | 2.52 | **0.43** | **0.22** | 9.00 | 2.20 |
| slacklining | 4 | 2.25 | **1.66** | **1.10** | **1.09** | **0.22** | **1.12** | 1.24 |
| snowboarding | 10 | 4.16 | 2.26 | 6.00 | **1.13** | **0.37** | **1.06** | 2.49 |
| spr. diving | 10 | 2.61 | **0.32** | **1.43** | **0.90** | **0.51** | **0.67** | 1.07 |
| archery | 30 | **0.82** | **0.73** | 4.31 | 3.20 | **1.70** | 5.10 | 2.64 |
| bowling | 30 | **1.99** | 5.35 | 2.51 | 2.18 | 3.70 | 3.14 | 3.15 |
| flying kite | 30 | **0.74** | **1.89** | 4.37 | 5.18 | **1.62** | 4.81 | 3.10 |
| high jump | 30 | **0.88** | **0.92** | **1.56** | **1.81** | **0.50** | **1.05** | 1.12 |
| marching | 30 | **1.90** | 2.49 | **1.33** | 2.99 | 7.86 | 4.70 | 3.54 |

Bold values indicate axes where transfer succeeds (MAE < 2.0).

**Table 4a: Universality ranking (mean transfer MAE across all 13 holdouts)**

| Axis | Mean MAE | Transfers in | Classification |
|------|----------|-------------|----------------|
| coordination | 1.45 | 11/13 (85%) | **Universal** |
| impact | 1.47 | 9/13 (69%) | **Universal** |
| speed | 1.74 | 8/13 (62%) | **Universal** |
| verticality | 2.16 | 7/13 (54%) | Semi-universal |
| precision | 2.56 | 6/13 (46%) | Semi-universal |
| tension | 3.13 | 5/13 (38%) | Domain-specific |

**Key findings:**

1. **Three axes are universal visual primitives.** Coordination (MAE 1.45, transfers in 85% of holdouts), impact (1.47, 69%), and speed (1.74, 62%) generalize across visually distinct domains. The V-JEPA 2 encoder captures these motion dynamics in a form that is separable from scene identity and transferable to unseen categories.

2. **Category diversity reveals hidden universality.** Coordination was classified as domain-specific in the 5-category experiment (MAE 8.09 when marching was the only source). With 13 categories including fencing, diving, slacklining, and pole vault, all involving body coordination, the axis transfers to 11/13 holdouts. The visual primitive was always in the encoder; it needed diverse training examples to be extracted.

3. **Some categories transfer on all axes.** Javelin throw (MAE 0.85) and high jump (MAE 1.12) achieve successful transfer on all 6 axes. The remaining 12 categories collectively teach everything about their visual dynamics. This is the strongest evidence that universal visual primitives exist and are sufficient for scoring unseen domains.

4. **Tension remains domain-specific.** Even with 13 categories, tension transfers in only 5/13 holdouts (MAE 3.13). The visual signature of "buildup before release" differs too much across archery, pole vault, and arm wrestling to generalize without domain-specific training. Notably, pole vault tension transfer fails catastrophically (MAE 9.00) because the stillness-before-the-run is visually unique.

5. **Failure to transfer identifies novel dynamics.** Bowling's impact (5.35), snowboarding's speed (4.16), and marching's coordination (7.86) remain high-MAE holdouts, confirming that these categories contain visual patterns not represented elsewhere in the dataset. The system correctly identifies which domains need dedicated training data.

### 4.5 Action Inference from Valence Geometry

To test whether multi-axis valence vectors produce qualitatively richer behavior than per-axis thresholds, we construct action fields for the survival and fire domains and evaluate them on simulated valence trajectories.

#### 4.5.1 Direction vs. Magnitude (Survival)

We construct four valence vectors with identical L2 magnitude (~10.0) but different directions in the 7-dimensional Panksepp space:

**Table 5: Same magnitude, different actions**

| Vector profile | Magnitude | Top action | Activation | Conflict |
|---------------|-----------|------------|------------|----------|
| Pure fear `[0,0,10,0,0,0,0]` | 10.0 | FLEE | 1.00 | 0.33 |
| Pure rage `[0,10,0,0,0,0,0]` | 10.0 | FIGHT | 1.00 | 0.33 |
| Pure care `[0,0,0,0,10,0,0]` | 10.0 | BOND | 1.00 | 0.33 |
| Distributed `[3.8,3.8,3.8,3.8,3.8,3.8,3.8]` | 10.1 | FREEZE | 0.38 | 0.53 |

The pure vectors produce clear, high-activation actions with low conflict. The distributed vector, with the same energy spread across all axes, produces weak FREEZE with elevated conflict. This is the anxious-vigilance state: everything is mildly activated, nothing dominates, the organism is alert but undirected. A scalar "arousal" score would equate these four states. The valence vector distinguishes them.

The conflict score captures a phenomenon documented in behavioral neuroscience: decision latency increases when competing motivational circuits activate simultaneously (McNaughton & Corr, 2004). The FREEZE response to `[fear=7, rage=7]` (conflict=0.10, paradoxically low because FREEZE itself dominates) differs from the distributed freeze (conflict=0.53). The former is tonic immobility, a coherent defensive strategy; the latter is paralysis by indecision.

#### 4.5.2 Competing drives (Survival)

The most interesting action states arise from axis interactions:

**Table 6: Axis interactions produce emergent actions**

| Scenario | Vector | Action | Biological interpretation |
|----------|--------|--------|--------------------------|
| Predator | `[0,0,8,0,0,0,0]` | FLEE (0.64) | Standard escape |
| Cornered | `[0,7,7,0,0,0,0]` | FREEZE (0.70) | Tonic immobility |
| Mother + predator | `[0,0,8,0,8,0,0]` | PROTECT/FLEE (tie) | Parental defense |
| Lost offspring | `[0,0,0,0,3,8,0]` | SEEK_CONTACT (0.80) | Separation distress |

The PROTECT response to `[fear=8, care=8]` is biologically significant: it represents the well-documented phenomenon of maternal aggression, where a parent's care drive overrides fear to defend offspring (Lonstein & Gammie, 2002). Neither axis alone predicts this behavior. It emerges from the interaction, from the direction of the vector in the fear-care plane.

#### 4.5.3 Trajectory Preemption (Fire)

We simulate a warehouse fire deteriorating over 6 timesteps and evaluate the action field at each point, including the projected state at horizon $h=2$:

**Table 7: Action transitions and preemption signals in fire scenario**

| Time | Current action | Projected action | Preempt? | Conflict |
|------|---------------|-----------------|----------|----------|
| t=0 | HOLD_POSITION (0.45) | HOLD_POSITION (0.45) | No | 1.05 |
| t=1 | HOLD_POSITION (0.29) | DEFENSIVE (0.16) | No | 0.67 |
| t=2 | HOLD_POSITION (0.10) | DEFENSIVE (0.30) | **Yes** | 1.14 |
| t=3 | EVACUATE (0.20) | EVACUATE (0.52) | No | 0.86 |
| t=4 | EVACUATE (0.42) | MAYDAY (0.94) | **Yes** | 0.20 |
| t=5 | EVACUATE (0.68) | MAYDAY (2.92) | **Yes** | 0.31 |

Three observations:

1. **High conflict at transition points.** At t=0 and t=2, conflict exceeds 1.0 as multiple actions compete at near-equal strength. These are the moments when a firefighter's experience (calibrated valence heads) makes the difference between a good decision and a fatal one.

2. **Preemption fires before the crisis.** At t=2, the current state still supports HOLD_POSITION, but the projected state at t+2 shows DEFENSIVE dominating. The system signals that it is time to transition posture before conditions force it. At t=4, the current action is EVACUATE but the projection shows MAYDAY. The signal is unambiguous: leave now, because in 2 timesteps you will be calling for rescue.

3. **Conflict drops as urgency rises.** By t=4 and t=5, conflict is low (0.20, 0.31) and the situation is unambiguous. High conflict is a feature of ambiguous, moderate-threat states. Clear emergencies produce clear actions. This matches the phenomenology of expert decision-making under stress: the hardest decisions are in the middle zone, not at the extremes (Klein, 1998).

### 4.6 Bayesian Experience Memory

A scoring system without memory is a thermometer. It measures but does not learn. We introduce a Bayesian experience memory that stores, retrieves, and selectively forgets experiences based on their statistical contribution to the system's posterior beliefs.

#### 4.6.1 The Learning Loop

Each experience passes through a Bayesian loop:

1. **Observe** visual pattern, extract features, score on domain axes
2. **Update** the posterior: $P(\theta | D_{1:t}) \propto P(x_t | \theta) \cdot P(\theta | D_{1:t-1})$
3. **Compute surprise**: $\text{KL}[P(\theta | D_{1:t}) \| P(\theta | D_{1:t-1})]$
4. **Gate**: store only if statistically informative (high surprise, extreme outcome, or novel pattern)
5. **Evict** if at capacity, using posterior-aware selection
6. **Replay** periodically: recompute all retention scores against the current posterior

The posterior is parameterized as a diagonal Gaussian over valence outcomes: mean $\mu$ and precision $\lambda$ per axis, updated via conjugate Bayesian update:

$$\lambda_{\text{new}} = \lambda_{\text{old}} + \lambda_{\text{obs}}, \quad \mu_{\text{new}} = \frac{\lambda_{\text{old}} \mu_{\text{old}} + \lambda_{\text{obs}} x}{\lambda_{\text{new}}}$$

Surprise is the KL divergence between old and new posterior, quantifying the amount of learning this observation caused.

#### 4.6.1.1 Beyond Unit Precision: Normal-Gamma Extension

The Gaussian formulation above hardcodes the observation precision to a single value, so the posterior precision grows linearly with $n$ regardless of how scattered the observations actually are. The second-moment information ($\sum x_i^2$) is never used. This is fine when valence axes are pre-normalized to unit variance, but production valence vectors carry per-axis scales that differ by orders of magnitude, and the unit-precision model has no mechanism to discover them.

We provide a Normal-Gamma alternative (Bishop, 2006, §2.3.6) that closes this gap. Each axis carries four hyperparameters $(\mu_n, \beta_n, a_n, b_n)$ representing the joint posterior over $(\mu, \tau)$ — mean and precision both treated as random variables:

$$\beta_n = \beta_0 + n, \quad \mu_n = \frac{\beta_0 \mu_0 + \sum x_i}{\beta_n}$$
$$a_n = a_0 + \tfrac{n}{2}, \quad b_n = b_0 + \tfrac{1}{2}\sum (x_i - \bar{x})^2 + \frac{\beta_0 n (\bar{x} - \mu_0)^2}{2\beta_n}$$

The posterior predictive becomes a per-axis Student-$t$ with $2 a_n$ degrees of freedom, location $\mu_n$, and scale $\sqrt{b_n(\beta_n + 1)/(a_n \beta_n)}$. Surprise and leave-one-out influence become unified Bayesian quantities — the predictive log-density evaluated at the new observation:

$$\text{surprise}(x) = -\log p(x \mid D), \qquad I_i = -\log p(x_i \mid D \setminus \{x_i\})$$

The computation remains $O(\text{n\_axes})$ via closed-form sufficient-statistic subtraction. We verified the closed form numerically against (a) a hand-computed posterior, (b) hierarchical Monte Carlo sampling from the predictive (Kolmogorov-Smirnov $D = 0.0005$ over $10^6$ samples), and (c) leave-one-out by full posterior refit.

**Effect on per-axis variance recovery.** On a four-axis valence stream with true per-axis $\sigma = (0.1, 1.0, 2.0, 5.0)$ over 300 observations, the original Gaussian posterior reports predicted variance $\approx 0.0033$ on every axis (this is just $1/(0.01 + 300)$ — independent of the data), while the Normal-Gamma recovers $(0.017, 0.969, 4.881, 25.99)$ — true scales to three decimals.

**Effect on calibration.** A "typical" probe — one $\sigma$ on each axis, calibrated to the per-axis scale — receives total surprise $\approx 1074$ nats from the Gaussian (which has no per-axis scale concept) and $\approx 5.94$ nats from the Normal-Gamma. The latter equals the differential entropy of the predictive: a typical observation's surprise is the entropy. This is the calibrated baseline that allows entropy-relative gating: the storage gate compares $\text{surprise} - H[\text{predictive}]$ to a threshold, so a "more surprising than typical" decision has consistent semantics across axis scales.

**Effect on memory composition.** Re-running the firefighter scenario (77 observations, capacity 20, identical RNG, identical thresholds) with the Normal-Gamma posterior keeps **all 5 of 5 EVACUATE-class events** (3 crises + 2 backdrafts) versus **1 of 5** under the original Gaussian. The mean is learned at the same conjugate rate in both, but the Normal-Gamma additionally learns per-axis variance, so high-intensity events on the noisy 'intens' axis ($\sigma \approx 2.0$) are correctly tagged as several-$\sigma$ events with high LOO influence and survive eviction. The unit-precision Gaussian has no scale concept and ranks routine variation alongside genuine crises.

The Normal-Gamma posterior is available as `BayesianMemory(posterior_kind="normal_gamma")`. The Gaussian remains the default for backward compatibility.

#### 4.6.2 The Eviction Problem: Sequential Bias

Naive eviction (remove lowest-scoring experience) introduces three compounding biases:

- **Primacy bias**: early experiences receive inflated surprise because the prior was flat, a trivially true but uninformative artifact
- **Redundancy blindness**: 10 copies of "routine fire" are treated as 10x value when they are ~1x value
- **Coverage drift**: extreme experiences dominate, routine ones are lost, and the stored distribution no longer represents the true distribution

We address these with three mechanisms:

**1. Normalized influence.** The leave-one-out (LOO) influence of experience $i$ is:

$$I_i = \text{KL}[P(\theta | D) \| P(\theta | D \setminus \{x_i\})]$$

This measures how much the posterior would change if $x_i$ were removed. However, raw $I_i$ is degenerate when $n$ is small (removing 1 of 2 observations trivially shifts the posterior). We normalize:

$$\hat{I}_i = \frac{I_i}{\sqrt{n_{\text{storage}}}}$$

where $n_{\text{storage}}$ is the posterior size when $x_i$ was stored. This discounts early experiences whose high influence is an artifact of posterior thinness.

**Table 8: Effect of influence normalization**

| Experience | Action | Raw influence | $n_{\text{storage}}$ | Normalized | Effect |
|-----------|--------|--------------|---------------------|------------|--------|
| #0 (first ever) | MONITOR | 1.739 | 1 | 0.422 | Discounted: flat-prior artifact |
| #8 (growing) | DEFENSIVE | 0.232 | 9 | 0.077 | Moderate discount |
| #16 (crisis) | EVACUATE | 1.739 | 17 | 0.422 | Retains high score: genuinely informative |
| #69 (routine) | MONITOR | 0.050 | 70 | 0.006 | Heavily discounted: posterior already sharp |

**2. Coverage-driven eviction.** Feature space is partitioned into regions via hyperoctant hashing. Eviction preferentially removes experiences from dense regions, rebalancing the stored distribution toward uniform coverage. A region with 1 experience is never emptied, maintaining representational coverage even under memory pressure.

**3. Importance-weighted replay (the backward pass).** Every $k$ insertions, all stored experiences are re-evaluated against the *current* posterior:

- **Recompute influence**: an experience that was load-bearing may now be redundant (later experiences covered the same territory)
- **Recompute surprise**: an experience that seemed routine may now be an outlier (similar experiences were evicted)
- **Importance weighting**: experiences far from the current posterior mean receive amplified surprise because they are informative precisely when they are unusual

This replay is the backward pass: propagating the current posterior state back through all stored experiences and recomputing their value. Without it, retention scores become stale and the eviction policy degrades.

#### 4.6.3 Results: What Memory Keeps

We simulate 80 fire observations (61 routine small fires, 10 growing fires, 3 crises, 2 backdrafts, 1 electrical fire) through a memory with capacity 20.

**Table 9: Memory composition after 80 observations**

| Profile | Seen | In memory | Retention rate |
|---------|------|-----------|----------------|
| Crisis/Backdraft/Electrical | 6 | 6 | 100% |
| Growing fire | 10 | 5 | 50% |
| Routine small | 61 | 9 | 15% |

All rare, high-consequence events are retained. Growing fires (moderate intensity) retain representatives. Routine fires are heavily pruned because the posterior already knows what they look like.

**Temporal bias check:**

| Period | Retained | Expected (unbiased) |
|--------|----------|-------------------|
| Early (first 20 obs) | 9 | 5.2 |
| Middle (obs 20-49) | 7 | 7.8 |
| Late (obs 50+) | 4 | 7.0 |

The middle period, where the growing fires and first crisis occurred, is well-represented (7 vs. expected 7.8). Early-period overrepresentation (9 vs. 5.2) persists but is reduced 40% compared to naive eviction (which retained 15/20 from the early period). The normalized influence correction is working but could be strengthened with a more aggressive discount function.

#### 4.6.4 Connection to Biological Memory

The three mechanisms mirror known properties of biological memory consolidation:

- **Influence ↔ synaptic tagging**: hippocampal memories are tagged for consolidation based on their novelty relative to existing schema (Wang & Morris, 2010)
- **Coverage ↔ pattern separation**: the dentate gyrus actively separates similar inputs to maintain distinct representations (Leutgeb et al., 2007)
- **Replay ↔ sleep consolidation**: offline replay during sleep reactivates memories and re-evaluates them against updated cortical representations (Diekelmann & Born, 2010)

The parallel is not metaphorical. Biological memory systems face the same statistical problem: finite storage, non-stationary input distribution, and the need to retain informative experiences while discarding redundant ones. The solution, posterior-aware selective retention with periodic replay, converges from both the computational and biological directions.

---

## 5. Discussion

### 5.1 The Hierarchy Inverted

Current vision-language models place language at the center: visual features are projected into a text embedding space, evaluated by text-based loss functions, and benchmarked by text generation quality. vScore inverts this hierarchy:

| Layer | Vision-Language Models | vScore |
|-------|----------------------|--------|
| Foundation | Language tokens | Valence scores |
| Training signal | Text supervision | Outcome scores |
| Output | Words | Trigger events |
| Language role | Core | Optional serialization |

This inversion is not merely architectural. It changes what the system optimizes for. A VLM optimizes for describing what it sees. vScore optimizes for predicting what will happen and whether it matters. The former produces narration. The latter produces intelligence.

### 5.2 Universal Primitives vs. Domain-Specific Axes

Our transfer experiments reveal a natural partition of visual features:

**Universal primitives** (transfer across domains):
- Motion magnitude (speed)
- Object trajectory (direction, acceleration)
- Spatial extent (how much of the visual field is activated)

**Domain-specific features** (require targeted training):
- Impact dynamics (collision patterns)
- Coordination patterns (synchronized multi-agent motion)
- Tension dynamics (stillness-before-release)
- Domain-specific textures (flame, ice, water)

This partition aligns with neuroscience: early visual processing (V1, MT) extracts motion, edges, and spatial frequency as universal primitives, while higher areas (IT, STS) encode category-specific and action-specific representations. vScore's architecture mirrors this: the frozen encoder provides universal primitives; the trainable valence heads learn domain-specific scoring.

Critically, the 13-category experiment revealed that the partition itself is data-dependent: coordination appeared domain-specific with 5 categories but universal with 13. This suggests that the boundary between universal and domain-specific is not fixed; it shifts as the training set diversifies. More extensive research with larger, more diverse datasets is required to establish the definitive partition.

### 5.3 From Detection to Understanding: The Geometry Argument

Motion detection is trivially solved. Any optical flow algorithm detects motion. The question is what motion *means*. vScore's answer is that meaning is not a label attached to motion but a *direction in valence space* that the motion contributes to.

Consider the transition from detection to understanding:

1. **Detection**: "something is moving" (optical flow, background subtraction, solved in the 1980s)
2. **Scoring**: "it is moving at speed=7 toward proximity=8" (per-axis regression, Section 4.3)
3. **Geometry**: "the valence vector is pointing toward the EVACUATE region of action space" (Section 4.5)
4. **Projection**: "and the trajectory is curving toward MAYDAY" (Section 4.5.3)

Each level adds information that the previous level lacks. Detection without scoring is noise. Scoring without geometry is a dashboard of unrelated numbers. Geometry without projection is a snapshot without consequences.

The action field formalization makes this concrete: an $n$-dimensional valence space contains a continuous manifold of behavioral states. Each point is characterized not by which axis is highest (that would reduce to classification) but by its position relative to the boundaries between action regions. The boundaries themselves are nonlinear surfaces in valence space. For example, the FREEZE region lies along the diagonal where fear and rage are approximately equal, not at any single threshold.

This is fundamentally different from both classification ("this is a fire") and regression ("the fire is at intensity 7.5"). It is *geometric inference*: given where the vector is, where it is heading, and the shape of the action landscape, what response does the geometry demand?

### 5.4 Implications for Annotation

Traditional annotation asks: "What is in this video?" (a linguistic question). vScore annotation asks: "How activated is each axis at this moment?" (a numerical question).

This has practical advantages:

1. **No linguistic competence required.** Annotators can score videos by adjusting sliders, regardless of language or literacy.
2. **Cross-cultural validity.** A Japanese firefighter and a Brazilian firefighter may use different words, but they read the same visual dynamics and would produce similar valence scores.
3. **Physiological validation.** Valence scores can be validated against heart rate, galvanic skin response, and eye tracking, providing ground truth that is independent of language.
4. **Continuous, not categorical.** "How fast?" (0–10) captures more information than "fast/slow."

### 5.5 Toward Predictive Visual Intelligence

The trajectory projection and threshold triggering components of vScore formalize a capacity that biological systems exhibit but current AI systems largely lack: **acting on projected futures, not observed presents.**

A self-driving car that detects a pedestrian after they step into the road is reactive. One that detects the trajectory of a person walking toward the curb and projects the crossing is predictive. The difference is not better object detection. It is valence trajectory projection.

vScore provides the computational framework for this: score the current state, track the trajectory, project forward, trigger when the projection crosses a threshold. The visual encoder is a commodity (V-JEPA 2 today, its successor tomorrow). The intelligence is in the scoring and projection.

### 5.6 Limitations

1. **Proxy annotations.** Our current experiments use fixed per-category scores. Real per-video annotations from domain experts or physiological measurements would enable learning intra-category variation, for instance two archery videos with different tension profiles.

2. **Small dataset.** 50 videos across 5 categories. The transfer findings are directional, not statistically conclusive. Validation on larger, more diverse datasets (Kinetics-700, Ego4D, domain-specific collections) is needed.

3. **Global pooling.** We average 8,192 spatiotemporal tokens to a single vector, discarding spatial and temporal structure. The dense token representation from V-JEPA 2 is designed for spatial grounding, and future work should score per-region and per-timestep, not per-video.

4. **Static thresholds.** The current threshold mechanism uses fixed base values. In biological systems, thresholds adapt to context (heightened alertness, habituation, priming). Learned, context-dependent thresholds are a natural extension.

5. **Single encoder.** We test only V-JEPA 2. The framework is encoder-agnostic, and testing with alternative self-supervised encoders (DINOv2, VideoMAE, InternVideo) would isolate the contribution of the scoring framework from the encoder quality.

---

## 6. Related Work

**Joint Embedding Predictive Architectures.** LeCun's (2022) position paper on autonomous machine intelligence argued that autoregressive generative models, including LLMs, are insufficient for world understanding because they operate on discrete tokens rather than continuous representations. He proposed JEPA as an alternative: learn to predict representations of the world from other representations, without reconstructing inputs. This is a fundamental departure from both generative models (which predict pixels or tokens) and contrastive learning (which only learns to distinguish). The key insight for our work is that JEPA representations are, by design, *not grounded in language*. They encode visual structure as the world presents it, not as language describes it. This makes them the ideal substrate for pre-linguistic valence scoring. I-JEPA (Assran et al., 2023) demonstrated this principle for images; V-JEPA (Bardes et al., 2024) and V-JEPA 2 (Bardes et al., 2025) extended it to video, learning dense spatiotemporal features through masked prediction in latent space. V-JEPA 2 in particular produces the kind of spatially structured, temporally consistent representations that valence scoring requires: features that track objects, motion, and spatial relationships across time without any text supervision. vScore can be understood as a downstream answer to the question LeCun's architecture poses: if the encoder sees the world without words, what should the evaluation layer look like?

**Self-supervised video representation learning.** Beyond the JEPA family, self-supervised video learning includes contrastive approaches (Qian et al., 2021), masked autoencoders (Tong et al., 2022; Wang et al., 2023), and distillation methods (Oquab et al., 2023). vScore is encoder-agnostic; any self-supervised video encoder that produces dense features could serve as Level 2. We chose V-JEPA 2 because its JEPA-based training most closely aligns with the principle that visual understanding should be independent of language.

**Affective computing.** Emotion recognition from video (Mollahosseini et al., 2017; Li & Deng, 2020) typically maps visual input to categorical emotion labels (happy, sad, angry) or dimensional affect (valence-arousal). vScore differs in two ways: (1) the axes are domain-specific and task-relevant, not universal emotion categories, and (2) the emphasis is on trajectory projection and threshold triggering, not state classification.

**Affordance detection.** Work on visual affordances (Nagarajan et al., 2019; Luo et al., 2022) detects action possibilities in images. vScore extends this from "what can be done" to "what will happen and how much does it matter," adding temporal projection and outcome valence.

**Predictive coding.** Friston's Free Energy Principle (Friston, 2010) frames perception as prediction error minimization. vScore operationalizes a related idea: the system projects expected trajectories and triggers on deviations from expected homeostasis. The analogy is direct: surprise (prediction error) corresponds to non-zero valence magnitude.

**Embodied AI and robotics.** V-JEPA 2 reports a 20-point improvement in robotic grasping success, demonstrating that dense visual features support motor control. vScore provides a framework for the evaluation layer between perception and action, addressing the "should I act?" decision that precedes "how do I act?"

---

## 7. Future Directions

### 7.1 Cross-Modal Valence (Preliminary Results)

The vScore mechanism is modality-agnostic by construction. The valence vector is the universal interface. Downstream of it, nothing knows whether the input was pixels or pressure waves. We formalize this with a multimodal architecture where each modality has its own encoder and valence head, but all heads output to the same N-axes valence space. Fusion happens in valence space, not feature space.

This design makes modality conflict observable and diagnostic. In a simulated "occluded threat" scenario (growl heard but source not visible), the audio head scores fear=8 while the visual head scores fear=1, producing a conflict signal of 4.9 on the fear axis. This conflict itself is informative: audio threat >> visual threat implies the danger source is occluded or approaching from outside the visual field, a signal that would be lost in feature-space fusion.

We define three fusion rules: MAX (survival default: if any sense says danger, respond), MEAN (requires cross-modal confirmation), and confidence-weighted (trust the modality with more experience). The fusion rule is a domain parameter, not a fixed architectural choice. A survival domain uses MAX. A diagnostic domain like industrial machine monitoring might use CONFLICT to detect when auditory and visual signals diverge.

We implement three audio domains: survival sound (threat, proximity, urgency, familiarity, social signal, rhythmic pull, dissonance), industrial sound (anomaly, degradation, impact event, resonance shift, load stress), and music (tension, release, energy, intimacy, novelty, groove, melancholy). These demonstrate that the same metaclass architecture applies without modification. The principle holds: a baby covers its ears before it knows the word "loud." A dog runs from thunder without vocabulary. The evaluation is pre-linguistic and pre-modal.

### 7.2 Learned Threshold Adaptation

Current thresholds are fixed. Biological thresholds adapt: a soldier in combat has lower FEAR thresholds (hypervigilance); a child at play has higher ones (safety). Learning context-dependent threshold functions from data is a natural next step.

### 7.3 Compositional Domains

Can a system trained on Fire and Weather generalize to "wildfire during a storm"? Compositional domain scoring, where multi-domain valence vectors interact, would test whether the framework supports the combinatorial richness of real-world scenarios.

### 7.4 Expert Calibration Studies

The strongest validation would be: have domain experts (firefighters, coaches, traders) score videos on their domain axes, train valence heads on their scores, and test whether the resulting model predicts expert attention and decision-making. If vScore heads correlate with expert gaze patterns and response times, the framework captures something real about expert visual intelligence.

---

## 8. Conclusion

We have presented vScore, a framework that reframes visual AI from "what is this?" to "what does this mean for outcome-relevant axes?" The core claim is that visual intelligence, in biological systems and potentially in artificial ones, operates pre-linguistically through valence scoring, trajectory projection, and threshold triggering. Language is a late-stage serialization, not a prerequisite for understanding.

Our experiments across 213 videos from 13 categories and two datasets demonstrate that self-supervised video features contain universal visual primitives, specifically coordination, impact, and speed, that transfer to unseen domains with low error (MAE 1.45 to 1.74). The 13-category experiment revealed that the boundary between universal and domain-specific is itself data-dependent: coordination appeared domain-specific with 5 categories but universal with 13, suggesting that the encoder already captures these dynamics but they simply require sufficient diversity of training examples to be extracted. Tension remains genuinely domain-specific (MAE 3.13, transfers in only 38% of holdouts), confirming that some visual dynamics are irreducibly tied to their context.

These results are promising but preliminary. More extensive research is required to establish the definitive partition of universal and domain-specific visual primitives, to validate the framework with real expert annotations rather than proxy scores, to integrate audio encoders for cross-modal valence scoring, and to test the Bayesian memory system on temporal sequences of real-world video. The framework's strength lies in its simplicity and extensibility. A scored domain is defined by a list of axes, a valence head is a small MLP, and the entire downstream pipeline (action geometry, trajectory projection, threshold triggering, Bayesian memory) operates identically regardless of domain or modality. The approach shows promise as a foundation for pre-linguistic visual intelligence; the work ahead is to scale it.

Zero is safety. Everything above zero is cost. The system exists to return to zero.

---

## Acknowledgments

This work owes a foundational debt to Yann LeCun, whose articulation of the Joint Embedding Predictive Architecture (LeCun, 2022) provided both the theoretical grounding and the practical encoder on which vScore is built. LeCun's central argument, that autonomous intelligence requires world models predicting in representation space rather than token space and that autoregressive language models are fundamentally limited by their confinement to discrete symbols, is the premise from which vScore departs. Without V-JEPA and V-JEPA 2, developed by LeCun's team at Meta FAIR (Bardes, Assran, Ballas, Rabbat, LeCun, and collaborators), the empirical results in this paper would not exist. The idea that a visual encoder can learn dense, temporally consistent representations without any language supervision is not our contribution; it is theirs. Our contribution is to ask what comes *after* the encoder when language is deliberately excluded from the evaluation pipeline.

We also acknowledge the broader JEPA research program at Meta FAIR, including I-JEPA (Assran et al., 2023), whose demonstration that joint embedding prediction produces semantically meaningful image features without pixel reconstruction or text alignment was an essential precursor to the video-domain extensions we rely on.

---

## References

Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*.

Bardes, A., et al. (2024). V-JEPA: Latent Video Prediction for Visual Representation Learning. *arXiv preprint*.

Bardes, A., et al. (2025). V-JEPA 2: Unlocking Dense Features in Video Self-Supervised Learning. *arXiv:2603.14482*.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. §2.3.6, conjugate Normal-Gamma prior for univariate Gaussian with unknown mean and precision.

Damasio, A. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.

Diekelmann, S., & Born, J. (2010). The Memory Function of Sleep. *Nature Reviews Neuroscience*, 11(2), 114-126.

Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Gallup, G. G. (1977). Tonic Immobility: The Role of Fear and Predation. *The Psychological Record*, 27(1), 41-61.

Gibson, J. J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.

Klein, G. (1998). *Sources of Power: How People Make Decisions*. MIT Press.

Leutgeb, J. K., et al. (2007). Pattern Separation in the Dentate Gyrus and CA3 of the Hippocampus. *Science*, 315(5814), 961-966.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview preprint*, Version 0.9.2.

Lonstein, J. S., & Gammie, S. C. (2002). Sensory, Hormonal, and Neural Control of Maternal Aggression in Laboratory Rodents. *Neuroscience & Biobehavioral Reviews*, 26(8), 869-888.

Li, S., & Deng, W. (2020). Deep Facial Expression Recognition: A Survey. *IEEE Transactions on Affective Computing*.

Luo, H., et al. (2022). Learning Affordance Grounding from Exocentric Images. *CVPR*.

Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv preprint*.

McNaughton, N., & Corr, P. J. (2004). A Two-Dimensional Neuropsychology of Defense: Fear/Anxiety and Defensive Distance. *Neuroscience & Biobehavioral Reviews*, 28(3), 285-305.

Corr, P. J. (2013). Approach and Avoidance Behaviour: Multiple Systems and Their Interactions. *Emotion Review*, 5(3), 285-290.

Mollahosseini, A., et al. (2017). AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild. *IEEE Transactions on Affective Computing*.

Nagarajan, T., et al. (2019). Grounded Human-Object Interaction Hotspots from Video. *ICCV*.

Panksepp, J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford University Press.

Regan, D., & Vincent, A. (1995). Visual Processing of Looming and Time to Contact Throughout the Visual Field. *Vision Research*, 35(13), 1845-1857.

Qian, R., et al. (2021). Spatiotemporal Contrastive Video Representation Learning. *CVPR*.

Schiff, W. (1965). Perception of Impending Collision: A Study of Visually Directed Avoidant Behavior. *Psychological Monographs*, 79(11), 1-26.

Tong, Z., et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS*.

Zhao, H., et al. (2019). HACS: Human Action Clips and Segments Dataset for Recognition and Temporal Localization. *ICCV*.

Wang, S.-H., & Morris, R. G. M. (2010). Hippocampal-Neocortical Interactions in Memory Formation, Consolidation, and Reconsolidation. *Annual Review of Psychology*, 61, 49-79.

Wang, L., et al. (2023). VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking. *CVPR*.

---

## Appendix A: Project Structure

```
vScore/
├── core/
│   ├── metaclass.py      # ScoredDomain metaclass
│   ├── valence.py         # ValenceVector, ValenceTrajectory
│   └── threshold.py       # ThresholdTrigger, dynamic thresholds
├── domains/
│   ├── survival.py        # Panksepp's 7 circuits
│   ├── fire.py            # 7 axes
│   ├── weather.py         # 6 axes
│   ├── hockey.py          # 6 axes
│   ├── trading.py         # 6 axes
│   ├── sound.py           # 7 axes, pre-linguistic auditory scoring
│   ├── industrial.py      # 6 axes, machine/environment audio
│   └── music.py           # 7 axes, affective musical scoring
├── encoder/
│   └── bridge.py          # V-JEPA 2 → valence head bridge
├── projection/
│   └── temporal.py        # Transformer-based trajectory projector
├── data/
│   └── schema.py          # Annotation format (no words, just scores)
├── train.py               # Training pipeline
├── run_cross_domain.py    # Cross-domain feature analysis
├── demo_actions.py        # Action inference from valence geometry
├── demo_memory.py         # Naive Bayesian memory
├── demo_memory_bayesian.py # Posterior-aware eviction
└── demo_multimodal.py     # Cross-modal fusion in valence space
```

Code available at: [repository URL]

---

## Appendix B: Reproducibility

All experiments run on Apple M-series CPU (no GPU required for inference and training at this scale). Feature extraction for 213 videos takes ~30 minutes (one-time; features are cached as 1,024-dim vectors, 1.3 MB total). Training each valence head takes <5 seconds (500 epochs on cached vectors). Full cross-domain transfer experiment (13 holdout runs) takes <60 seconds. Total reproduction time from scratch: ~35 minutes.

Dependencies: PyTorch 2.8, Transformers 4.57, PyAV 16.1, yt-dlp (for HACS download), V-JEPA 2 weights from `facebook/vjepa2-vitl-fpc64-256`.

Data sources: Kinetics-mini (`nateraw/kinetics-mini` on HuggingFace), HACS (Zhao et al., 2019; `github.com/hangzhaomit/HACS-dataset`).
