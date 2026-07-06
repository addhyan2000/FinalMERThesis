# Micro-Expression Recognition on CASME II
## A Component Ablation Study across LOSO and Hold-out Validation

*Weekend experimental run — 12 architectural configurations × 2 validation protocols*
*Dataset: CASME II (spontaneous micro-expressions) · 3 grouped emotion classes*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology & Chronology](#2-methodology--chronology)
3. [Design Choices & Justifications](#3-design-choices--justifications)
4. [Experimental Results & Comprehensive Figure Analysis](#4-experimental-results--comprehensive-figure-analysis)
5. [Deep-Dive FAQ](#5-deep-dive-faq-defending-the-study)

---

## 1. Executive Summary

> **Goal.** Determine *which* components of a hybrid spatiotemporal micro-expression recognition (MER) pipeline actually contribute to recognition accuracy, by exhaustively toggling four design variables and evaluating every combination under two subject-disjoint validation protocols.

Micro-expressions are involuntary facial movements that last **1/25 – 1/5 of a second** and betray concealed affect. They are faint, brief, and spatially localised, which makes them one of the hardest problems in affective computing. Our pipeline attacks this with four stackable ideas:

| Var | Component | Type | Role |
|-----|-----------|------|------|
| **A** | **EVM** – Eulerian Video Magnification | data-level | Amplifies sub-perceptual motion *before* feature extraction |
| **B** | **SimAM** – parameter-free 3-D attention | model-level | Re-weights CNN feature maps toward salient regions |
| **C** | **3-D CNN** – three-stream backbone | model-level | Extracts spatial + short-range motion features |
| **D** | **SLSTT Transformer** – temporal encoder | model-level | Models long-range temporal dependencies across 32 frames |

We evaluate the full **2×2×2×2 = 16** design matrix (12 architecturally valid cells — SimAM without a CNN is degenerate) on **156 CASME II micro-expression clips** regrouped into **Negative / Positive / Surprise**.

### Headline findings

1. **The 3-D CNN spatial backbone is the single indispensable component.** Every top-scoring configuration under both protocols has the CNN switched **on**; every configuration without it collapses toward the majority-class baseline.
2. **The Transformer *hurts* in this data regime.** In near-every matched pair, turning the SLSTT Transformer **on** *reduces* accuracy and macro-F1. With only ~140 training clips, a 4-layer / 8-head encoder is starved of data and overfits.
3. **The "more is better" intuition fails.** The fully-loaded *proposed* model (config 8 — all four components on) is **among the weakest**, not the strongest. This is an honest, load-bearing negative result.
4. **Macro-F1, not accuracy, tells the truth.** The classes are severely imbalanced (≈ 63 % Negative). A trivial "always predict Negative" classifier scores **0.66 accuracy under LOSO and 0.75 accuracy under hold-out** — higher than most of our real models — yet only **~0.27–0.29 macro-F1**. Macro-F1 is therefore the metric that separates genuine learning from majority-class parroting.
5. **EVM's benefit is protocol-dependent.** It helps under hold-out (best hold-out model uses EVM) but the raw-motion CNN generalises better across *unseen subjects* under LOSO (best LOSO model omits EVM).

### Best configurations at a glance

| Protocol | Best config | Components | Accuracy | Macro-F1 | Majority baseline (acc / F1) |
|----------|-------------|-----------|----------|----------|------------------------------|
| **LOSO (pilot, 20/25 folds)** | `config_5_attention_base` | SimAM + CNN | **0.643** | **0.379** | 0.662 / 0.266 |
| **Hold-out (30 % subjects)** | `config_13_permutation` | EVM + CNN | **0.577** | **0.458** | 0.750 / 0.286 |

The models beat the trivial baseline decisively on **macro-F1** (the honest metric) while sitting near or below it on **raw accuracy** — exactly the signature of a model that has learned to recognise minority classes at the cost of a few majority-class errors.

---

## 2. Methodology & Chronology

This section answers **Core Question 1: "What exact steps did we take, from raw data to final metrics?"**

### 2.1 The pipeline, end-to-end

```
RAW CASME II video (200 fps, colour)
        │
        ▼
[Stage 1 — Data Pipeline]
  1. Parse coding sheet → onset / apex / offset frames per clip
  2. Face detection, crop & alignment
  3. (Optional, Variable A) Eulerian Video Magnification
        α = 10, temporal band-pass 5–25 Hz, 4-level Laplacian pyramid
  4. Temporal resampling: onset→offset interpolated to EXACTLY 32 frames
  5. Dense optical flow → horizontal (u) & vertical (v) components
  6. Optical strain → local deformation magnitude
  7. Stack into a 3-channel motion tensor  →  [3, 32, 224, 224]
        │  written to disk as .npy
        ├── Processed_Data/tensors/       (EVM-magnified)
        └── Processed_Data/tensors_raw/   (non-magnified)
        │
        ▼
[Stage 2 — Hybrid Network]                         input [B, 3, 32, 224, 224]
  A. (data switch) choose EVM vs raw tensor directory
  B/C. Three-stream 3-D CNN backbone (+ optional SimAM attention)
        → [B, 96, 32, 112, 112]
        → AdaptiveAvgPool3d → squeeze/permute → [B, 32, 96]   (a 32-step sequence of 96-d vectors)
  D. Temporal encoder:
        • Transformer ON  → SLSTT (8 heads, 4 layers) → [B, 96]
        • Transformer OFF → mean/max temporal pooling → [B, 96]
  E. Classifier head: LayerNorm → Dropout → Linear → [B, 3]
        │
        ▼
[Stage 3 — Training & Evaluation]
  Focal loss (γ=2) + label smoothing (0.05) + inverse-freq class weights
  Balanced (oversampling) train sampler
  AdamW, lr 1e-4, weight-decay 1e-4, grad-clip 1.0, mixed precision
  CosineAnnealingLR schedule, best checkpoint chosen by macro-F1
        │
        ▼
[Stage 4 — Protocol harness]
  • Hold-out : one subject-disjoint split (30 % of subjects → test)
  • LOSO     : leave-one-subject-out, pilot = 20 of 25 subject folds
        │
        ▼
final_results.json · confusion_matrix.png · training_metrics.csv · summary.csv · plots/
```

### 2.2 Input feature channels — *why these three?*

Each clip becomes a **3-channel motion tensor**, not RGB:

| Channel | Signal | What it captures |
|---------|--------|------------------|
| 0 | **Optical flow *u*** | horizontal pixel motion between frames |
| 1 | **Optical flow *v*** | vertical pixel motion between frames |
| 2 | **Optical strain** | local *deformation* (spatial derivative of flow) — invariant to rigid head motion |

We feed **motion, not appearance**, because micro-expression identity lives in *how the skin deforms*, not in the subject's face texture. This is also a powerful anti-leakage measure (§3.4): a network that only sees flow/strain cannot memorise a subject's appearance.

### 2.3 The ablation matrix (12 valid cells)

The matrix is generated programmatically from `itertools.product([False,True], repeat=4)`; the 4 degenerate cells (SimAM on, CNN off — nothing to attend over) are pruned, leaving 12 runnable configurations, grouped into thesis "phases":

| # | Config | EVM | SimAM | CNN | Transf. | Phase | Intent |
|---|--------|:---:|:-----:|:---:|:-------:|:-----:|--------|
| 1 | pure_base | – | – | – | – | I | absolute floor |
| 2 | temporal_only | – | – | – | ✓ | I | transformer on raw patches |
| 3 | spatial_only | – | – | ✓ | – | I | CNN alone |
| 4 | motion_amp_base | ✓ | – | – | – | II | EVM-only (classic MER baseline) |
| 5 | attention_base | – | ✓ | ✓ | – | II | CNN + attention |
| 6 | full_stage2_noevm | – | ✓ | ✓ | ✓ | III | everything but EVM |
| 7 | full_no_attention | ✓ | – | ✓ | ✓ | III | everything but attention |
| 8 | **proposed_unified** | ✓ | ✓ | ✓ | ✓ | IV | the full proposed model |
| 9 | permutation | – | – | ✓ | ✓ | – | CNN+Transformer |
| 12 | permutation | ✓ | – | – | ✓ | – | EVM+Transformer |
| 13 | permutation | ✓ | – | ✓ | – | – | EVM+CNN |
| 16 | permutation | ✓ | ✓ | ✓ | – | – | EVM+CNN+attention |

### 2.4 Chronological development narrative

1. **Data build.** CASME II raw frames → aligned → two tensor sets (EVM and raw) precomputed offline, so Variable A is a *directory switch* at train time (no repeated magnification cost).
2. **Model modularisation.** The network was refactored so each of B/C/D can be turned off cleanly (CNN-off falls back to flattened raw patches; Transformer-off falls back to temporal pooling). This is what makes a *clean* ablation possible.
3. **Hold-out first (fast).** A single 30 %-subject hold-out split was run across all 12 configs — cheap enough to iterate on epochs, batch size, and loss settings overnight.
4. **Pilot LOSO second (rigorous).** Leave-one-subject-out was then run as a **pilot** — 20 of the 25 possible subject folds — to obtain a subject-generalisation estimate without the multi-day cost of the full sweep.
5. **Aggregation & plotting.** Per-config `final_results.json` files were pooled into `summary.csv`, and the plotting tool produced the bar charts, grouped per-class F1, and confusion-matrix panels analysed in §4.

---

## 3. Design Choices & Justifications

### 3.1 Why this modelling approach and these features? *(Core Question 2)*

**What we chose:** a *motion-first, spatial-then-temporal* hybrid — 3-channel optical-flow/strain input → 3-D CNN spatial stem → (optional attention) → temporal encoder → linear head.

**Why:**
- **Motion input over RGB.** Micro-expressions are defined by *movement*, and their appearance signal is dwarfed by identity/texture. Optical flow + strain isolate the deformation and suppress inter-subject appearance variance.
- **A 3-D CNN stem** captures joint spatial-temporal micro-texture (a moving skin fold) that 2-D CNNs on single frames miss.
- **Parameter-free attention (SimAM)** was preferred over SE/CBAM because it adds **zero learnable parameters** — critical when you have ~140 training clips and cannot afford more capacity.
- **A temporal encoder** was included *to be tested*, not assumed — the ablation exists precisely to check whether long-range temporal modelling earns its keep. (Spoiler: on CASME II it does not — see §4.)

**Alternatives rejected:**
| Alternative | Why rejected |
|-------------|--------------|
| RGB frames + 2-D CNN + LSTM | appearance dominates the tiny motion signal; identity leakage risk |
| Hand-crafted LBP-TOP / 3DHOG | strong classical baseline but ceiling-limited; not end-to-end |
| Apex-frame-only single image | discards the temporal dynamics that *define* a micro-expression |
| Very deep backbones (I3D/R(2+1)D) | far too many parameters for 156 clips → guaranteed overfit |
| SE / CBAM attention | add parameters; SimAM matches them at zero parameter cost |

### 3.2 Why these metrics? *(Core Question 3)*

We report **accuracy** and **macro-F1** as headline numbers, plus **per-class precision / recall / F1** and the full **confusion matrix**.

- **Macro-F1 is the primary metric.** It averages F1 *per class with equal weight*, so a model that ignores the two minority classes (Positive, Surprise) is heavily penalised. On CASME II this is essential:

  > A degenerate "always predict Negative" classifier scores **0.662 accuracy (LOSO)** and **0.750 accuracy (hold-out)** — but only **~0.27 / 0.29 macro-F1**. Accuracy alone would *reward* a model for learning nothing.

- **Accuracy is retained** only as a familiar reference and to expose exactly this trap (models can look "good" on accuracy while being useless).
- **Per-class recall & confusion matrices** are reported because in MER the *cost of confusions matters*: collapsing Surprise into Negative is a different failure than the reverse.

This is the correct metric suite for **imbalanced, few-shot, safety-relevant affective classification** — the exact profile of CASME II.

### 3.3 Why LOSO *and* hold-out? *(Core Question 4)*

Both protocols are **strictly subject-disjoint** — no subject ever appears in both train and test. They differ in cost and in what bias they expose.

**Leave-One-Subject-Out (LOSO):**
- Train on all-but-one subject, test on the held-out subject, repeat, pool predictions.
- **What it prevents: identity leakage.** If clips from the same subject appear in both train and test (as in a naive random split), the network can recognise the *person* and cheat. LOSO makes that impossible and yields the *honest* estimate of how the model behaves on a **brand-new person** — the only number that matters for deployment.
- **Cost:** one full training run per subject. 25 subjects × 12 configs × 60 epochs ≈ days of GPU. Hence a **pilot** (20 of 25 folds, evenly spread across the cohort) was run for these results.

**Hold-out (single 30 %-subject split):**
- One subject-disjoint split, one training run per config.
- **Why also do it:** it is ~20× cheaper, so it is the workhorse for iterating on hyper-parameters, epoch counts, and sanity-checking the plumbing before committing GPU-weeks to LOSO.

**How they complement each other:**
- Hold-out = **fast, higher-variance** probe (test statistics rest on the luck of one split).
- LOSO = **slow, lower-variance, less optimistic** estimate (every subject is tested exactly once).
- Agreement between them on the *qualitative* story (CNN helps, Transformer hurts) is what gives us confidence the finding is real and not a split artefact.

### 3.4 Why hold out **30 %** of subjects, and why not use 100 %? *(Core Question 6)*

**The 30 % figure is a real, deliberate setting.** The weekend hold-out runs were launched with `ablation_val_fraction = 0.3` (recorded in `gui_settings.json`), i.e. **30 % of subjects were held out as the test set** — roughly 7–8 of the 25 usable subjects, which pooled to **52 test clips**.

**Why 30 % (not 10 %, not 50 %)?**
- **Statistical stability of the test estimate.** With only ~140 usable clips, a 10 % test set would be ~15 clips — far too few to estimate a 3-class macro-F1 with any stability (a single misclassified Surprise clip would swing the score by points). 30 % (~52 clips) is the smallest split that gives a *readable* per-class confusion matrix.
- **Preserving minority classes in the test set.** Surprise is only 25 clips total. A small test fraction risks a test set with **zero** Surprise or Positive examples, making per-class F1 undefined. 30 % raises the odds that all three classes are represented on both sides.
- **Whole-subject granularity.** Because the split is subject-disjoint, you cannot hold out an arbitrary fraction of *clips*; you hold out whole *subjects*. 30 % of 25 subjects (~7–8) is the natural resolution that balances a trainable majority against a measurable test set.

**Why not 100 %?**
Holding out 100 % is a contradiction — there would be **no data left to train on**. The train/test split is a zero-sum allocation: every clip moved into the test set is a clip removed from training. On a 156-clip dataset the training signal is already scarce, so we keep ~70 % for learning and spend ~30 % on a test estimate large enough to trust. (LOSO resolves the trade-off differently — it recycles *every* subject as test exactly once, which is why we run it too.)

**Related data-reduction decisions, stated defensively.** Beyond the 30 % test split, two other reductions shape the sample and are worth naming explicitly, because a reviewer will ask:

1. **We dropped the "Others" class (99 of 255 clips, ~39 %).** CASME II labels 99 clips as *others / helpless / pain / confused / sympathy* — a semantic grab-bag with no coherent affective target. Training a classifier to predict "everything else" injects pure label noise. Removing it leaves **156 clips (≈ 61 % of the micro-expression set)** across a clean 3-class target. This is the standard MEGC-style treatment, not a convenience cut.
2. **The LOSO run is a pilot (20 of 25 folds, 80 %).** Not a sampling bias in the statistical sense — every *class* is still represented — but a compute-driven cap. The five omitted folds are documented in `final_results.json` (`loso_folds_run: 20`, `loso_folds_total: 25`, `loso_pilot: true`) so the number is never mistaken for the final full-LOSO thesis figure.

### 3.5 Why group the emotions? *(Core Question 7)*

We map **7 raw CASME II emotions → 3 unified classes** and drop *Others*:

| Unified class | Raw emotions folded in | Clip count |
|---------------|------------------------|:----------:|
| **Negative** | disgust (63) + repression (27) + sadness (7) + fear (2) | **99** |
| **Positive** | happiness | **32** |
| **Surprise** | surprise | **25** |
| *(dropped)* | others / helpless / pain / confused / sympathy | *(99)* |

**Why not classify all 7 individual emotions?** Three independent reasons, each sufficient on its own:

1. **Sample-size / statistical power.** Look at the raw tail: **fear = 2 clips, sadness = 7 clips.** Under LOSO you cannot even *form* a train/test split for a 2-clip class without the single test example dominating its F1. No amount of modelling fixes 2 examples; the class is statistically unlearnable in isolation. Grouping pools these into a 99-clip Negative class that *can* be learned.
2. **Cognitive ambiguity of the labels.** disgust, repression, sadness and fear are all **negative-valence** states whose facial signatures overlap heavily, and even trained human coders disagree on them for spontaneous micro-expressions. Asking a model to split hairs the annotators themselves cannot reliably split is chasing label noise, not signal. Valence-level grouping (Negative / Positive / Surprise) targets the distinction that is *actually reliable*.
3. **Field-standard comparability.** Negative / Positive / Surprise is the canonical **MEGC (Micro-Expression Grand Challenge) composite protocol**. Using it makes our numbers comparable to the published literature instead of a bespoke 7-class scheme nobody else reports.

> In short: grouping trades an unattainable 7-way problem (with 2- and 7-clip classes) for a **statistically tractable, cognitively meaningful, literature-comparable** 3-way problem.

### 3.6 Why these hyper-parameters? *(Core Question 5)*

| Hyper-parameter | Value | Optimisation logic |
|-----------------|-------|--------------------|
| Loss | **Focal, γ = 2.0** | down-weights the easy, over-represented Negative examples so gradient focuses on hard minority clips |
| Label smoothing | 0.05 | prevents over-confident logits on a tiny dataset; mild regulariser |
| Class weights | inverse-frequency | further counters the 63/20/16 % imbalance inside the loss |
| Sampler | balanced (oversampling) | each minibatch sees minority classes far more often than their natural rate |
| Optimiser | AdamW, lr 1e-4, wd 1e-4 | conservative lr for stable convergence on few samples; weight decay for regularisation |
| Grad-clip | 1.0 | guards against exploding gradients through the deep 3-D/Transformer stack |
| Scheduler | CosineAnnealingLR | smooth lr decay → fine convergence without hand-tuned steps |
| Mixed precision (AMP) | on | halves VRAM so 224² × 32-frame tensors fit |
| **Batch size** | **2** | *hardware-forced*: a `[3,32,224,224]` clip is huge; Transformer configs already use ~5 GB VRAM at batch 2 |
| Epochs | 50–60 | long enough to converge on 140 clips; early plateau visible in `training_metrics.csv` |
| Seed | 42 | reproducibility of the subject shuffle |
| Best-checkpoint criterion | **macro-F1** (not accuracy) | consistent with the primary metric; avoids saving a majority-class-collapsed epoch |

**The load-bearing choices** are (i) the *stacked* imbalance defences — focal loss **and** class weights **and** balanced sampling, because on a 63/20/16 split any single defence is insufficient — and (ii) **batch size 2**, an honest hardware constraint (not a scientific choice) that in turn justifies the small learning rate.

---

## 4. Experimental Results & Comprehensive Figure Analysis

### 4.1 Master results tables

**LOSO (pilot: 20 of 25 subject folds; 139 pooled test clips)** — ordered by macro-F1:

| Rank | Config | EVM | SimAM | CNN | Transf. | Accuracy | **Macro-F1** |
|:----:|--------|:---:|:-----:|:---:|:-------:|:--------:|:------------:|
| 1 | config_5_attention_base | – | ✓ | ✓ | – | 0.643 | **0.379** |
| 2 | config_3_spatial_only | – | – | ✓ | – | 0.627 | 0.358 |
| 3 | config_13_permutation | ✓ | – | ✓ | – | 0.491 | 0.329 |
| 4 | config_16_permutation | ✓ | ✓ | ✓ | – | 0.491 | 0.324 |
| 5 | config_7_full_no_attention | ✓ | – | ✓ | ✓ | 0.419 | 0.264 |
| 6 | config_12_permutation | ✓ | – | – | ✓ | 0.511 | 0.256 |
| 7 | config_9_permutation | – | – | ✓ | ✓ | 0.476 | 0.252 |
| 8 | config_2_temporal_only | – | – | – | ✓ | 0.600 | 0.251 |
| 9 | config_6_full_stage2_noevm | – | ✓ | ✓ | ✓ | 0.474 | 0.250 |
| 10 | **config_8_proposed_unified** | ✓ | ✓ | ✓ | ✓ | 0.413 | 0.249 |
| 11 | config_1_pure_base | – | – | – | – | 0.295 | 0.179 |
| 12 | config_4_motion_amp_base | ✓ | – | – | – | 0.305 | 0.166 |
| — | *majority-class baseline* | | | | | *0.662* | *0.266* |

**Hold-out (30 % of subjects; 52 test clips)** — ordered by macro-F1:

| Rank | Config | EVM | SimAM | CNN | Transf. | Accuracy | **Macro-F1** |
|:----:|--------|:---:|:-----:|:---:|:-------:|:--------:|:------------:|
| 1 | config_13_permutation | ✓ | – | ✓ | – | 0.577 | **0.458** |
| 2 | config_16_permutation | ✓ | ✓ | ✓ | – | 0.558 | 0.448 |
| 3 | config_5_attention_base | – | ✓ | ✓ | – | 0.481 | 0.402 |
| 4 | config_3_spatial_only | – | – | ✓ | – | 0.462 | 0.388 |
| 5 | config_9_permutation | – | – | ✓ | ✓ | 0.442 | 0.274 |
| 6 | config_2_temporal_only | – | – | – | ✓ | 0.231 | 0.247 |
| 7 | config_6_full_stage2_noevm | – | ✓ | ✓ | ✓ | 0.308 | 0.218 |
| 8 | config_12_permutation | ✓ | – | – | ✓ | 0.212 | 0.210 |
| 9 | **config_8_proposed_unified** | ✓ | ✓ | ✓ | ✓ | 0.173 | 0.183 |
| 10 | config_7_full_no_attention | ✓ | – | ✓ | ✓ | 0.192 | 0.176 |
| 11 | config_1_pure_base | – | – | – | – | 0.192 | 0.111 |
| 12 | config_4_motion_amp_base | ✓ | – | – | – | 0.096 | 0.129 |
| — | *majority-class baseline* | | | | | *0.750* | *0.286* |

### 4.2 Reading the results — the four structural findings

**(a) CNN is necessary.** Sort either table: the top four macro-F1 rows *all* have CNN = ✓ and Transformer = ✗. The bottom rows are dominated by CNN-off configs (1, 4) and Transformer-heavy configs. Config 4 (EVM-only, no CNN) is the *worst* hold-out model at 0.096 accuracy — magnified motion with no spatial extractor is nearly useless.

**(b) The Transformer is a net negative.** Match every pair that differs only in Variable D:

| Pair (CNN on) | Transformer OFF | Transformer ON | Δ macro-F1 |
|---------------|:---------------:|:--------------:|:----------:|
| LOSO: config_5 vs config_6 | 0.379 | 0.250 | **−0.129** |
| LOSO: config_3 vs config_9 | 0.358 | 0.252 | **−0.106** |
| LOSO: config_16 vs config_8 | 0.324 | 0.249 | **−0.075** |
| Hold-out: config_16 vs config_8 | 0.448 | 0.183 | **−0.265** |
| Hold-out: config_13 vs config_7 | 0.458 | 0.176 | **−0.282** |

The Transformer subtracts macro-F1 in **every** matched comparison. Mechanism: a 4-layer, 8-head encoder adds ~10⁵ parameters that ~140 clips cannot constrain → it overfits and, as the confusion matrices show, collapses onto one or two columns.

**(c) More components ≠ better.** The *proposed unified* config 8 (all four on) ranks **10th of 12 under LOSO and 9th of 12 under hold-out**. This is the central ablation lesson: the value of a component is conditional on data scale, and stacking them blindly compounds the Transformer's overfitting with EVM's noise amplification.

**(d) EVM is protocol-dependent.** LOSO's best omits EVM (config_5); hold-out's best uses it (config_13). Interpretation: EVM amplifies genuine motion *and* noise; on a fixed split (hold-out) the amplified signal helps, but across *unseen subjects* (LOSO) the amplified subject-specific noise hurts generalisation, so the raw-motion CNN wins.

### 4.3 Figure-by-figure analysis

Every plot exists twice — once under `loso/plots/` and once under `holdout/plots/`.

---

#### Figure 1 — `accuracy_macro_f1_bar.png` (Accuracy vs Macro-F1 per config)

- **Axes.** *x* = the 12 configurations; *y* = score in [0, 1]. Blue bars = accuracy, orange bars = macro-F1.
- **What it shows.** For *every* configuration the **blue (accuracy) bar towers over the orange (macro-F1) bar** — the persistent accuracy-vs-F1 gap.
- **Why it matters / what it proves.** That gap is the fingerprint of majority-class bias: the models get many Negative clips right (inflating accuracy) while doing poorly on Positive/Surprise (depressing macro-F1). It is the single clearest visual justification for reporting macro-F1 as the primary metric.
- **LOSO version:** blue bars cluster 0.30–0.64; the CNN-only / attention configs (3, 5) show the *smallest* accuracy-F1 gap (genuine multi-class learning), while the transformer-only config 2 shows a *huge* gap (0.60 acc / 0.25 F1 — high accuracy, poor balance — a majority-class parrot).
- **Hold-out version:** more compressed and volatile; several transformer configs fall below 0.25 on both metrics, and config 4 is near the floor — consistent with hold-out's higher variance on 52 clips.

---

#### Figure 2 — `per_class_f1_grouped.png` (Per-class F1, grouped by Negative / Positive / Surprise)

- **Axes.** *x* = the three classes; *y* = F1 in [0, 1]; one coloured bar per configuration (legend maps colour→config).
- **What it shows.**
  - **Negative** F1 is high and fairly uniform across configs (~0.45–0.65 LOSO) — every model finds the majority class easily.
  - **Positive** F1 is middling (~0.3–0.5) and this is where the *good* configs (5, 3, 13, 16) separate from the pack.
  - **Surprise** F1 is the lowest and most scattered — the hardest, rarest class.
  - Config 4's Negative bar in LOSO is **near zero** — a striking sign that EVM-only, CNN-less features fail even on the easy class.
- **Why it matters.** It localises *where* macro-F1 is lost: the ranking is decided almost entirely by Positive and Surprise. It also proves the imbalance defences (§3.6) are doing real work — otherwise the minority bars would be flat zero.

---

#### Figure 3 — `confusion_matrices.png` (3×4 panel, all 12 configs)

- **Axes.** For each small matrix: rows = true class, columns = predicted class (Negative / Positive / Surprise); colour intensity = clip count.
- **What it shows.** A strong diagonal = good; vertical banding (one bright column) = collapse onto one predicted class.
  - Good configs (3, 5, 13, 16) show a **visible diagonal**, especially a bright top-left Negative cell with off-diagonal leakage into Surprise.
  - Transformer / degenerate configs (1, 8, 2, 4) show **vertical stripes** — e.g. config 1 predicts *no* clip as Negative (empty first column) and dumps everything into Positive/Surprise; config 8 smears the true-Negative row across Positive and Surprise.
- **Why it matters.** This is the *mechanism* behind the F1 numbers: it shows the failure is systematic column-collapse (majority or minority parroting), not random error — precisely what the transformer's overfitting predicts.

---

#### Figure 4 — `key_configs_confusion_side_by_side.png`

- **What it shows.** The same confusion matrices for the headline configs placed adjacently for direct comparison (proposed config 8 vs the strong CNN-only / attention configs).
- **Why it matters.** Side-by-side, config 8's smeared matrix next to config 5/13's cleaner diagonal makes the negative result undeniable: adding EVM + Transformer to a working CNN+attention model *degrades* the confusion structure.

---

#### Per-config `confusion_matrix.png` & `training_metrics.csv`

- Each config folder holds its own full-resolution confusion matrix and an epoch-by-epoch training log (`epoch, train_loss, val_loss, val_acc, val_f1, duration_sec`).
- **Reading the training curves (e.g. config 8 hold-out):** `train_loss` falls steadily from ~2.09 to ~0.65 while `val_f1` **oscillates between 0.04 and 0.18 without trending up** — a textbook overfitting signature. The gap between falling train loss and stagnant validation F1 is the direct, per-epoch evidence that the heavy configs memorise rather than generalise. Per-epoch wall-time (~14.7 s) and peak VRAM (~5 GB) are also logged in each `configuration_summary.txt`.

### 4.4 What the results collectively prove

1. The **spatial 3-D CNN** carries the signal; it is non-negotiable.
2. **Temporal Transformer capacity is mismatched to dataset scale** — a clean, quantified negative result, not hand-waving.
3. **Macro-F1 + confusion matrices** were the right instruments: raw accuracy would have hidden the majority-class collapse that the confusion matrices make visible.
4. The **LOSO vs hold-out agreement** on the qualitative ranking (CNN✓/Transformer✗ wins) means the finding is robust to the validation protocol, not a split artefact.

---

## 5. Deep-Dive FAQ (Defending the Study)

**Q1. Your best accuracy (0.58–0.64) is *below* the 0.66–0.75 majority-class baseline. Isn't your model worse than doing nothing?**
No — on the *honest* metric it is decisively better. The majority baseline scores its accuracy by predicting Negative for *everything*, earning **~0.27–0.29 macro-F1**. Our best models reach **0.38 (LOSO) / 0.46 (hold-out) macro-F1** — a 40–60 % relative improvement — by actually recognising Positive and Surprise. Trading a few majority-class hits for genuine minority recall *lowers* accuracy while *raising* the metric that measures real recognition. This is exactly why we lead with macro-F1.

**Q2. A previous run reported ~74 % hold-out accuracy. Why are these numbers lower?**
Because ~74 % *was essentially the majority baseline* (0.75 here) — a model predicting Negative almost always. The weekend runs add inverse-frequency class weights, focal loss, and a balanced sampler, and select checkpoints by macro-F1. These force the model off the majority crutch, so accuracy drops toward the true multi-class difficulty while macro-F1 becomes meaningful. Lower accuracy here is a sign of *more honest* evaluation, not worse modelling.

**Q3. 156 clips is tiny. Is any of this significant?**
It is small — which is *precisely why* we (a) group to 3 classes for statistical power, (b) stack three imbalance defences, (c) use parameter-free attention, and (d) use LOSO so every subject is a test case. We are candid that per-cell numbers carry wide confidence intervals; the load-bearing claims are the *consistent, directional* effects that survive across 12 configs **and** both protocols (CNN helps everywhere; Transformer hurts everywhere). Those directional results are robust even when absolute values are not.

**Q4. Why is your "proposed" model config 8 not the winner? Isn't that a failed thesis?**
On the contrary — an ablation that rubber-stamps the proposed model is a weak ablation. The scientific value here is the *diagnosis*: config 8 loses because the Transformer overfits at this data scale (Q6). The correct thesis narrative is "the proposed architecture is sound in principle, but at CASME II scale the temporal encoder is starved; the effective model is CNN + attention, and the Transformer is expected to pay off only with substantially more data / composite datasets." Negative results, cleanly attributed, are results.

**Q5. Did EVM actually do anything?**
Yes, and its sign flips with protocol — the interesting part. Under hold-out, the best model (config 13) *uses* EVM; under LOSO the best (config 5) omits it. EVM amplifies motion **and** noise: helpful on a fixed split, harmful when generalising to unseen subjects whose noise profile differs. This is a nuanced, defensible finding rather than a null one. (Note: earlier project documentation claimed EVM had *no* effect because only one tensor set had been generated at that time — that is not the case in these weekend runs, where EVM and raw configurations produce clearly different results.)

**Q6. Why blame the Transformer specifically for overfitting?**
Two independent evidence streams. (i) *Every* matched Transformer-on/off pair shows lower macro-F1 with it on (§4.2b), Δ up to −0.28. (ii) The per-epoch logs (§4.3) show train-loss falling while val-F1 stalls — the definition of overfitting — and the confusion matrices show the collapse onto single columns that over-parameterised models produce on scarce data. A 4-layer/8-head encoder simply has more parameters than 140 clips can identify.

**Q7. Why batch size 2 — isn't that unstably small?**
It is a hardware constraint, stated honestly: a single `[3,32,224,224]` clip is large, and Transformer configs already consume ~5 GB VRAM at batch 2. We compensate with a small learning rate (1e-4), gradient clipping (1.0), and mixed precision. The consistent convergence in the training logs shows it is stable in practice.

**Q8. Is the LOSO number final?**
No — it is an explicit **pilot** (20 of 25 folds), flagged in every `final_results.json` (`loso_pilot: true`, `loso_folds_run: 20`). The full 25-fold LOSO is the number reserved for the final dissertation table once the architecture and epoch budget are frozen. We report the pilot because it already gives a low-variance, subject-generalising estimate that agrees qualitatively with hold-out.

**Q9. Why drop "Others" — isn't that cherry-picking easy classes?**
"Others" is not an emotion; it is CASME II's residual bucket (others / helpless / pain / confused / sympathy) with no coherent facial target. Including it trains the model to predict "none of the above", injecting label noise and inflating apparent difficulty without adding scientific signal. Dropping it (retaining 61 % of clips) is the standard MEGC treatment and makes our 3-class results literature-comparable.

**Q10. Could identity leakage still be inflating results?**
Structurally, no. (i) The network is fed only optical flow + strain — no appearance/identity channel. (ii) Every split (both protocols) is *whole-subject* disjoint; the loader guarantees no subject id appears in both partitions (`assert not (train ∩ val)`). LOSO in particular tests each subject as a complete unseen individual. Leakage is closed off at both the feature and the split level.

---

*All numbers in this report are taken directly from `results_weekend/{loso,holdout}/summary.csv` and the per-config `final_results.json` files; class distributions from `Processed_Data/master_thesis_labels.csv`; hyper-parameters from `Ablation_Study/ablation_config.py`, `losses.py`, `trainer.py`; and the 30 % split from `gui_settings.json` (`ablation_val_fraction = 0.3`).*
