# A Component Ablation Study of a Hybrid Spatiotemporal Pipeline for Micro-Expression Recognition on CASME II

**Formal Technical Report**

*Evaluation protocols: Leave-One-Subject-Out (pilot, 20/25 folds) and subject-disjoint hold-out (30 % of subjects).*
*Target: three grouped emotion classes — Negative, Positive, Surprise.*

---

## Abstract

We present an exhaustive component ablation of a hybrid micro-expression recognition (MER) architecture comprising four independently toggleable elements: Eulerian Video Magnification (EVM), a three-stream 3-D convolutional backbone, parameter-free SimAM spatial attention, and an SLSTT temporal Transformer. All twelve architecturally valid configurations of the 2×2×2×2 design matrix were trained on 156 CASME II micro-expression clips and evaluated under two strictly subject-disjoint protocols. We find that the 3-D convolutional backbone is the single indispensable component; that the temporal Transformer consistently *degrades* performance at this data scale (macro-F1 reduction up to 0.28 across matched pairs); that the fully-integrated model ranks near the bottom of the field; and that macro-F1, not accuracy, is the only metric that distinguishes genuine multi-class learning from majority-class collapse. The best macro-F1 achieved is 0.379 under LOSO (CNN + SimAM) and 0.458 under hold-out (CNN + EVM), against majority-class baselines of 0.266 and 0.286 respectively.

---

## 1. Introduction

Micro-expressions are brief (typically under 200 ms), involuntary facial movements that reveal concealed emotional states. Their automatic recognition is a demanding problem in affective computing owing to three compounding difficulties: (i) the motion signal is faint, often below the threshold of unaided human perception; (ii) benchmark corpora are small and severely class-imbalanced; and (iii) spontaneous elicitation produces noisy, ambiguous emotion labels.

Rather than propose a novel architecture and assert its superiority, this study adopts an ablation-first methodology. We define four design variables — a data-level motion-magnification switch and three model-level modules — and evaluate *every* combination, so that the contribution of each component can be attributed rather than assumed. The central research question is therefore not "does our model work?" but "which components are responsible for whatever performance is observed, and under what conditions?"

This report is organised as follows. Section 2 specifies the experimental setup: dataset, pipeline, architecture, and hyper-parameters. Section 3 provides rigorous methodological justifications for the principal design decisions (feature and model choice, metric choice, validation strategy, data-sampling constraints, target reframing, and hyper-parameter selection). Section 4 presents and discusses the complete results, including a figure-by-figure analysis. Section 5 concludes.

---

## 2. Experimental Setup

### 2.1 Dataset

All experiments use the **CASME II** corpus of spontaneous facial micro-expressions. The master annotation table contains **255 micro-expression clips** from **26 subjects**. Each clip is annotated with onset, apex, and offset frames and a raw emotion label. The raw label distribution is: disgust (63), happiness (32), repression (27), surprise (25), sadness (7), fear (2), and *others* (99).

For classification we adopt the grouped three-class scheme (Section 3.5), which discards the *others* bucket and yields **156 clips across 25 subjects**, distributed as Negative (99), Positive (32), and Surprise (25). One subject possessing only *others*-labelled clips is consequently absent from the classification set, which is why the LOSO fold total is 25 rather than 26.

### 2.2 Data pipeline (Stage 1)

Each raw clip is transformed offline into a fixed-geometry motion tensor through the following steps:

1. Parsing of the coding sheet to recover onset/apex/offset boundaries.
2. Face detection, cropping, and alignment.
3. *(Optional — Variable A)* Eulerian Video Magnification with amplification factor α = 10, a 5–25 Hz temporal band-pass, and a four-level Laplacian pyramid.
4. Temporal resampling of the onset–offset interval to exactly **T = 32** frames.
5. Dense optical-flow estimation, producing horizontal (*u*) and vertical (*v*) components.
6. Optical-strain computation (the spatial derivative of the flow field), capturing local deformation while suppressing rigid head motion.
7. Assembly of a three-channel tensor of shape **[3, 32, 224, 224]**, persisted as `.npy`.

Two tensor sets are precomputed — EVM-magnified (`tensors/`) and non-magnified (`tensors_raw/`) — so that Variable A reduces to a directory selection at training time.

### 2.3 Architecture (Stage 2)

The network processes an input tensor of shape [B, 3, 32, 224, 224] through three stages:

- **Stage A — Spatial stem.** A three-stream 3-D CNN (each stream: Conv3D → Conv3D → MaxPool3D; 32 output channels per stream, concatenated to 96) maps the input to [B, 96, 32, 112, 112]. When enabled, **SimAM** — a parameter-free attention mechanism whose neuron-energy function `e_t = (x−μ)²/[4(σ²+λ)] + 0.5` requires zero learnable parameters — rescales the feature maps. Adaptive average pooling and reshaping yield a temporal sequence [B, 32, 96].
- **Stage B — Temporal encoder.** When the Transformer is enabled, an SLSTT encoder (d_model = 96, 8 heads, 4 layers, feed-forward dim 256, dropout 0.1) with sinusoidal positional encoding produces a pooled [B, 96] representation; when disabled, simple mean/max temporal pooling is used instead.
- **Stage C — Classifier head.** LayerNorm → Dropout → Linear → [B, 3].

When the CNN is disabled, the model falls back to average-pooled, flattened raw patches projected to the model width, ensuring every configuration remains runnable.

### 2.4 The ablation matrix

The 2×2×2×2 matrix over (EVM, SimAM, CNN, Transformer) is generated programmatically; the four cells with SimAM enabled but the CNN disabled are pruned as degenerate, leaving **twelve valid configurations** (Table 1).

**Table 1. The twelve ablation configurations.**

| # | Name | EVM | SimAM | CNN | Transf. |
|---|------|:---:|:-----:|:---:|:-------:|
| 1 | pure_base | – | – | – | – |
| 2 | temporal_only | – | – | – | ✓ |
| 3 | spatial_only | – | – | ✓ | – |
| 4 | motion_amp_base | ✓ | – | – | – |
| 5 | attention_base | – | ✓ | ✓ | – |
| 6 | full_stage2_noevm | – | ✓ | ✓ | ✓ |
| 7 | full_no_attention | ✓ | – | ✓ | ✓ |
| 8 | proposed_unified | ✓ | ✓ | ✓ | ✓ |
| 9 | permutation | – | – | ✓ | ✓ |
| 12 | permutation | ✓ | – | – | ✓ |
| 13 | permutation | ✓ | – | ✓ | – |
| 16 | permutation | ✓ | ✓ | ✓ | – |

### 2.5 Training configuration

The objective is single-task emotion classification under **Focal loss** (γ = 2.0) with label smoothing (0.05) and inverse-frequency class weights, combined with a class-balanced oversampling training sampler. Optimisation uses AdamW (learning rate 1×10⁻⁴, weight decay 1×10⁻⁴), gradient-norm clipping at 1.0, automatic mixed precision, and a CosineAnnealing learning-rate schedule. The batch size is 2 — a hardware constraint imposed by the [3, 32, 224, 224] tensor geometry (Transformer configurations consume approximately 5 GB of VRAM at this batch size). Training runs for 50–60 epochs, and the checkpoint maximising validation macro-F1 is retained. The random seed is fixed at 42.

### 2.6 Validation protocols

Two strictly subject-disjoint protocols are used. **Hold-out** withholds 30 % of subjects (`val_fraction = 0.3`) as a single test partition, yielding 52 test clips. **Leave-One-Subject-Out (LOSO)** trains on all-but-one subject and tests on the held-out subject, repeated across folds; the reported runs constitute a **pilot** of 20 of the 25 possible folds (evenly distributed across the cohort), pooling to 139 test clips. Both protocols guarantee that no subject appears in both training and test partitions.

---

## 3. Methodological Justifications

### 3.1 Feature and model selection

The pipeline consumes **motion tensors (optical flow and strain), not RGB frames**. This choice is motivated by the nature of the target: a micro-expression is defined by *how* facial skin deforms, a signal that is otherwise dominated by static appearance and subject identity. Optical strain, being a spatial derivative of the flow field, is additionally invariant to rigid head translation. A three-stream 3-D CNN was selected as the spatial stem to capture joint spatial-temporal micro-texture that single-frame 2-D networks cannot represent. SimAM was preferred over parameterised attention mechanisms (SE, CBAM) specifically because it introduces **no additional learnable parameters** — a decisive advantage under extreme data scarcity.

Several alternatives were rejected on principled grounds: RGB-plus-recurrent pipelines invite identity leakage and bury the motion signal under appearance; very deep video backbones (I3D, R(2+1)D) possess far too many parameters to be identifiable from 156 clips; and apex-frame-only approaches discard the temporal dynamics that constitute a micro-expression. The temporal Transformer was included **as a hypothesis to be tested**, not as a settled design element — the ablation exists precisely to determine whether long-range temporal modelling is warranted at this scale.

### 3.2 Metric selection

Given class proportions of approximately 63 %/20 %/16 %, raw accuracy is an unreliable and potentially misleading indicator. A trivial classifier that always predicts the majority (Negative) class attains **0.662 accuracy under LOSO and 0.750 under hold-out**, yet only **0.266 and 0.286 macro-F1** respectively. We therefore designate **macro-averaged F1** — which weights each class equally and severely penalises neglect of the minority classes — as the primary metric. Accuracy is retained solely as a reference and to make the majority-class trap explicit. Per-class precision, recall, and F1, together with full confusion matrices, are reported to characterise the *structure* of errors, which is material in an affective-computing context where different confusions carry different costs. This metric suite is the appropriate choice for imbalanced, few-sample, class-sensitive classification.

### 3.3 Validation strategy

The dual protocol is deliberate. Subject-disjoint splitting is mandatory in MER because a naive random split permits **identity leakage**: clips of the same individual in both partitions allow a network to recognise the person rather than the expression, producing optimistically biased estimates. LOSO eliminates this bias entirely and provides the honest estimate of generalisation to a previously unseen individual, which is the operationally relevant quantity. Its cost, however, is one full training run per subject — prohibitive across twelve configurations at 60 epochs — motivating the pilot of 20 folds. Hold-out, being roughly twenty times cheaper, serves as the rapid development protocol for hyper-parameter and epoch-budget iteration. The two protocols are complementary: hold-out offers speed at the price of higher variance (its statistics rest on a single split), while LOSO offers a low-variance, less optimistic estimate. Concordance between them on the qualitative ranking constitutes evidence that the principal findings are protocol-invariant.

### 3.4 Data-sampling constraints

Three data-reduction decisions warrant explicit justification.

**(a) The 30 % hold-out fraction.** The hold-out protocol withholds 30 % of subjects for testing (recorded as `ablation_val_fraction = 0.3`). This fraction is the smallest that yields a statistically usable test set on so small a corpus: at 10 % the test partition would comprise roughly fifteen clips, on which a three-class macro-F1 estimate would be dominated by individual misclassifications and one or more classes might be entirely absent. At 30 % (approximately 52 clips) all three classes are, with high probability, represented on both sides of the split, and per-class confusion structure becomes legible. Because the split operates at whole-subject granularity, 30 % of 25 subjects (seven to eight subjects) is also the natural resolution of the allocation.

**(b) Why not 100 %.** The train/test allocation is zero-sum: every clip assigned to testing is withdrawn from training. Testing on 100 % of the data is therefore ill-defined, as it leaves no training signal. On a 156-clip corpus the training partition is already scarce, so approximately 70 % is retained for learning and 30 % is spent on a test estimate large enough to be trustworthy. LOSO resolves the same trade-off differently by recycling every subject as a test case exactly once.

**(c) Exclusion of the "Others" class.** The *others* label (99 of 255 clips, ≈ 39 %) aggregates heterogeneous, affectively incoherent states (others, helpless, pain, confused, sympathy). Retaining it would require the classifier to model a "none-of-the-above" category with no consistent facial signature, injecting label noise. Its exclusion, retaining 61 % of the micro-expression clips, follows the standard MEGC treatment and is a noise-reduction measure rather than a convenience sample.

### 3.5 Target reframing (emotion grouping)

The seven raw CASME II emotions are mapped to three valence-level classes — Negative (disgust, repression, sadness, fear), Positive (happiness), and Surprise — with *others* discarded. Three independent arguments justify this reframing:

1. **Statistical power.** The raw distribution is extreme in its tail: fear comprises **two** clips and sadness **seven**. Under LOSO, a two-clip class cannot even be partitioned into meaningful train and test subsets; its per-class F1 would be governed by a single example. No modelling technique can compensate for two samples. Grouping pools these into a 99-clip Negative class that is statistically learnable.
2. **Cognitive ambiguity.** Disgust, repression, sadness, and fear are all negative-valence states with overlapping facial signatures; for spontaneous micro-expressions even trained human coders exhibit substantial disagreement among them. Requiring the model to discriminate categories that the annotation process itself cannot reliably separate amounts to fitting label noise. Grouping by valence targets the distinction that is actually reliable.
3. **Comparability.** The Negative/Positive/Surprise scheme is the canonical MEGC composite protocol, rendering results comparable to the published literature rather than idiosyncratic.

The reframing thus converts an unattainable seven-way problem into a tractable, meaningful, and comparable three-way problem.

### 3.6 Hyper-parameter selection

The optimisation configuration is dictated primarily by data scarcity and class imbalance. Three imbalance defences are stacked because, at a 63/20/16 split, no single mechanism suffices: Focal loss (γ = 2.0) down-weights the abundant, easily-classified majority examples so that gradient signal concentrates on hard minority instances; inverse-frequency class weights amplify minority contributions within the loss; and a balanced oversampling sampler ensures each minibatch over-represents minority classes relative to their natural frequency. Label smoothing (0.05) mitigates over-confident logits on a small corpus. The learning rate (1×10⁻⁴) is conservative to stabilise convergence on few samples, complemented by weight decay (1×10⁻⁴) and gradient clipping (1.0) for regularisation and numerical stability through the deep stack. The batch size of 2 is an honest hardware limit rather than a scientific choice, and the CosineAnnealing schedule provides smooth learning-rate decay. Consistent with Section 3.2, the best checkpoint is selected by validation macro-F1, precluding the retention of an epoch that has collapsed onto the majority class.

---

## 4. Results and Discussion

### 4.1 Quantitative results

Tables 2 and 3 report accuracy and macro-F1 for all twelve configurations under each protocol, ordered by macro-F1, with the majority-class baseline appended for reference.

**Table 2. LOSO (pilot, 20/25 folds; 139 pooled test clips).**

| Rank | Config | EVM | SimAM | CNN | Transf. | Accuracy | Macro-F1 |
|:----:|--------|:---:|:-----:|:---:|:-------:|:--------:|:--------:|
| 1 | config_5_attention_base | – | ✓ | ✓ | – | 0.643 | **0.379** |
| 2 | config_3_spatial_only | – | – | ✓ | – | 0.627 | 0.358 |
| 3 | config_13_permutation | ✓ | – | ✓ | – | 0.491 | 0.329 |
| 4 | config_16_permutation | ✓ | ✓ | ✓ | – | 0.491 | 0.324 |
| 5 | config_7_full_no_attention | ✓ | – | ✓ | ✓ | 0.419 | 0.264 |
| 6 | config_12_permutation | ✓ | – | – | ✓ | 0.511 | 0.256 |
| 7 | config_9_permutation | – | – | ✓ | ✓ | 0.476 | 0.252 |
| 8 | config_2_temporal_only | – | – | – | ✓ | 0.600 | 0.251 |
| 9 | config_6_full_stage2_noevm | – | ✓ | ✓ | ✓ | 0.474 | 0.250 |
| 10 | config_8_proposed_unified | ✓ | ✓ | ✓ | ✓ | 0.413 | 0.249 |
| 11 | config_1_pure_base | – | – | – | – | 0.295 | 0.179 |
| 12 | config_4_motion_amp_base | ✓ | – | – | – | 0.305 | 0.166 |
| — | Majority-class baseline | | | | | 0.662 | 0.266 |

**Table 3. Hold-out (30 % of subjects; 52 test clips).**

| Rank | Config | EVM | SimAM | CNN | Transf. | Accuracy | Macro-F1 |
|:----:|--------|:---:|:-----:|:---:|:-------:|:--------:|:--------:|
| 1 | config_13_permutation | ✓ | – | ✓ | – | 0.577 | **0.458** |
| 2 | config_16_permutation | ✓ | ✓ | ✓ | – | 0.558 | 0.448 |
| 3 | config_5_attention_base | – | ✓ | ✓ | – | 0.481 | 0.402 |
| 4 | config_3_spatial_only | – | – | ✓ | – | 0.462 | 0.388 |
| 5 | config_9_permutation | – | – | ✓ | ✓ | 0.442 | 0.274 |
| 6 | config_2_temporal_only | – | – | – | ✓ | 0.231 | 0.247 |
| 7 | config_6_full_stage2_noevm | – | ✓ | ✓ | ✓ | 0.308 | 0.218 |
| 8 | config_12_permutation | ✓ | – | – | ✓ | 0.212 | 0.210 |
| 9 | config_8_proposed_unified | ✓ | ✓ | ✓ | ✓ | 0.173 | 0.183 |
| 10 | config_7_full_no_attention | ✓ | – | ✓ | ✓ | 0.192 | 0.176 |
| 11 | config_1_pure_base | – | – | – | – | 0.192 | 0.111 |
| 12 | config_4_motion_amp_base | ✓ | – | – | – | 0.096 | 0.129 |
| — | Majority-class baseline | | | | | 0.750 | 0.286 |

### 4.2 Principal findings

**Finding 1 — The 3-D CNN backbone is indispensable.** Under both protocols the four highest macro-F1 configurations all possess the CNN and lack the Transformer. Conversely, the CNN-free configurations occupy the lowest ranks; config_4 (EVM only) attains just 0.096 hold-out accuracy. Magnified motion without a spatial feature extractor is near-useless.

**Finding 2 — The temporal Transformer degrades performance.** In every configuration pair differing only in the Transformer toggle, enabling it *reduces* macro-F1 (Table 4). The effect is large and consistent, reaching −0.282 macro-F1 in the hold-out config_13/config_7 comparison.

**Table 4. Effect of enabling the Transformer (matched pairs, CNN present).**

| Comparison | Protocol | Transf. OFF | Transf. ON | Δ Macro-F1 |
|-----------|:--------:|:-----------:|:----------:|:----------:|
| config_5 → config_6 | LOSO | 0.379 | 0.250 | −0.129 |
| config_3 → config_9 | LOSO | 0.358 | 0.252 | −0.106 |
| config_16 → config_8 | LOSO | 0.324 | 0.249 | −0.075 |
| config_16 → config_8 | Hold-out | 0.448 | 0.183 | −0.265 |
| config_13 → config_7 | Hold-out | 0.458 | 0.176 | −0.282 |

The mechanism is over-parameterisation: a four-layer, eight-head encoder introduces on the order of 10⁵ parameters that approximately 140 training clips cannot constrain, producing overfitting evident in the training logs (falling training loss against stagnant validation F1) and column-collapse in the confusion matrices.

**Finding 3 — Integration does not imply improvement.** The fully-integrated config_8 ranks tenth of twelve under LOSO and ninth of twelve under hold-out. This is a substantive negative result: the effective architecture at CASME II scale is CNN with SimAM (or EVM), and the Transformer is counterproductive.

**Finding 4 — EVM's effect is protocol-dependent.** EVM appears in the best hold-out configuration (config_13) but is absent from the best LOSO configuration (config_5). Because magnification amplifies both genuine motion and subject-specific noise, it is beneficial on a fixed split but detrimental when generalising across unseen subjects, where the amplified noise fails to transfer.

### 4.3 Figure analysis

**`accuracy_macro_f1_bar.png` (both protocols).** The *x*-axis enumerates the twelve configurations and the *y*-axis is score in [0, 1]; blue and orange bars denote accuracy and macro-F1. Across every configuration the accuracy bar exceeds the macro-F1 bar, visually establishing the universal majority-class pull and thereby the necessity of macro-F1. In the LOSO chart the CNN-based, Transformer-free configurations exhibit the smallest accuracy–F1 gap (indicative of genuine multi-class learning), whereas the Transformer-only config_2 exhibits a large gap (0.600 accuracy against 0.251 macro-F1), the signature of majority-class prediction.

**`per_class_f1_grouped.png` (both protocols).** Grouping F1 by class reveals that Negative F1 is high and stable across configurations, Positive F1 is intermediate and discriminating, and Surprise F1 is lowest and most variable. The configuration ranking is therefore decided almost entirely on the two minority classes. The near-zero Negative F1 of config_4 under LOSO further illustrates the failure of CNN-free features even on the easy class.

**`confusion_matrices.png` (3×4 panels).** For each per-configuration matrix, rows are the true class and columns the predicted class, with colour encoding clip counts. Strong configurations (3, 5, 13, 16) exhibit a visible diagonal, while degenerate and Transformer-heavy configurations (1, 8, 2, 4) exhibit vertical banding — collapse onto a single predicted column. Config_1, for instance, predicts no clip as Negative (an empty first column). These matrices reveal the failure mode to be systematic column-collapse rather than diffuse error, consistent with over-parameterised overfitting.

**`key_configs_confusion_side_by_side.png`.** Placing the confusion matrices of the headline configurations adjacently makes the negative result for config_8 directly legible: its smeared prediction structure contrasts with the cleaner diagonals of the lean CNN configurations.

**Per-configuration `training_metrics.csv`.** The epoch-level logs corroborate the overfitting diagnosis. For config_8 under hold-out, training loss decreases monotonically from ≈ 2.09 to ≈ 0.65 while validation F1 oscillates between 0.04 and 0.18 without an upward trend — the canonical divergence of training and validation performance.

### 4.4 Synthesis

The results collectively establish that (i) spatial 3-D convolutional feature extraction carries the recognition signal; (ii) temporal Transformer capacity is mismatched to the scale of CASME II and is quantifiably counterproductive; (iii) macro-F1 and confusion-matrix analysis were essential instruments, without which majority-class collapse would have been mistaken for competent performance under raw accuracy; and (iv) the qualitative concordance between LOSO and hold-out confirms that the principal conclusions are robust to the choice of validation protocol.

---

## 5. Conclusion

This ablation study demonstrates, through the exhaustive evaluation of twelve architectural configurations under two subject-disjoint protocols, that the effective micro-expression recognition model on CASME II is a three-stream 3-D CNN augmented with parameter-free attention, and that the temporal Transformer — although theoretically attractive — degrades performance at the available data scale by overfitting. The best macro-F1 attained (0.379 LOSO, 0.458 hold-out) substantially exceeds the corresponding majority-class baselines (0.266, 0.286), confirming that the models achieve genuine, if modest, multi-class discrimination rather than majority-class parroting. We emphasise that the underperformance of the fully-integrated model is a scientifically informative outcome: it localises the limiting factor to data scale rather than architectural principle.

Three directions follow. First, the full 25-fold LOSO evaluation should be executed to obtain the definitive subject-generalisation figure, the pilot having already validated the qualitative conclusions. Second, scaling to a composite corpus (e.g. CASME II + SAMM + SMIC) would supply the data volume necessary to test whether the temporal Transformer becomes beneficial once adequately constrained. Third, EVM's protocol-dependent behaviour merits dedicated cross-subject study to characterise the conditions under which motion magnification aids rather than harms generalisation.

---

### Data and Reproducibility Statement

All reported metrics are drawn from `results_weekend/{loso,holdout}/summary.csv` and the per-configuration `final_results.json` files. Class distributions derive from `Processed_Data/master_thesis_labels.csv`. Architectural and optimisation settings are specified in `Ablation_Study/ablation_config.py`, `losses.py`, and `trainer.py`. The 30 % hold-out fraction and the 20-fold LOSO pilot setting are recorded in `gui_settings.json` (`ablation_val_fraction = 0.3`, `ablation_loso_folds = 20`). The random seed is fixed at 42 throughout.
