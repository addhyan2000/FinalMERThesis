# Chapter 4 — Methodology

This chapter specifies the study as it was carried out: the experimental design, the corpus and its preparation, the twelve configurations trained, the architecture and its ablatable components, the training procedure, the evaluation protocol, and the computational conditions under which the sweep ran.

Chapter 2 explains what each mechanism is and how it works, and Chapter 3 reviews the literature for each and identifies the gap. This chapter states what was done and with which values, so that the work can be reproduced; where a mechanism has already been explained it is cross-referenced rather than re-derived. Every parameter reported here is taken from the configuration and code that produced the results in Chapter 5.

---

## 4.1 Research Approach and Experimental Design

### 4.1.1 The form of the study

This chapter does not describe the proposal and evaluation of a single architecture. It describes a factorial ablation: four components of a micro-expression recognition pipeline are each represented by a boolean flag, every viable combination of flags is trained and scored under an identical protocol, and each component's contribution is read off as a marginal effect computed over pairs of runs that differ in exactly that one flag. No single "best" model is nominated as the object of study. The object of study is the set of runs itself, and the question the design is built to answer is not "how well does this pipeline perform" but "which of its stages are responsible for that performance, and by how much".

That form of study follows directly from the gap identified in Chapter 3 (§3.10.3). The literature reviewed there reports end-to-end pipelines — magnification followed by a hand-chosen network, or an attention module inserted into a hand-chosen backbone — and reports a single accuracy or F1 score for the resulting system. When a new component is added, it is typically added to a pipeline that already differs from its predecessor in other respects, so an improvement in the reported number cannot be attributed to the new component alone with any confidence: it might equally be a consequence of a different backbone, a different training schedule, or a different subset of the corpus. One reviewed study does toggle a single stage against an otherwise fixed pipeline: Li et al. [14] sweep the magnification factor with α = 1 as an explicit no-magnification arm (§3.2.3). But that stage is a preprocessing step, and no reviewed study isolates a *learned* component — a convolutional backbone, an attention module, a temporal encoder — against an otherwise identical pipeline, nor varies several components jointly so that their interactions can be read. Consequently, no existing result can say, for this task, whether motion magnification is doing the work usually credited to it, whether an attention mechanism contributes anything once a convolutional stem is present, or whether a temporal transformer earns its cost over a simpler alternative. The factorial design in this chapter is built specifically to produce that evidence: every comparison it supports is a matched-pair comparison, in which the only thing that changed between two trained models is the single component under discussion.

### 4.1.2 The four variables

Four independently toggled components define the design space. Each is implemented in the codebase as a boolean configuration flag, and each corresponds to a mechanism described in Chapter 2:

- `use_evm` — Eulerian video (motion) magnification applied as a pre-processing stage before any frame reaches the network, amplifying the subtle motion a micro-expression produces (§2.3).
- `use_simam` — SimAM, a parameter-free attention module inserted into the convolutional stream to re-weight spatial feature activations without adding trainable parameters (§2.6.3).
- `use_cnn` — the three-stream convolutional spatial stem that extracts per-frame spatial features from the motion tensor (§2.6.1).
- `use_transformer` — the temporal transformer encoder that models dependencies across the frame sequence after spatial features have been extracted (§2.6.4).

Each flag is a strict on/off switch: there is no intermediate setting, no alternative implementation substituted when a flag is off, and no interaction term outside the model itself. When `use_cnn` is false, the pipeline does not substitute a different spatial feature extractor; the design simply does not train that cell if the resulting configuration is invalid, as described below.

### 4.1.3 Twelve configurations, not sixteen

Four independent binary flags define a design space of $2^4 = 16$ cells. The implementation constructs this space directly, by taking the Cartesian product of the two flag states across all four variables (`itertools.product([False, True], repeat=4)`), so every one of the sixteen combinations is generated as a candidate configuration.

Not every generated cell is trained. Each candidate is passed through a validity check, `AblationConfig.is_valid()`, which rejects any configuration in which `use_simam=True` while `use_cnn=False`. The stated reason is architectural rather than empirical: SimAM re-weights activations in a convolutional feature map, and without a convolutional stem there is no such feature map for it to act on, so a configuration combining SimAM with no CNN is not a weaker version of the pipeline but a specification that does not correspond to a well-formed model. This rejection is applied per candidate at run time, after the full set of sixteen has been generated, rather than by excluding those cells from the design space in advance. Of the sixteen generated cells, exactly four have `use_simam=True` and `use_cnn=False`, and all four are rejected. The remaining twelve configurations are trained and evaluated.

### 4.1.4 Matched pairs and their counts

Because the retained twelve cells form a (near-)complete factorial design, the effect of each of the four flags can be estimated by matched-pair comparison: for a given flag, every pair of trained configurations that share identical settings on the other three flags and differ only in that one flag constitutes one matched comparison. Summing these pairs across the design gives a different count for each flag, because the four invalid cells removed in §4.1.3 would have supplied matched partners for some flags and not others.

A matched pair for a given flag exists wherever both settings of that flag are valid once the other three flags are fixed. Validity depends only on the joint setting of `use_simam` and `use_cnn` (§4.1.3), so toggling `use_evm` or `use_transformer` alone never changes whether a cell is valid: for every one of the six distinct combinations of the other three flags that is itself valid, both the `use_evm=False` and `use_evm=True` versions survive, and likewise for `use_transformer`. Both flags are therefore measured over **six** matched pairs. Toggling `use_simam` or `use_cnn`, by contrast, can cross the validity boundary. A `use_simam` pair exists only where `use_cnn=True`, since a `use_cnn=False` cell is valid only at `use_simam=False` and so has no `use_simam=True` partner; this leaves four valid combinations of the other three flags with `use_cnn=True`, and hence four matched pairs for SimAM. Symmetrically, a `use_cnn` pair exists only where `use_simam=False`, since only then is the `use_cnn=False` cell itself valid; this again leaves four matched pairs for the CNN stem. The asymmetry — six pairs for EVM and the transformer, four for SimAM and the CNN — is a direct arithmetic consequence of the exclusion rule in §4.1.3, not an oversight in counting: SimAM and the CNN stem are the two flags entangled by that rule, and each loses exactly the pairs that would have crossed it.

### 4.1.5 A stated limitation

Every one of the twelve configurations is trained exactly once, using a single fixed random seed (42). No configuration is retrained under multiple seeds, and no variance across seeds is estimated for any of the twelve runs. Consequently, every difference reported between configurations in this thesis — including the matched-pair effects described in §4.1.4 — is a difference between two single point estimates, and carries no confidence interval or significance test. A difference that appears in Chapter 5 could, in principle, narrow, widen, or reverse under a different seed; this design cannot distinguish a robust effect of a component from noise attributable to initialisation or batch ordering alone. This limitation is stated here, at the point where the experimental design is fixed, rather than deferred to the discussion of results, because it is a property of the design itself and not of any particular outcome it produces.

---

## 4.2 The Corpus and Its Preparation

### 4.2.1 The corpus used

This study trains and evaluates exclusively on CASME II [1]. No other corpus reviewed in Chapter 3 — not SAMM, not CAS(ME)$^2$, not the original CASME — contributes a single clip to any of the twelve configurations described in §4.1: every number reported in Chapter 5 is a within-corpus result. Chapter 3 §3.1 reviews the corpus literature and the reasons CASME II is a standard choice for micro-expression work, and Chapter 2 §2.2 describes what the corpus's recording hands the pipeline as input — the original frames, uncropped and unregistered; neither is repeated here. What follows is what this study specifically drew from the corpus, and how that selection was reduced to the set of clips actually used for training and evaluation.

### 4.2.2 From 255 rows to 156 clips

The pipeline's label table, `Processed_Data/master_thesis_labels.csv`, holds 255 rows. All 255 belong to CASME II, and all 255 are coded `micro-expression` — this is the dataset already filtered by the first two of the four row-level conditions described in Chapter 2 §2.2.4. Its raw label distribution, recorded under the column `Unified_Emotion`, is: Negative 99, Others 99, Positive 32, Surprise 25.

Of these four categories, `Others` is excluded before any model is trained, leaving 156 clips. `Others` is not itself an emotion label; it is the residual category into which the corpus's original coding scheme places clips that do not fit cleanly into one of the named expression classes. A clip coded `Others` therefore carries no positive claim about what expression it shows, only that it does not fit elsewhere. Because the task this thesis addresses is recognising a named expression, a category that by construction names nothing cannot support a training signal for that task, and including it would ask the network to learn a class defined by exclusion rather than by content. `Others` is accordingly dropped, and the 99 Negative, 32 Positive and 25 Surprise clips — 156 in total — form the working set for every configuration described in §4.1.

### 4.2.3 The three-class grouping

The 156 retained clips are not classified under CASME II's original per-emotion labels. They are mapped to three coarser classes by `GROUPED_EMOTION_MAP`, which assigns `{"Negative": 0, "Positive": 1, "Surprise": 2}`. This mapping is constructed from the underlying raw CASME II categories as follows: disgust, fear, repression and sadness are grouped into Negative; happiness is mapped to Positive; surprise is kept as its own class, Surprise. The resulting three-class distribution is 99 : 32 : 25 for Negative : Positive : Surprise, a ratio of roughly 4 : 1.3 : 1 — markedly imbalanced, with Negative outnumbering Surprise by a factor of nearly four.

This three-way grouping is not a choice made for this thesis; it is the convention already established in the field and reviewed in Chapter 3 §3.1.3, adopted here rather than re-argued. It is stated in full in this section only because §4.1's ablation and Chapter 5's results are reported against these three classes, and the exact class boundaries and their resulting imbalance are needed to interpret those results correctly — in particular, to distinguish a genuine effect of an ablated component from an artefact of the class imbalance itself.

### 4.2.4 Subjects and the leave-one-subject-out protocol

CASME II records each clip against the subject who produced it. Across the full 255-row micro-expression set — the corpus before the `Others` exclusion described in §4.2.2 — 26 distinct subjects appear. Once `Others` is excluded and the three-class subset of 156 clips is formed, however, only 25 of those subjects remain represented: subject 18 contributes only clips coded `Others`, and none coded Negative, Positive or Surprise, so that subject disappears entirely once the exclusion in §4.2.2 is applied.

This distinction matters directly to the evaluation protocol used throughout this thesis. The leave-one-subject-out cross-validation scheme described in Chapter 2 §2.8 is run over the 156-clip three-class subset, and consequently produces 25 folds, one held-out subject at a time — not 26. The number 26 describes the subject count of the corpus as recorded in the label table before class filtering; the number 25 describes the subject count of the data this thesis actually trains and evaluates on. Reporting either figure in place of the other would misstate the cross-validation protocol, so both are given here explicitly rather than only one being carried forward.

A further consequence of subject-level splitting on a corpus this size is that fold sizes are uneven by construction: each fold's held-out set is exactly one subject's clips, and CASME II subjects do not contribute equal numbers of clips, so one fold may evaluate on a handful of clips while another evaluates on considerably more. This is a property of leave-one-subject-out evaluation applied to CASME II generally, inherited rather than introduced by this study's ablation design, and is noted here because every accuracy figure reported per fold in Chapter 5 should be read against it.

### 4.2.5 Preparation for the network

Before reaching any of the four ablatable components described in §4.1.2, each of the 156 clips is reduced to a single three-channel motion tensor of fixed shape $[3, 32, 224, 224]$: 32 temporal frames, each $224 \times 224$, with the three channels holding horizontal optical flow, vertical optical flow, and optical strain magnitude respectively. The procedure by which a variable-length sequence of greyscale frames — the corpus's original recordings, uncropped and unregistered (§2.2.2) — is converted into this fixed-shape tensor — frame selection, flow computation, strain derivation, and the role of motion magnification in producing the flow the tensor encodes — is described in Chapters 2 §2.2 and §2.4, and is not re-derived here. The values that procedure was run with are recorded here, since they are fixed throughout the study and are needed to reproduce it. Magnification, where a configuration enables it, uses an amplification factor of $\alpha = 10$, a temporal band of 5–25 Hz and four pyramid levels, with the amplified output clipped to $[0,255]$ and requantised to 8-bit (§2.3.5 explains the band and §3.2.7 records the divergence it embodies). The frames spanning onset to offset are uniformly resampled to 33, and flow and strain are then computed exactly as §2.4.4 specifies — the Farnebäck settings and the gradient discretisation are given there and are not repeated — after which each of the three channels is min–max normalised to $[0,1]$ across the clip before the tensor is stored. None of these values is varied by the ablation. What matters for this chapter is only that every one of the 156 clips, regardless of which of the twelve valid ablation configurations subsequently processes it, enters the network as a tensor of this same fixed shape, so that no configuration in §4.1's design receives an input representation that differs from any other's.

---

## 4.3 The Ablation Matrix

### 4.3.1 The twelve trained configurations

Table 4.1 lists the twelve configurations retained after the validity check of §4.1.3, exactly as they are named in the codebase (`Ablation_Study/ablation_config.py`). Each row is a fixed setting of the four boolean flags introduced in §4.1.2; `T`/`F` stand for `True`/`False`.

**Table 4.1 — The twelve valid ablation configurations.**

| name | evm | simam | cnn | transformer | descriptor |
|---|---|---|---|---|---|
| `config_1_pure_base` | F | F | F | F | no component active; raw motion tensor pooled and classified directly |
| `config_2_temporal_only` | F | F | F | T | transformer only, on unmagnified input |
| `config_3_spatial_only` | F | F | T | F | convolutional stem only, on unmagnified input |
| `config_4_motion_amp_base` | T | F | F | F | magnification only, no learned spatial or temporal component |
| `config_5_attention_base` | F | T | T | F | convolutional stem with SimAM, no magnification, no transformer |
| `config_6_full_stage2_noevm` | F | T | T | T | full learned stack without magnification |
| `config_7_full_no_attention` | T | F | T | T | magnification, CNN and transformer, without SimAM |
| `config_8_proposed_unified` | T | T | T | T | all four components active |
| `config_9_permutation` | F | F | T | T | CNN and transformer, no magnification, no SimAM |
| `config_12_permutation` | T | F | F | T | magnification and transformer, no CNN, no SimAM |
| `config_13_permutation` | T | F | T | F | magnification and CNN, no SimAM, no transformer |
| `config_16_permutation` | T | T | T | F | magnification, CNN and SimAM, no transformer |

### 4.3.2 The four excluded cells

Of the sixteen cells generated by the full Cartesian product of the four flags, four are rejected by `AblationConfig.is_valid()` and never trained. All four share `use_simam=True, use_cnn=False`, the combination §4.1.3 identifies as architecturally ill-formed rather than merely weak:

- `config_10` — evm F, simam T, cnn F, transformer F
- `config_11` — evm F, simam T, cnn F, transformer T
- `config_14` — evm T, simam T, cnn F, transformer F
- `config_15` — evm T, simam T, cnn F, transformer T

§4.1.3 already gives the reason for the exclusion; it is not repeated here.

### 4.3.3 Naming is not contiguous

Configurations 1 through 8 carry the descriptive names of the original eight-cell design that preceded the move to a full factorial. Configurations 9, 12, 13 and 16 were added when that design was completed to all sixteen generated cells, and each carries the generic suffix `_permutation` rather than a descriptive name. The numbering therefore skips 10, 11, 14 and 15 — the four excluded cells of §4.3.2 — and a reader scanning results tables by configuration number will find that gap. It is a consequence of how the sixteen candidates are generated and filtered, not an error in the results or in this account of them.

### 4.3.4 The baseline, and the control it is read against

Two configurations are easily confused and serve different purposes in the design:

- **`config_4_motion_amp_base` is the study's baseline.** It applies Eulerian motion magnification and nothing else — no SimAM, no convolutional stem, no transformer — and is the reference point against which the contribution of every learned component is ultimately read.
- **`config_1_pure_base` is not a second baseline.** It is the EVM-off control: identical to `config_4` in every flag except magnification. Its purpose is to isolate the effect of magnification itself by matched-pair comparison against `config_4` (§4.3.5), not to serve as an alternative starting point for the study.

`config_8_proposed_unified` is the full system, with all four components active, and is not privileged in the design beyond that: it is one of twelve cells, not the object the ablation is built to justify.

### 4.3.5 Matched pairs

Table 4.2 lists all twenty matched pairs, in the sense defined in §4.1.4, written `off → on`.

**Table 4.2 — Matched pairs by variable.**

| variable | pairs (off → on) | pair count |
|---|---|--:|
| Transformer | 1→2, 4→12, 3→9, 13→7, 5→6, 16→8 | 6 |
| EVM | 1→4, 3→13, 9→7, 6→8, 5→16, 2→12 | 6 |
| SimAM | 3→5, 9→6, 7→8, 13→16 | 4 |
| 3D-CNN | 1→3, 2→9, 4→13, 12→7 | 4 |

A variable's reported effect (Chapter 5) is the mean of the per-pair differences over the rows listed for it in Table 4.2. The pair count is reported alongside every such effect because an average computed over four comparisons is a noisier estimate of the same kind of quantity than one computed over six; §4.1.4 gives the reason the counts differ. Any comparison between the size of a four-pair effect and a six-pair effect in Chapter 5 should be read with this asymmetry in mind.

No accuracy, F1 score, or ranking of configurations is reported in this section; those results, and the marginal effects computed from the pairs above, belong to Chapter 5.

---

## 4.4 Model Architecture

### 4.4.1 Which implementation this is

Every result reported in this thesis was produced by a single implementation: `AblationMERModel`, defined in `Ablation_Study/models.py`. A second implementation exists in the repository, `Stage2_Architecture/models/hybrid_model.py`; it is an earlier, fixed prototype with no ablation toggles and was not used to produce any run described here. A reader browsing the codebase will find both — only the former is the subject of this section and of Chapter 5.

`AblationMERModel` is a single `nn.Module` whose forward pass conditionally executes or skips each of the three model-level components (SimAM, the 3D-CNN backbone, the transformer) according to the boolean flags fixed at construction; the fourth variable, motion magnification (`use_evm`), is applied at the data level, before a tensor reaches the model, and is not part of the network graph described in this section. Building all twelve configurations as branches of one module, rather than as twelve separate model classes, is what allows the matched-pair logic of §4.3 to hold exactly: the shared code path guarantees that any two configurations differing in one flag are identical in every other respect, down to weight initialisation and layer ordering, because they are constructed by the same class with the same remaining arguments.

### 4.4.2 Input and the three streams

The model receives the tensor described in §4.2.5 with a batch dimension prepended, giving `[B, 3, 32, 224, 224]`. Each of its three motion channels is routed to its own convolutional stream with **unshared weights** — `ThreeStreamCNNBackbone` instantiates three independent copies of the stem described below, one per channel. Three streams therefore means three motion channels processed independently, not three architectural stages of a single pipeline.

### 4.4.3 The convolutional stem

When the 3D-CNN backbone is active (`use_cnn=True`), each stream applies the following stack, implemented as `SingleStream3DCNN`:

```
Conv3d(1 → 16, kernel=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
BatchNorm3d → ReLU → Dropout3d(p=0.3)
Conv3d(16 → 32, kernel=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
BatchNorm3d → ReLU → Dropout3d(p=0.3)
MaxPool3d(kernel=(1,2,2), stride=(1,2,2))
```

**No kernel in this stem spans the temporal axis**, for the reason §2.6.1 sets out. Concretely, that leaves the sequence length $T=32$ unchanged through the entire stem, while height and width are halved once, from $224$ to $112$.

Per stream, the shape transformation is $[B,1,32,224,224] \rightarrow [B,32,32,112,112]$. The three streams are concatenated along the channel dimension to give $[B,96,32,112,112]$.

### 4.4.4 SimAM placement

When SimAM is active (`use_simam=True`, and only meaningful when `use_cnn=True`, for the reason given in §4.1.3 — the model forces `use_simam` to `False` internally otherwise), it is applied **per stream**, immediately after that stream's convolutional output and before the three streams are concatenated. Each of the three streams carries its own `SimAM3D` instance. The gate itself is as given in §2.6.3 [8], applied to the five-dimensional feature map $x \in \mathbb{R}^{B \times C \times D \times H \times W}$ each stream produces, with the variance Bessel-corrected over $n = D \cdot H \cdot W - 1$. No weight or bias is learned anywhere in it, so SimAM introduces **zero learnable parameters** (confirmed in the totals of §4.4.9) regardless of how many streams or channels it is applied to. The regularisation constant is fixed at $\lambda = 1 \times 10^{-4}$ throughout, matching `SimAM3D`'s default and the value passed at construction.

### 4.4.5 The transformer

The transformer's input sequence is formed from the concatenated feature map $[B,96,32,112,112]$ by `AdaptiveAvgPool3d((32,1,1))`, which collapses the two spatial dimensions while leaving the temporal dimension at 32. After squeezing and permuting the singleton spatial dimensions away, this yields a sequence $[B,32,96]$: 32 time steps, each described by a 96-dimensional feature vector.

This sequence is passed to `SLSTTTransformer`, a `nn.TransformerEncoder` configured as follows:

- $d_{\text{model}} = 96$
- $n_{\text{head}} = 8$
- number of encoder layers $= 4$
- feed-forward dimension $= 256$
- dropout $= 0.1$
- positional encoding: fixed sinusoidal (parameter-free)
- `norm_first=True` (pre-norm layer ordering)
- activation: GELU
- `batch_first=True`

The positional encoding follows the standard fixed sinusoidal form, added to the sequence before it enters the first encoder layer: for position $p$ and feature index $2i$ (even) or $2i+1$ (odd) of the 96-dimensional embedding,

$$\mathrm{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/96}}\right), \qquad \mathrm{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/96}}\right).$$

This encoding is a fixed buffer, not a learned parameter, and contributes nothing to the parameter counts in §4.4.9. The encoder output, still a sequence of 32 steps, is reduced to a single vector by **mean-pooling over the time axis**, giving $[B,96]$.

### 4.4.6 The classifier head

Every configuration, regardless of which upstream components are active, ends in the same head:

```
LayerNorm(96) → Dropout(p=0.3) → Linear(96, 3)
```

### 4.4.7 Fallbacks for disabled components

When a component is switched off, it is not simply removed from the graph; it is replaced by a deliberately minimal substitute so that the disabled configuration remains a comparable, fair control rather than an incomplete network with a truncated forward pass.

- **`use_cnn=False` → `RawPatchEmbedding`.** The raw input $[B,3,32,224,224]$ is average-pooled with `AdaptiveAvgPool3d((32,4,4))`, leaving the temporal axis untouched and crushing each frame to a $4\times4$ spatial grid across all three channels. The result is reshaped to $[B,32,48]$ (three channels $\times$ $4\times4$ grid) and passed through a single `Linear(48, 96)` layer to match the sequence width the downstream stage expects. This fallback **adds 4,704 parameters**.
- **`use_transformer=False` → `TemporalPooling`.** The 32-step sequence is collapsed to one vector by a plain mean (or max) over the time axis. This fallback has **zero parameters** and, critically, no notion of frame order or inter-frame dependency of any kind — whatever temporal structure a micro-expression's onset-apex-offset carries is unavailable to any configuration using it.

### 4.4.8 Weight initialisation

Every configuration is initialised by the same routine, applied once at construction and independent of which components are present: `Conv3d` weights are initialised with Kaiming-normal initialisation (`fan_out` mode, ReLU nonlinearity), `Linear` weights with Xavier-uniform initialisation, and the scale and shift parameters of every `BatchNorm3d` or `LayerNorm` layer are initialised to one and zero respectively. All biases, where present, are initialised to zero. Because this routine is applied uniformly across the twelve configurations, differences in outcome between two matched configurations cannot be attributed to a difference in how their shared components were initialised.

### 4.4.9 Parameter counts

Four components combine to give the total parameter count of any configuration; the classifier head and, where present, SimAM are common to every configuration that uses them (SimAM contributing zero parameters). The components' individual counts are:

- Convolutional backbone (`ThreeStreamCNNBackbone`, all three streams): **14,544**
- Transformer encoder (`SLSTTTransformer`): **348,736**
- `RawPatchEmbedding` (no-CNN fallback): **4,704**
- Classifier head: **483**

These combine into four distinct totals, one per combination of the two components that vary parameter count (Table 4.3). SimAM (zero parameters) and EVM (a pre-processing step, not a network component) do not appear as separate rows because neither changes a configuration's parameter count.

**Table 4.3 — Parameter totals by architectural combination.**

| configuration type | example | parameters |
|---|---|--:|
| no CNN, no transformer | `config_1`, `config_4` | 5,187 |
| CNN, no transformer | `config_3`, `config_5`, `config_13`, `config_16` | 15,027 |
| no CNN, transformer | `config_2`, `config_12` | 353,923 |
| CNN and transformer | `config_6`, `config_7`, `config_8`, `config_9` | 363,763 |

In a full configuration, the transformer encoder accounts for roughly **96%** of total parameters (348,736 of 363,763), while the convolutional backbone carries only about 14.5 thousand. This is reported here as a fact about the implementation's parameter budget; it is not a claim about which component contributes more to classification performance, which is a question for Chapter 5.

---

## 4.5 Training Procedure

Every one of the twelve valid configurations of §4.1 is trained by the same procedure: the same loss, the same sampler, the same optimiser and schedule, the same numerical safeguards, and the same seed. This section states that procedure's fixed values, exactly as they are configured in `Ablation_Study/ablation_config.py` and implemented in `Ablation_Study/losses.py` and `Ablation_Study/run_ablation_experiments.py`. §2.7 describes what each mechanism computes; nothing here repeats that derivation.

### 4.5.1 Loss

The classification objective is focal loss with $\gamma = 2.0$ and label smoothing of $0.05$ (§2.7.2 gives the mechanism this configuration instantiates). Both values are fixed defaults on `ExperimentConfig` and are identical across every configuration and every fold.

### 4.5.2 The class-weighting rule

`ExperimentConfig` exposes an optional per-class $\alpha$ weight vector for the loss, and `use_class_weights` defaults to `True`. It is not, however, the flag that decides whether $\alpha$ is actually applied. The effective switch, computed at the start of every training run, is:

```python
use_loss_weights = self.exp.use_class_weights and not self.exp.use_balanced_sampler
```

`use_balanced_sampler` also defaults to `True` (§4.5.3), and no run reported in this thesis overrides either default. Consequently **the $\alpha$ term is inactive in every run reported in this thesis**: `use_loss_weights` evaluates to `False`, `class_weights` is never constructed, and `FocalLoss` is instantiated with `alpha=None`. The code states this outcome explicitly at run time, logging `"Class weights in loss disabled (balanced sampler already active)."` whenever both flags are on, which is every run here.

The intent behind the rule, and the reason it is not simply an oversight, is that the $\alpha$ term and the balanced sampler of §4.5.3 both correct for the same class skew, at two different points in the pipeline — one by re-weighting the loss, the other by re-weighting which examples a batch draws from — and applying both at once corrects for that skew twice. §2.7.4 sets out why this codebase resolves that redundancy by disabling the loss-side correction whenever the sampler-side one is active, rather than defaulting to both; that argument is not repeated here. What must be stated precisely is what remains active once $\alpha$ is switched off: focal loss still operates through its difficulty-focusing term, $(1-p_t)^\gamma$, which down-weights confidently correct predictions regardless of class, and through label smoothing, which regularises against over-confident targets. Neither of those mechanisms looks at class frequency. It is only the explicit, frequency-based $\alpha$ re-weighting that is disabled — the loss used throughout this thesis is focal loss with label smoothing and no per-class weighting, and every run's frequency correction, such as it is, comes entirely from the sampler.

### 4.5.3 Sampler

The training loader for every fold and every configuration uses a `WeightedRandomSampler`. Each training example is assigned a weight equal to $1/n_c$, where $n_c$ is the number of training examples belonging to that example's class within the current fold's training split; sampling proceeds `with replacement=True`, drawing as many indices per epoch as the training split contains. This is the mechanism that makes the class-weighting rule of §4.5.2 well founded rather than merely convenient: the correction for class skew genuinely happens here, so switching off the equivalent correction in the loss avoids double-counting it rather than removing it altogether.

### 4.5.4 Optimiser and learning-rate schedule

Every configuration is optimised with AdamW, learning rate $1\times10^{-4}$, weight decay $1\times10^{-4}$, applied uniformly across all parameters. The learning-rate schedule combines a linear warmup with cosine annealing: `LinearLR` with `start_factor=0.1` ramps the rate from a tenth of its target value to full over `warmup_epochs=5`; a `SequentialLR` wrapper then hands off, at the warmup boundary, to `CosineAnnealingLR` with `T_max = epochs - warmup_epochs` and `eta_min=1e-7`, which anneals the rate to that floor over the remaining epochs. Both schedulers are stepped once per epoch. Training runs for **50 epochs** in total, with no early stopping — every configuration and every fold trains the full 50 epochs regardless of when its best validation epoch occurred.

### 4.5.5 Batch size

`ExperimentConfig` declares `batch_size: int = 2` as its default, annotated in the source as a deliberately small value chosen because the transformer-bearing configurations use approximately 10 GB of VRAM at batch size 4. That default was not, however, the value used to produce the results in this thesis. The GPU runner, `tools/run_ablation_gpu.py`, invokes the training script with an explicit override:

```python
cmd = [
    sys.executable, str(runner),
    "--device", "cuda",
    "--batch_size", "4",
    *sys.argv[1:],
]
```

Neither that default nor the GPU runner's override is the value that produced the results reported here. The sweep was launched from the project's graphical front-end, whose persisted state is committed as `gui_settings.json` and records `"ablation_batch_size": "8"` alongside the protocol, fold and epoch settings of the runs of record; `tools/run_gui.py` passes that value through to the training script. **Every result reported in Chapter 5 was therefore produced with batch size 8.**

Two details are worth recording, because a reader reconstructing the command from the source alone would get a different answer. First, the GPU runner's `--batch_size 4` is not authoritative even when it is used: it places `*sys.argv[1:]` after its own arguments, so any value supplied downstream wins under `argparse`. Second, the run of record did not pass through that runner at all — the command is `run_ablation_experiments.py` invoked directly. No result file stores the batch size, so the evidence for 8 is the committed launcher state and the run command recorded with it, not the artefacts themselves.

### 4.5.6 Numerical safeguards

Three safeguards operate during every training step. Gradients are clipped to a global norm of $1.0$ via `clip_grad_norm_`, applied after the AMP gradient scaler's `unscale_` step so that clipping acts on true, not scaled, gradient magnitudes. Before the clipped gradients are allowed to reach the optimiser, every parameter's gradient is checked for non-finite values; if any parameter's gradient contains a NaN or an infinity, the optimiser step is skipped entirely, the gradients are zeroed, and a warning is logged, rather than allowing a corrupted update to reach the weights. Automatic mixed precision is enabled (`use_amp=True` by default) but is effective only when the device is CUDA — the trainer constructs its `use_amp` flag as `use_amp and device.type == "cuda"`, so a CPU run trains in full precision regardless of the configuration flag's value.

### 4.5.7 Seeding and replication

A single global seed, `42`, is reset — via `random.seed`, `numpy.random.seed`, `torch.manual_seed` and `torch.cuda.manual_seed_all` — once at the start of the whole experiment run and again immediately before every individual training-and-evaluation call, meaning at the start of every configuration and, under the LOSO protocol of §4.6, at the start of every one of its 25 folds. This guarantees that the sampler's draw order, the model's weight initialisation, and any other stochastic step begin from the same fixed state each time a fold is trained.

It does not guarantee, and is not intended to guarantee, anything about how sensitive the reported outcome is to that state. **Every configuration was trained exactly once, under this one seed, for every fold. No configuration was retrained under a second seed, and no seed variance was estimated anywhere in this study.** §4.1 raises this as a property of the experimental design; stated concretely in terms of the training procedure, it means that a single run of 50 epochs on one initialisation is the entire evidentiary basis for each fold's contribution to a configuration's aggregate score, A different seed could in principle move any individual fold's result, and by how much is not known from this design. §4.1.5 states the consequence for what may be claimed.

### 4.5.8 No hyperparameter search

Every value in this section — the focal loss parameters, the learning rate and weight decay, the warmup length and annealing floor, the batch size, the clipping threshold, and the seed itself — is a fixed default on `ExperimentConfig`, or a single hard-coded override in the GPU launch script, carried unchanged across all twelve configurations and all 25 folds of every configuration. No grid search, random search, or manual tuning pass was run over any of these values, for any configuration, at any point in this study. This is stated explicitly because the ablation design of §4.1 could otherwise be misread as having tuned each configuration to its best achievable performance before comparing it with the others; it has not. Every comparison in Chapter 5 is a comparison between architectures trained under one common, untuned recipe, not between architectures each given the chance to be tuned to its own advantage.

---

## 4.6 Evaluation Protocol

Every configuration in §4.1 is scored by the same protocol: the same fold structure, checkpoint-selection rule, cross-fold aggregation, and persisted output files, exactly as implemented in `Ablation_Study/dataset.py`, `run_ablation_experiments.py` and `metrics.py`. §2.8 gives the general mechanism behind macro-F1, pooling and subject-disjoint cross-validation; §3.1.6 and §3.1.8 argue why this corpus calls for the choices below. Neither is re-derived here — this section states what was actually run, and what its properties are.

### 4.6.1 Leave-one-subject-out folds

Evaluation uses leave-one-subject-out (LOSO) cross-validation. §4.2 establishes that the 156-clip, three-class working set retains 25 of CASME II's 26 subjects — subject 18 contributes only clips excluded from that subset — so LOSO over this data produces **25 folds**, each holding out one subject's clips and training on the remaining 24 subjects'. Every stored result confirms this: `final_results.json` records `"protocol": "loso"`, `loso_folds_run: 25`, `loso_folds_total: 25`, and `loso_pilot: false`. The codebase supports a pilot mode, `select_loso_folds(max_folds=...)`, selecting an evenly-spaced subject subset for fast development iteration; it was not used for any result reported here. Every reported figure reflects the full 25-fold sweep.

### 4.6.2 Checkpoint selection and the absent inner validation split

Within each fold, the trainer of §4.5 tracks validation macro-F1 after every epoch and keeps the weights from whichever epoch scored highest — `trainer.fit(train_loader, val_loader)`, then `trainer.load_best()`. The fold's final score is then obtained by evaluating that restored checkpoint on `val_loader` a second time: `trainer.evaluate(val_loader)`. In both calls, `val_loader` holds the same held-out subject's clips.

Stated plainly, because it is the single most significant methodological limitation of this study's evaluation protocol: **there is no inner validation split**. The held-out subject's clips are used twice — once as the signal deciding which epoch's weights to keep, and again as the data that kept epoch is finally scored against. A genuine inner split would carve a further validation subset out of the training subjects, select the checkpoint against that, and reserve the held-out subject purely for a final, unseen evaluation; this pipeline does not do that. The consequence is an optimistic bias of unknown size in every fold's reported score: model selection has already seen the exact data it is subsequently scored on.

The magnitude of that bias cannot be quantified from the stored output (§4.6.6), but its scale can be reasoned about from the fold structure itself. Each fold's held-out subject often contributes only a handful of clips — §4.2.4 and §3.1.7(c) record fold sizes varying by more than a factor of thirty — so in the smallest folds the checkpoint search is a selection over 50 epochs against a handful of examples, with considerable freedom to fit noise in the very set it is later judged against. This is not a minor caveat to Chapter 5's numbers; it is a property of every fold's score, recorded here, at the point the protocol is fixed, rather than deferred to a discussion of results.

One further value should be noted so it is not mistaken for an alternative already in use here: `val_fraction = 0.2` exists on `ExperimentConfig`, consumed by `dataset.subject_disjoint_split` under a separate `"holdout"` protocol — a single subject-disjoint split, evaluated once rather than 25 times. That protocol produced none of the results reported in this thesis; every figure comes from the `"loso"` branch above, where `val_fraction` plays no role.

### 4.6.3 Aggregation across folds

Having trained and scored all 25 folds, `MetricsComputer.average_results` combines them into the single result stored for that configuration. It computes, and `final_results.json` persists:

- `accuracy` — the **mean of the 25 per-fold accuracies**.
- `macro_f1` — the **mean of the 25 per-fold macro-F1 scores**.
- a confusion matrix obtained by **summing** the 25 per-fold confusion matrices, element-wise, into one pooled $3\times3$ matrix.
- `per_class_f1`, `per_class_precision`, `per_class_recall` — precision, recall and F1 computed per class **from that pooled matrix**, not averaged from per-fold values.
- `micro_f1` — the pooled matrix's trace over its total, which for single-label multi-class classification is pooled accuracy (§2.8.2).

The distinction between the first two quantities and the rest matters throughout the rest of this section: `accuracy` and `macro_f1`, as the code names them, are averages of per-fold statistics, while everything else is computed once, over all folds' predictions pooled together.

### 4.6.4 The primary metric: pooled macro F1

This thesis's primary reported metric is **pooled macro F1** — the mean of the three values in the stored `per_class_f1` array. **No run artefact stores this quantity under any key.** It exists nowhere in `final_results.json`, `summary.csv`, or any other artefact written by the training run; it is derived after the fact — by the reporting scripts under `tools/`, and for the figures of Chapter 5 — by averaging the three `per_class_f1` numbers. A reader expecting the headline number under a field called `macro_f1` will instead find the per-fold-averaged quantity of §4.6.3 — a different estimator, addressed next — and will not find pooled macro F1 recorded anywhere.

### 4.6.5 Why mean-of-folds macro F1 is rejected

Pooled macro F1 is used as the primary metric, and the stored `macro_f1` field is not, because of how CASME II's 156 clips distribute across the 25 held-out subjects. Of the 25 folds, **10 contain clips of only one class**, **8 contain clips of two classes**, and **7 contain all three**. A macro F1 computed within a fold is the unweighted mean of that fold's three per-class F1 scores, and a class entirely absent from a fold's ground truth contributes an F1 of zero regardless of what the model does. A single-class fold can therefore reach at most $1/3$; a two-class fold, at most $2/3$; only a three-class fold can reach $1$ — ceilings fixed by which subject was held out, before any model is trained, unrelated to classification accuracy.

Averaging macro F1 across the 25 folds therefore averages seventeen sub-unity ceilings into every configuration's score, regardless of how it performs. Taking each fold at its own maximum — $1/3$ for the 10 single-class folds, $2/3$ for the 8 two-class folds, $1$ for the 7 three-class folds — bounds the mean-of-folds macro F1 at
$$\frac{10 \cdot \tfrac13 + 8 \cdot \tfrac23 + 7 \cdot 1}{25} = \frac{3.333\ldots + 5.333\ldots + 7}{25} \approx 0.627.$$
No configuration, however accurately it classifies every clip it sees, can exceed approximately **0.627** on the mean-of-folds macro F1 stored under that name. This ceiling is a property of the fold structure CASME II's subject distribution imposes, not of any model evaluated against it, and it applies identically to every one of the twelve configurations in §4.1.

Pooling the confusion matrices before computing macro F1, as §4.6.3 describes, avoids this ceiling entirely: TP, FP and FN are accumulated across all 25 folds — across all 156 clips — before any per-class F1 is computed, so a class locally absent from one fold no longer forces a zero into that fold's contribution. §3.1.6 sets out this argument in full and establishes that pooled macro F1, computed this way, is identical in construction to the Unweighted F1 (UF1) metric of the MEGC 2019 benchmark protocol [11]; that argument is not repeated here. The consequence recorded in this section is arithmetic: a metric with a hard ceiling below 0.7 cannot serve as this study's headline result, and the pooled quantity of §4.6.4 is used instead.

The same fold composition biases mean-of-folds **accuracy** in the same direction, without an equivalent hard ceiling: a single-class fold is easy to score well on, since one predicted label suffices for perfect accuracy on it, whereas pooled accuracy is computed once over the correctly-distributed set of 156 clips. For the full configuration, mean-of-folds accuracy exceeds pooled accuracy by **6.3 points**. No configuration is named here, and no other score from either metric is given — that comparison belongs to Chapter 5. Both figures are stated only because they are properties of the evaluation protocol itself, fixed before any model is trained.

### 4.6.6 What is stored

For each configuration, `Ablation_Study/results/<config_folder_name>/` holds: `final_results.json` (the metrics of §4.6.3, plus `toggles`, `class_names`, and an `extra` block carrying the protocol metadata of §4.6.1); `confusion_matrix.npy` and a rendered `confusion_matrix.png`; and `configuration_summary.txt`, recording toggle settings and hardware statistics. Every configuration also contributes one row to a master `summary.csv`.

Two further artefacts are written per configuration but are not aggregates: `training_metrics.csv` and `checkpoints/best_model.pth` are overwritten on every fold's completion, so what survives after a configuration finishes is the training curve and checkpoint of its **last fold only** — not a representative fold, not the best fold, simply whichever subject was held out last.

**Per-clip predictions are not saved at all.** No file records which fold evaluated a given clip, what the model predicted, or whether the prediction was correct — only the aggregate metrics and matrices of §4.6.3 persist. This is why no paired significance test between any two of the twelve configurations can be constructed from the stored output: such a test needs to know which specific clips each configuration got right or wrong, and that record does not exist.

No score, ranking, or marginal effect between configurations is reported in this section. The 0.627 ceiling and the 6.3-point gap are the only figures given here, because both describe the protocol rather than any model's performance under it; every result the protocol produces is reported in Chapter 5.

### 4.6.7 Earlier evaluations, and why they are excluded

The protocol specified above is not the only evaluation this project ran over the ablation matrix of §4.3. Four earlier sweeps preceded it, each producing a complete set of scores for the configurations and clips it covered. They are recorded here rather than dropped without mention, because what disqualifies each of them is a property of the evaluation itself — its routing, its class support, its coverage — and is therefore a matter for this chapter rather than for Chapter 5.

**Table 4.4 — The five evaluations of the ablation matrix, and the status of each.**

| run | protocol | clips | configurations | support Neg/Pos/Sur | EVM pairs bit-identical | top-ranked configuration |
|---|---|--:|--:|---|---|---|
| A | holdout, 60 epochs | 39 | 12 | 29 / 9 / 1 | 6 of 6 | `config_5_attention_base`, 0.7427 (tied) |
| B | LOSO, 5 of 25 folds | 24 | 8 | 19 / 5 / 0 | 2 of 2 | `config_6_full_stage2_noevm`, 0.6274 |
| C | holdout, 50 epochs | 52 | 12 | 39 / 10 / 3 | 0 of 6 | `config_13_permutation`, 0.4575 |
| D | LOSO, 20 of 25 folds | 139 | 12 | 92 / 30 / 17 | 0 of 6 | `config_16_permutation`, 0.5473 |
| E | LOSO, 25 of 25 folds | 156 | 12 | 99 / 32 / 25 | 0 of 6 | `config_2_temporal_only`, 0.7122 |

**The imbalance treatment also changed across these runs, and that is the largest difference between them and run E.** The rule of §4.5.2 — loss-side class weighting active only when the balanced sampler is not — was not present for most of this history. The earliest runs applied inverse-frequency class weights in the loss unconditionally, with no sampler in the codebase at all. A later revision added the `WeightedRandomSampler` while leaving the loss weighting unconditional, so both corrections acted simultaneously. The guard was introduced only afterwards, and run E is the first full evaluation conducted under it. §3.9.7 reports what each of those regimes did to the per-class predictions; the point here is narrower, that runs A–D are not comparable to run E on the training regime either, independently of the protocol differences the table records.

Every score in Table 4.4 is pooled macro F1, recomputed from each run's stored `per_class_f1` array; the `macro_f1` and `accuracy` keys those files also carry are the mean-of-folds quantities rejected in §4.6.5, and are used nowhere. Run E is the protocol of §4.6.1 to §4.6.6 and the only one Chapter 5 reports.

Runs A and B carry the data-routing defect of §3.2.7. In run A all six magnification matched pairs are bit-identical, and in run B both of the two pairs its eight configurations contain are likewise identical, so in each case one half of the matrix duplicates the other and the magnification variable was never in fact exercised. Run A's leading score is consequently a tie between a configuration and its own duplicate. Runs C and D show all six pairs differing: the repair was made before run C, not by the reported run.

Runs A and B also lack usable class support. Run A's Surprise class contains one clip — its pooled confusion matrix is `[[23, 6, 0], [5, 4, 0], [0, 0, 1]]` — and that clip was classified correctly, so a per-class F1 of 1.0 contributes a full third of the macro mean on the evidence of a single example. This is why run A's 0.7427 stands above the reported run's 0.7122 while resting on a fraction of the data. Run B contains no Surprise clips at all, so its mean is taken over two classes that are present and one that is not. §4.2 gives the corpus's class distribution and the scarcity of Surprise within it.

Coverage is incomplete in two further senses. Run B scores only eight of the twelve configurations, so it does not span the matrix and the matched-pair effects of §4.3.5 cannot be computed from it. Runs B and D stop short of the full fold set — five and twenty of the twenty-five subjects — so neither holds out every subject, and the fold-composition variation §4.6.5 describes is sampled rather than covered. Only run E has all 25 folds, the full 156-clip working set, all twelve configurations, and all three classes at usable support.

A different configuration ranks first in each of the five runs. That is not evidence that protocol choice reorders an otherwise fixed ranking, and it is not offered as such: the runs differ in sample count, training budget and class coverage as well as in protocol, so no pair of them isolates a single factor, and the margins involved are in places negligible — run D's top two configurations are separated by 0.0007. What the sequence does show is that a ranking read off an incomplete evaluation of a corpus this size is not stable. That is the reason this thesis reports one complete run and no other, and the reason the four incomplete ones are listed above rather than quietly discarded.

---

## 4.7 Computational Environment

### 4.7.1 A gap in the record, stated before anything else

No file produced by this study's training runs records what hardware they ran on. `final_results.json`, `configuration_summary.txt`, `summary.csv` and the training logs contain no GPU name, no CPU identity, no RAM figure, and no recorded version of Python, PyTorch, or CUDA. A script exists that would have printed exactly this information — `tools/check_environment.py` reports the Python and PyTorch versions, CUDA availability, `torch.cuda.get_device_name(0)`, and the CUDA runtime version — but it is a standalone diagnostic, run on its own from the command line, and it is never called from `run_ablation_experiments.py`, `tools/run_ablation_gpu.py`, or any other path the twelve configurations' training runs actually executed. This is a limitation of what those runs left behind, not an omission from this account of them: no GPU model, CPU, memory figure, or library version is invented to fill the gap, and §4.7.5 states what can still be inferred from the code's behaviour.

### 4.7.2 What is recorded, and for one fold only

What each configuration's `configuration_summary.txt` does record is a training time and a peak VRAM figure, written by `ResultWriter` from a `TrainState`. Both describe **one fold, not the twenty-five-fold sweep**: the LOSO loop of §4.6.1 constructs a fresh `AblationTrainer` for every fold, each returning its own `TrainState`, and `total_train_time_sec` and `peak_vram_mb` are taken from whichever one is in hand when the summary is written rather than accumulated across folds. What survives once a configuration's sweep completes is whichever fold ran last — not a total, not an average, and not necessarily representative, exactly as §4.6.6 notes for `training_metrics.csv` and the saved checkpoint. A reader taking either figure below as a sweep total would overstate it by roughly a factor of twenty-five.

Read as single-fold snapshots, the twelve values are:

| configuration | last-fold train time (s) | peak VRAM (MB) |
|---|--:|--:|
| `config_1_pure_base` | 56.47 | 165.5 |
| `config_4_motion_amp_base` | 58.12 | 165.5 |
| `config_12_permutation` | 67.58 | 174.1 |
| `config_2_temporal_only` | 68.96 | 174.1 |
| `config_3_spatial_only` | 830.94 | 14,742.0 |
| `config_13_permutation` | 831.79 | 14,742.0 |
| `config_7_full_no_attention` | 832.81 | 14,747.4 |
| `config_9_permutation` | 832.81 | 14,747.4 |
| `config_5_attention_base` | 924.85 | 20,034.0 |
| `config_16_permutation` | 925.61 | 20,034.0 |
| `config_6_full_stage2_noevm` | 927.65 | 20,039.4 |
| `config_8_proposed_unified` | 930.63 | 20,039.4 |

### 4.7.3 The convolutional stem, not the transformer, sets the cost

The table splits cleanly in two. The four configurations without the 3D-CNN backbone — `config_1_pure_base`, `config_4_motion_amp_base`, `config_12_permutation`, `config_2_temporal_only` — run their last fold in **56–69 seconds** at **165–174 MB** of peak VRAM. The eight carrying that backbone run in **831–931 seconds** at **14.7–20.0 GB**: roughly **thirteen times the time and over one hundred times the memory**, and the split tracks `use_cnn` exactly, with no exception either way.

This is set by the stem, not by the transformer, even though the transformer holds the large majority of any configuration's parameters — about 96% of the total in a full configuration (§4.4.9). Parameter count and activation memory are different quantities. Per stream, the stem's second convolution reaches 32 channels while the spatial resolution is still the full $224 \times 224$ (§4.4.3); with three unshared streams and 32 time steps carried through untouched, the activations produced before the stem's single pooling step are correspondingly large, and it is activation memory, not weight count, that peak VRAM measures. The transformer, by contrast, operates on a sequence of only 32 steps at width 96, after the stem (or its fallback) has already collapsed the spatial extent. A component can dominate a configuration's compute and memory cost while holding a small fraction of its parameters; this is a statement about mechanism, not a result for Chapter 5.

### 4.7.4 GPU-hours: an extrapolation, not a measurement

The thesis's analysis scripts report a total of approximately **50.6 GPU-hours** across the sweep, of which about **48.9 hours — 96.6%** — is attributed to the eight CNN-bearing configurations. Both figures are extrapolations, not measurements: each configuration's single stored fold time (§4.7.2) is multiplied by 25, the fold count of §4.6.1, and divided into hours. This assumes every one of a configuration's 25 folds costs the same as the one fold whose timing survived; no other fold's time exists to check that assumption against.

The arithmetic is reproducible from the table above: the eight CNN-bearing configurations' stored fold times sum to 7,036 s, the remaining four to 251 s, for a sweep total of 7,287 s. Multiplied by 25 folds and divided by 3,600, this gives 50.6 hours overall, of which the CNN-bearing eight account for $7{,}036 \times 25 / 3600 \approx 48.9$ hours. The figure is the only cost summary the record supports, but it should be read for what it is: a single-fold time scaled by a fold count, not a sum of 300 measured fold durations.

### 4.7.5 Mixed precision, and what a reproduction would need

Mixed precision was enabled and, as §4.5.6 records, takes effect only on a CUDA device. The peak-VRAM figures in §4.7.2 exist at all only because `torch.cuda.max_memory_allocated()` returns a value on a CUDA device, so, although no GPU is named anywhere in the stored output, the runs that produced Chapter 5's results were GPU-based.

What is missing for exact reproduction is everything §4.7.1 already lists: the GPU model, the driver version, the CUDA runtime and PyTorch versions, and the wall-clock dates over which the sweep was run — none of which appears in any file this pipeline wrote. A future run of the same code, on different hardware or a different library version, could reproduce the same relative pattern between CNN-bearing and CNN-free configurations without reproducing these absolute timings or memory figures.

---

## 4.8 Conclusion

This chapter has specified, rather than argued for, what is needed to reproduce this thesis's experiments and read their results correctly. §4.1 set out a factorial ablation over four boolean components rather than a single architecture: sixteen cells from the full Cartesian product of the flags, four rejected as architecturally ill-formed because SimAM has no convolutional feature map to act on without the CNN stem, and the remaining twelve as the object of study, each component's effect read off matched pairs differing in exactly one flag. §4.2 fixed the shared data: CASME II's 255-row label set reduced, once the residual `Others` category is dropped, to 156 clips across 25 subjects in three classes — Negative, Positive, Surprise. §4.3 enumerated the twelve configurations and the twenty matched pairs — four each for SimAM and the CNN stem, six each for EVM and the transformer — over which §4.1's effects are computed. §4.4 described the shared architecture built to make those pairs exact: three unshared per-channel convolutional streams, an optional SimAM re-weighting, an optional transformer encoder, and a common classification head, each disabled component replaced by a near-parameter-free fallback rather than removed outright, so that what a flag removes is the component itself and not also a comparably sized replacement. §4.5 fixed the training procedure held constant across every configuration and fold: focal loss with label smoothing, a weighted sampler carrying the frequency correction, AdamW with warmup and cosine annealing over 50 epochs at batch size 8, and a single seed reset before every fold. §4.6 fixed the evaluation protocol: leave-one-subject-out cross-validation over the 25 remaining subjects, checkpoint selection by validation macro-F1 within each fold, and pooled macro-F1 — from confusion matrices summed across folds — as the primary metric, in preference to the mean-of-folds macro-F1 a subject's class composition can cap below 0.7 regardless of model quality. §4.7 closed the chapter with the computational conditions: GPU-based training of unrecorded hardware identity, with single-fold figures showing the convolutional stem, not the parameter-heavy transformer, driving roughly thirteen times the training time and over a hundred times the peak memory of the configurations that omit it.

Read together, these sections also fix the bounds within which Chapter 5's results may be interpreted, and none is softened by anything argued since it was first stated. A single fixed seed, with no repeat run and no variance estimate, means no difference between two configurations carries a confidence interval or a standard error. No hyperparameter search was run over any value in §4.5, so every comparison is between architectures given one common, untuned recipe, not each tuned to its own advantage. The protocol of §4.6 selects each fold's checkpoint against the same held-out subject it is then scored against, for want of an inner validation split, biasing every reported score optimistically by an amount this design cannot measure. And no per-clip prediction is saved, so no paired significance test between any two configurations can be constructed after the fact, however suggestive their pooled macro-F1 scores appear. These four limitations are not caveats appended to a result; they are properties of the design fixed here, bounding what any comparison in the next chapter is entitled to claim.

Chapter 5 now presents the pooled macro-F1 results this design and protocol produce, to be read strictly within the bounds these limitations impose.

---

