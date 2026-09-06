# Chapter 3 — Literature Review

## Scope and conventions

This chapter reviews the literature bearing on each component of the pipeline evaluated in this thesis. It is written to be read continuously: a fact is established once, in the section that owns it, and referred to by cross-reference thereafter. The corpus and its properties are established in §3.1, the metric convention in §3.1.6, and each component's measured contribution in the section that reviews it.

**Sources.** Every work cited is a paper held in the project's `docs/` corpus. Works cited only *inside* those papers are attributed in the text to the reviewed source that reports them, and are not listed as separate references; the reference list at the end of the chapter enumerates which works are handled this way. Two topics that appear in the project's reading list — Grad-CAM++ visual auditing and adversarial identity disentanglement — are excluded because the implemented system does not use them (§3.9).

**Structure.** Each section runs: what the technique addresses → how it works → the published evidence → its limitations → the implications for this thesis → a closing statement of the gap and how this work differs. §3.10 consolidates those ten closing statements into a single research gap, and Table 3.21 collects every point at which the implemented system departs from the literature it draws on.

**Verification.** Every quotation in this chapter was checked verbatim against the source PDF; every reference was confirmed to correspond to a PDF in `docs/`; and every project-specific figure was recomputed from the code and from `Ablation_Study/results/config_*/final_results.json` using pooled macro F1 (mean of `per_class_f1`), not the mean-of-folds column that §3.1.6 rejects. Bibliographic details that could not be confirmed because the copy in `docs/` is a preprint or author manuscript are listed in this folder's `README.md`.

## Contents

| § | Topic | Ablation variable |
|---|---|---|
| 3.1 | The evaluation corpus: CASME II and the benchmark family | — |
| 3.2 | Motion magnification: Eulerian Video Magnification | A |
| 3.3 | Motion representation: dense optical flow | — (input) |
| 3.4 | Deformation representation: the optical strain tensor | — (input) |
| 3.5 | Temporal normalisation | — (input) |
| 3.6 | The learned spatial backbone: shallow 3D CNNs | C |
| 3.7 | Parameter-free attention: SimAM | B |
| 3.8 | The temporal encoder: transformers | D |
| 3.9 | Training under severe class imbalance | — (constant) |
| 3.10 | Synthesis and research gap | — |

---

## 3.1 The Evaluation Corpus: CASME II and the Spontaneous Micro-Expression Benchmark Family

> **Scope of this section.** This is the first section of the literature review and it is deliberately narrow: it covers the corpus literature only — the papers that introduce, specify, or standardise the evaluation of the database on which every experiment in this thesis is run. Sections 3.2 onwards review the *methodological* literature (motion magnification, optical flow and strain, temporal normalisation, 3D convolutional backbones, attention, and transformers). The reason for treating the corpus first, and at length, is argued in §3.1.8: on a database of this size, several of the design decisions that are usually presented as modelling choices are in fact *forced* by properties of the data, and several results that are usually attributed to architecture turn out on inspection to be attributable to the evaluation protocol.

---

### 3.1.1 Why a dedicated spontaneous corpus was necessary

Automatic facial expression recognition matured on large, well-curated corpora of *posed* expressions — CK+, MUG, MMI, JAFFE and Multi-PIE among them — and by the early 2010s reported accuracies above 90 % on the six basic posed emotions (Yan et al., 2014). Micro-expression research could not follow the same path, for a reason that is intrinsic to the phenomenon rather than incidental to the engineering.

As §2.1 establishes, a micro-expression is involuntary and cannot be *performed* on instruction. Yan et al. (2014) make the consequence sharply in their review of the pre-2013 landscape, which contained only four micro-expression databases, of which only two were spontaneous:

| Corpus | Subjects | Samples | fps | Posed / spontaneous | FACS-coded | Classes |
|---|:--:|:--:|:--:|---|:--:|:--:|
| USF-HD (Shreve et al., 2011) | — | 100 | 30 | Posed | No | 6 |
| Polikovsky et al.† | 10 | 42 | 200 | Posed | No | 6 |
| SMIC (Li et al., 2013) | 16 valid | 164 | 100 | Spontaneous | No | 3 |
| CASME (Yan et al., 2013)† | 19 valid | 195 | 60 | Spontaneous | Yes | 7 |

*Table 3.1 — The micro-expression corpora available prior to CASME II, reproduced from Yan et al. (2014, Table 1). Rows marked † are described there; the primary papers are outside the review corpus and are not cited separately.*

Two problems in this landscape motivated CASME II directly, and both continue to shape the present work.

The first is **construct validity**. The posed corpora asked participants to produce brief facial movements deliberately. Yan et al. (2014) and Li et al. (2013) both note that posed "micro-expressions" differ systematically from spontaneous ones, and Li et al. (2013) further observe that USF-HD adopted a 2/3-second duration criterion that is longer than the definitional upper bound given in §2.1. A system trained on posed data is therefore not demonstrably a micro-expression recogniser at all; it may be a recogniser of fast deliberate movement.

The second is **acquisition quality**. Because the phenomenon is defined jointly by very short duration and very low intensity, it sits close to the noise floor of ordinary video. SMIC was recorded at 100 fps with a facial region of roughly 190 × 230 pixels, and CASME at 60 fps with roughly 150 × 190 pixels. Yan et al. (2014) argue that 60 fps in particular cannot adequately sample a movement that may complete in under 200 ms, and that flickering illumination — a known artefact of high-speed capture under alternating-current lighting — contaminated earlier recordings.

Early attempts to induce genuine micro-expressions through high-stakes deception paradigms (mock-theft scenarios, forced lying about opinions) produced samples heavily contaminated by conversational and other emotion-irrelevant facial movement (Yan et al., 2014). The paradigm that eventually proved workable — and that CASME II adopts — is to have participants watch high-arousal video episodes while attempting to keep the face neutral (Li et al., 2013).

---

### 3.1.2 CASME II: construction, apparatus, and annotation

CASME II (Yan et al., 2014) is the corpus on which every experiment in this thesis is performed. Its construction is documented in enough detail to be worth restating precisely, because several of its specifics propagate directly into the data pipeline described in Chapter 4.

**Participants and elicitation.** Thirty-five participants were recruited, with a mean age of 22.03 years (SD = 1.60). Participants watched emotionally high-valence video episodes, pre-rated by an independent panel of twenty raters on a seven-point Likert scale, and were instructed to suppress all facial movement. Yan et al. (2014) introduce one methodological refinement over both SMIC and CASME here, drawing on Ekman's distinction between *time-reduced full affect* micro-displays, of which the actor is unaware, and *squelched* micro-displays, which the actor senses and interrupts mid-performance. Earlier paradigms targeted only the former; CASME II varied the instruction in order to elicit both, on the hypothesis — explicitly left open by the authors — that the two may have different dynamic signatures.

**Apparatus.** Recording used a Point Grey GRAS-03K2C camera at 640 × 480 resolution in "Raw 8" mode to sustain **200 fps**, saved as MJPEG with no inter-frame compression. Four LED lamps under umbrella reflectors provided steady, high-intensity, flicker-free illumination — a deliberate correction of the artefacts observed in SMIC. The resulting facial region is approximately **280 × 340 pixels** — more than three times the facial area of CASME (150 × 190) and roughly double that of SMIC (190 × 230).

**Selection.** From nearly 3,000 elicited facial movements, a small subset was retained. Very subtle movements were discarded outright, on the grounds that their onset and offset frames could not be reliably located.

**Annotation.** Each retained sample carries onset and offset frame indices, Facial Action Coding System (FACS) action-unit labels assigned by two independent coders with disagreements arbitrated by discussion, and an emotion label. Critically, Yan et al. (2014) argue against forcing spontaneous micro-expressions into the six basic categories used for ordinary expressions, since a single action unit is frequently ambiguous — AU4 (brow lowerer) may indicate disgust, anger, attention or tension. The emotion label is therefore assigned from a *combination* of evidence: the coded AUs, the participant's own self-report, and the emotional content of the eliciting episode.

**Preprocessing in the original release.** The baseline evaluation applies a three-step normalisation that is worth noting because this thesis departs from it. A frontal neutral model face is chosen; 68 Active Shape Model landmarks are detected on the *first frame only* of each clip and registered to the model face by a Local Weighted Mean transform; the resulting transform is then applied unchanged to every frame of that clip, and the face is cropped by eye position. Yan et al. (2014) justify landmarking only the first frame on two grounds — that rigid head motion within so short a window is negligible, and that per-frame ASM landmarks are unstable enough to introduce jitter where the face did not in fact move.

**A note on sample count.** The published paper reports **247** micro-expressions from 26 valid participants. The coding file distributed with the released database contains **255** samples. This discrepancy — a consequence of post-publication revision of the coding — is rarely acknowledged in the literature, and it is one of several reasons why sample counts reported for "CASME II" vary between papers. The present work uses the released 255-sample coding; §3.1.3 traces exactly how those 255 become the 156 clips used here.

---

### 3.1.3 The label taxonomy, and the sparsity problem it creates

Yan et al. (2014) provide five main emotion categories, with labelling criteria expressed in terms of action units:

| Emotion | AU criterion | N (paper) |
|---|---|:--:|
| Happiness | AU6 or AU12 | 33 |
| Disgust | AU9, AU10, or AU4+AU7 | 60 |
| Surprise | AU1+2, AU25, or AU2 | 25 |
| Repression | AU15 or AU17, alone or combined | 27 |
| Others | other emotion-related facial movement | 102 |

*Table 3.2 — Emotion labelling criteria and frequencies, after Yan et al. (2014, Table 3).*

The authors are candid about the resulting imbalance, noting that "some types of facial expressions are difficult to elicit in laboratory situations, thus the samples in different categories distributed unequally."

The released coding file is finer-grained than the paper's five categories, exposing seven raw labels. Counting directly from the label table used in this thesis (`Processed_Data/master_thesis_labels.csv`, 255 rows) gives:

| Raw label | N |
|---|:--:|
| others | 99 |
| disgust | 63 |
| happiness | 32 |
| repression | 27 |
| surprise | 25 |
| sadness | 7 |
| fear | 2 |
| **Total** | **255** |

*Table 3.3 — Raw emotion labels in the released CASME II coding, as parsed for this thesis.*

**Seven-class classification is not viable on this corpus.** Fear is represented by two clips in the entire database and sadness by seven. Under any subject-disjoint protocol — and §3.1.4 shows that subject-disjointness has been the norm for this benchmark since its introduction — a two-clip class cannot be meaningfully trained or tested: a single fold either contains the class or does not, and per-class metrics degenerate.

The field's response has been to group by affective valence. The convention originates with SMIC, which was released with exactly three classes — positive, negative, surprise (Li et al., 2013) — and it was formalised for cross-corpus work by the Second Facial Micro-Expression Grand Challenge (MEGC 2019; See et al., 2019), which defines:

- **Negative** — Repression, Anger, Contempt, Disgust, Fear, Sadness
- **Positive** — Happiness
- **Surprise** — Surprise

Applying that mapping to the seven raw CASME II labels, and discarding the 99 unusable "others" clips, yields the working set used throughout this thesis:

| Grouped class | Composition | N |
|---|---|:--:|
| Negative | disgust (63) + repression (27) + sadness (7) + fear (2) | **99** |
| Positive | happiness (32) | **32** |
| Surprise | surprise (25) | **25** |
| **Total** | | **156** |

*Table 3.4 — The three-class working set derived from CASME II for this thesis.*

**An important divergence from the MEGC convention, declared explicitly.** MEGC 2019 reports the CASME II portion of its composite database as **145** samples, not 156, and footnotes that "Negative class of CASME II: Disgust and Repression" (See et al., 2019, Table III) — that is, the challenge builds its CASME II subset from the paper's *five-category* release, in which sadness and fear do not appear as separate labels and are absorbed into "others". This thesis instead uses the released seven-label coding and, following MEGC's own class definition, assigns sadness and fear to Negative. The consequence is an **eleven-clip** difference: N = 156 here against N = 145 in the challenge protocol, with a Negative class of 99 rather than 88. Nine of those eleven are the sadness (7) and fear (2) clips that the seven-label coding exposes and the five-category release absorbs into "others". The remaining two are unaccounted for by that mapping alone: disgust and repression together number 90 in the released coding used here, against the 88 the challenge reports, so the challenge's CASME II subset is two clips smaller than the same categories in this coding. That residual is noted rather than resolved; it is a further consequence of the version ambiguity described in §3.1.9.

The divergence is small but must be stated when comparing against published numbers, and it has a mild direction: the nine clips this coding adds are all in the majority class and all drawn from categories the original coders found hardest to assign, so if anything they make the present task marginally harder than the challenge's, not easier.

The residual imbalance is severe in either case — 99 : 32 : 25, a ratio of roughly 4 : 1.3 : 1. Everything in §3.1.8 concerning metric choice, sampling, and loss follows from this single table.

---

### 3.1.4 The original baseline, and what it does and does not establish

Yan et al. (2014) provide a baseline using Local Binary Patterns from Three Orthogonal Planes (LBP-TOP; Zhao & Pietikäinen, 2007) for feature extraction and a Support Vector Machine for classification. LBP-TOP extends the static LBP operator by extracting codes on the XY, XT and YT planes of the video volume and concatenating the three histograms, so that the XT and YT planes encode how pixel intensities along a row or column evolve over time — dynamic texture rather than static texture.

The baseline configuration divides the face into 5 × 5 blocks, sets the number of neighbouring points P = 4, and sweeps the spatial radii $R_X, R_Y \in \{1,2,3,4\}$ and the temporal radius $R_T \in \{2,3,4\}$. The temporal radius $R_T = 1$ is deliberately excluded on the grounds that at 200 fps the change between two adjacent frames is negligible — an early acknowledgement that a high sampling rate is not automatically an advantage. Reported five-class accuracies under **leave-one-subject-out cross-validation** cluster in a narrow band in the high fifties to low sixties, with the best configuration ($R_X = R_Y = 1$, $R_T = 4$) reaching **63.41 %**.

Three observations from this baseline matter for the present study.

**First, leave-one-subject-out is native to this benchmark.** It is not a stricter protocol adopted later by cautious authors; it is the protocol used in the paper that released the corpus. Any evaluation of CASME II that reports a single random train/test split is departing from the benchmark's own convention, and §3.1.7 and Chapter 5 quantify what that departure costs.

**Second, the baseline is reported as accuracy alone.** No per-class or macro-averaged score is given. On a five-class problem in which "others" accounts for 102 of 247 samples, accuracy is a weak instrument: the majority-class-constant classifier scores 41 % without modelling anything. In the three-class working set of this thesis the equivalent floor is higher still — always predicting Negative yields **63.46 % accuracy** while achieving a macro F1 of only **0.2588**. A headline accuracy near the original 63.41 % baseline is therefore not, on its own, evidence of a working system.

**A caution on this figure.** Yan et al. (2014) state plainly that leave-one-subject-out cross-validation was applied to obtain the 63.41 % result. Li et al. (2018), reproducing the same figure in their comparison table, mark it with an asterisk denoting "results achieved using leave-one-sample-out cross validation" — a materially easier protocol. The two primary sources contradict each other on the protocol behind the most widely quoted number on this benchmark. This thesis follows the originating paper's own account, but the discrepancy is recorded here because it affects how the 63.41 % figure should be read in any comparison.

**Third, the sensitivity of the baseline to a purely descriptive hyper-parameter is itself informative.** Accuracy moves by several points across the radius sweep with no change to the data, the split, or the classifier. Where effect sizes of that magnitude arise from descriptor settings, differences of similar magnitude between architectures cannot be interpreted without a variance estimate — a limitation this thesis inherits and declares in Chapter 6.

---

### 3.1.5 Sibling corpora, and why this study remains single-corpus

Three further databases define the context in which CASME II is used, and each is included in this review because it isolates a specific limitation of CASME II.

**SMIC (Li et al., 2013)** — the first spontaneous micro-expression database, and the origin of the three-class taxonomy. Twenty participants were recorded; sixteen produced usable micro-expressions, yielding **164 clips** in the high-speed (HS) subset at **100 fps**, distributed as 51 positive, 70 negative and 43 surprise. SMIC uniquely provides synchronised normal-visual (VIS) and near-infrared (NIR) recordings at 25 fps alongside the high-speed stream. Its baseline — LBP-TOP with a temporal interpolation model (TIM) normalising each clip to ten frames, evaluated under leave-one-subject-out — reaches **48.78 %** on the three-class HS task. Two elements of SMIC's design are directly inherited by this thesis: the three-class valence grouping, and the use of temporal interpolation to normalise clips of unequal length before feature extraction. SMIC is, however, not FACS-coded and provides no apex annotation, which is why it is not used here.

**SAMM (Davison et al., 2018)** — constructed explicitly to address the demographic narrowness of the CASME family. It comprises **159 micro-movements from 32 participants** at **200 fps** and **2040 × 1088** resolution (facial region approximately 400 × 400), FACS-coded, across seven emotion classes. Its decisive contribution is diversity: **13 ethnicities** and a mean participant age of **33.24 years (SD = 11.32)**, against CASME II's single ethnicity and mean age of 22.03 (SD = 1.60). Davison et al. (2018) are direct about the problem this addresses, observing that a corpus drawn from one ethnicity and one narrow age band restricts analysis to "similar looking participants". They also tailor stimulus selection to each participant in advance rather than relying on post-hoc self-report. The design is stronger than CASME II's on both counts; the corpus is nevertheless smaller in usable micro-expression samples, and its emotion labels are distributed across seven categories with the same sparsity difficulty.

**CAS(ME)² (Qu et al., 2016)** — extends the CASME family to the macro/micro relationship, providing **250 spontaneous macro-expression and 53 micro-expression samples elicited from the same participants**, recorded with a Logitech Pro C920 at **30 fps** and 640 × 480, and labelled in four categories (positive, negative, surprise, other) using the same combination of AUs, per-movement self-report, and stimulus emotion type. Its value is that it enables study of macro-to-micro domain transfer — the strategy behind several strong MEGC entries — because both expression types come from identical subjects under identical conditions. Two properties limit its use here: the 30 fps sampling rate is an order of magnitude below CASME II's and is poorly matched to sub-200-ms movements, and only 53 micro-expression samples are available. It is also worth noting that the CAS(ME)² baseline is evaluated leave-*one-video*-out rather than leave-one-subject-out, which does not enforce subject-disjointness and is therefore not comparable to the protocol used in this thesis.

| | SMIC (HS) | CASME | **CASME II** | SAMM | CAS(ME)² |
|---|:--:|:--:|:--:|:--:|:--:|
| Micro samples | 164 | 195 | **247 / 255** | 159 | 53 (+250 macro) |
| Participants | 16 valid | 19 valid | **26 valid** | 32 | 22 |
| Frame rate | 100 fps | 60 fps | **200 fps** | 200 fps | 30 fps |
| Facial resolution | 190 × 230 | 150 × 190 | **280 × 340** | 400 × 400 | — |
| FACS-coded | No | Yes | **Yes** | Yes | Yes |
| Emotion classes | 3 | 7 | **5 (7 raw)** | 7 | 4 |
| Ethnicities | 3 | 1 | **1** | 13 | 1 |
| Mean age (SD) | 26.7 | 22.03 (1.60) | **22.03 (1.60)** | 33.24 (11.32) | — |

*Table 3.5 — The spontaneous micro-expression corpora, compiled from Yan et al. (2014, Table 1), Davison et al. (2018, Table 1), Li et al. (2013) and Qu et al. (2016).*

**Why this thesis uses CASME II alone.** Three properties make it the appropriate single choice for an ablation study of temporal and motion modelling. It has the highest temporal resolution of any spontaneous corpus of comparable size, which is a precondition for motion magnification in a defined frequency band and for dense frame-to-frame flow. It is FACS-coded with onset, apex and offset annotation, which makes temporal normalisation to a fixed sequence length well-defined rather than heuristic. And it remains the most widely reported single-corpus benchmark in the field, which makes the comparison in Chapter 5 meaningful. The corresponding cost — that no claim in this thesis extends to other populations, other capture conditions, or other corpora — is stated as a limitation in Chapter 6 rather than argued away here.

---

### 3.1.6 How CASME II is evaluated in the field: MEGC 2019 and the metric convention

The most consequential development in how CASME II is *used*, as opposed to how it was built, is the Composite Database Evaluation (CDE) protocol of the Second Facial Micro-Expression Grand Challenge (See et al., 2019). Because this thesis adopts the challenge's metric definitions while deliberately declining its composite training regime, the protocol warrants precise description.

CDE merges SMIC, CASME II and SAMM into a single composite database under the three-class valence grouping of §3.1.3, giving **442 samples** — Negative 250, Positive 109, Surprise 83 — of which CASME II contributes 145. Leave-one-subject-out cross-validation is run across the composite, which the organisers note "ensures subject-independent evaluation", and results are reported both overall and per constituent database.

The organisers reject plain accuracy explicitly on the grounds of class imbalance and mandate two balanced metrics:

**Unweighted F1 (UF1)**, which they describe as the macro-averaged F1-score. It is computed by *accumulating* true positives, false positives and false negatives for each class **over all k folds of the LOSO run**, computing a per-class F1 from those pooled counts, and averaging the per-class scores without weighting:

$$F1_c = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}, \qquad \text{UF1} = \frac{1}{C}\sum_c F1_c$$

**Unweighted Average Recall (UAR)**, the balanced accuracy, obtained by averaging per-class recall.

UF1 is a **pooled** quantity: counts are accumulated across folds *before* any F1 is computed, which §2.6.3 explains is a different estimator from the average of per-fold macro F1 scores. That distinction is not cosmetic on CASME II. Because the corpus assigns very unequal numbers of clips to subjects, and because many single-subject folds contain only one of the three classes, a per-fold macro F1 is bounded well below 1 by the fold composition alone, and its average across folds is bounded correspondingly. The quantity this thesis reports as its primary metric, *pooled macro F1*, is therefore identical in construction to MEGC's UF1, and the per-fold average that appears in the raw experimental logs is not a metric the field uses at all. Chapter 5 quantifies the resulting ceiling for the fold structure of this study.

Published CDE results give the most current picture of performance on the CASME II subset:

| Method | Full composite UF1 / UAR | CASME II subset UF1 / UAR |
|---|:--:|:--:|
| LBP-TOP (Zhao & Pietikäinen, 2007) | 0.5882 / 0.5785 | 0.7026 / 0.7429 |
| Bi-WOOF (Liong et al., 2018) | 0.6296 / 0.6227 | 0.7805 / 0.8026 |
| OFF-ApexNet (Liong et al., 2019a) | 0.7196 / 0.7096 | 0.8764 / 0.8681 |
| Quang et al.† | 0.6520 / 0.6506 | 0.7068 / 0.7018 |
| Zhou et al.† | 0.7322 / 0.7278 | 0.8621 / 0.8560 |
| Liong et al. (2019b), STSTNet | 0.7353 / 0.7605 | 0.8382 / 0.8686 |
| Liu et al. (2019), EMR | **0.7885 / 0.7824** | 0.8293 / 0.8209 |

*Table 3.6 — MEGC 2019 CDE results, reproduced from See et al. (2019, Table IV). † outside the review corpus; reported here only as it appears in that table.*

**These numbers must be read with one structural caveat, and it is a large one.** Every row of Table 3.6 is produced by a model trained on the *composite* database — roughly 442 samples spanning three corpora and 68 subjects — and then scored on the CASME II portion of the pooled predictions. The CASME II column is therefore not a CASME II-only result. It reports how well a cross-corpus model performs on CASME II clips, with approximately **2.8 times the training data** available in any single-corpus fold, and with the additional regularising effect of training across three capture conditions. A single-corpus LOSO experiment of the kind conducted in this thesis trains each fold on roughly 150 clips from one corpus. The two regimes are not interchangeable, and single-corpus results should not be compared directly against Table 3.6 without stating the difference.

Two further observations from the challenge are relevant to this thesis's design. The organisers note that the top three submissions all chose **optical flow** as their feature representation — a convergence that supports the motion-based input used here. They also note that one submission substituted mid-position frames for annotated apex frames with little loss, which suggests that precise apex localisation may be less critical than the field had assumed, and that methods consuming the full onset-to-offset sequence rather than a single apex pair are not obviously disadvantaged.

---

### 3.1.7 Threats to validity carried by the corpus

The properties below are not criticisms of CASME II, which is a carefully constructed resource and remains the best available for this task. They are constraints that any study built on it inherits, and each one is answered by a specific decision in Chapter 4. The fold statistics quoted are computed directly from the label table used in this thesis.

**(a) Scale.** 156 usable clips across three classes. This is small enough that the number of learnable parameters is itself a risk factor, and it places a hard practical ceiling on the depth of any model trained from scratch.

**(b) Class imbalance.** 99 : 32 : 25. The majority-class-constant classifier attains 0.6346 accuracy. Any accuracy figure must be quoted against that floor.

**(c) Extreme subject-level non-uniformity.** Clips are distributed across subjects far more unevenly than is generally acknowledged. Of the 156 clips, **subject 17 alone contributes 33 — 21 % of the entire working set** — while subjects 8, 10 and 21 contribute a single clip each. Under leave-one-subject-out this has two consequences: fold sizes vary by a factor of 33, so any metric that averages across folds implicitly weights a one-clip subject equally with a thirty-three-clip subject; and **10 of the 25 folds contain clips from only one class**, with 8 containing two classes and only 7 containing all three. The single-class folds are what make per-fold macro F1 unusable, as §3.1.6 anticipates.

**(d) A subject that disappears under grouping.** CASME II contains 26 participants, but subject 18's three clips are all labelled "others" and are removed by the three-class filter. The LOSO run therefore has **25 folds, not 26** — a detail that must be stated for reproducibility, since fold counts differ between papers using this corpus.

**(e) Demographic homogeneity.** All participants are Chinese, with a mean age of 22.03 years and a standard deviation of 1.60 — that is, essentially a single ethnicity and a single narrow age band. Davison et al. (2018) identify this as a principal motivation for SAMM. Nothing established on CASME II can be assumed to generalise across appearance, age, or cultural display rules, and the MEGC organisers observed precisely this effect in reverse: domain adaptation from CK+ helped on SMIC and SAMM but "not so for CASME II, which contain predominantly Chinese subjects" (See et al., 2019).

**(f) Laboratory ecology.** Recording is frontal, under four fixed high-intensity lamps chosen to eliminate flicker, with participants instructed not to move and not speaking. Yan et al. (2014) themselves flag the extension to "natural conversation and interaction" as future work. The corpus therefore contains almost none of the head motion, illumination variation, occlusion or speech-related facial movement that a deployed system would face.

**(g) Annotation dependence.** Emotion labels derive from a combination of coded AUs, participant self-report, and the emotional content of the eliciting stimulus. This is a more defensible procedure than forcing six basic categories onto the data, and Yan et al. (2014) argue for it convincingly — but it means the ground truth is a construct produced by trained coders under a documented rubric, not an objectively measurable quantity. Inter-coder disagreements were arbitrated by discussion rather than reported as residual uncertainty.

**(h) Duration heterogeneity, including samples beyond the definitional bound.** Onset-to-offset lengths in the working set range from **31 to 126 frames**, with a median of 66. At 200 fps this is **0.155 s to 0.63 s**, median 0.33 s. The upper end exceeds the 1/2-second criterion given in §2.1 as the standard definitional bound. Any model consuming full sequences must therefore handle a four-fold variation in length, part of which sits outside the phenomenon's nominal definition.

**(i) Version ambiguity.** As noted in §3.1.2 and §3.1.3, the literature contains at least three different sample counts for "CASME II" — 247 (the paper), 255 (the released coding), and 145 (the MEGC subset) — and both five-class and seven-class label sets are in circulation. Papers rarely state which they used. This is a genuine obstacle to comparison and is one reason Chapter 5 reports N explicitly alongside every figure.

---

### 3.1.8 Implications for this research

This section makes explicit the claim advanced at the head of §3.1: on a corpus of this size and shape, several decisions normally presented as modelling choices are determined by the data. Table 3.7 maps each corpus property onto the design decision it forces and the chapter in which that decision is implemented.

| Corpus property (§) | Design decision taken in this thesis | Where |
|---|---|---|
| Seven raw labels with fear = 2, sadness = 7 (3.1.3) | Three-class valence grouping, following SMIC and the MEGC 2019 definition; "others" discarded; N = 156 declared with its 9-clip divergence from the MEGC subset | Ch. 4 |
| 26 subjects, single ethnicity, narrow age band (3.1.2, 3.1.7e) | Input is a **motion representation** — dense optical flow (u, v) plus optical strain — rather than raw pixels, since motion is invariant to identity and illumination whereas appearance is exactly what a 26-subject corpus would let a network memorise | Ch. 4 |
| 200 fps capture (3.1.2) | Eulerian magnification applied in a **5–25 Hz temporal band** matched to the recording rate, prior to flow computation | Ch. 4 |
| Onset-to-offset length varies 31–126 frames (3.1.7h) | Every clip temporally resampled to a fixed **T = 32** frames, following the TIM precedent established by SMIC's baseline (Li et al., 2013) | Ch. 4 |
| LOSO native to the benchmark (3.1.4); subject 17 = 21 % of clips (3.1.7c) | **Full 25-fold leave-one-subject-out**, every subject held out exactly once, no clip untested, subject-disjointness guaranteed by construction rather than by bookkeeping | Ch. 4 |
| 10 of 25 folds are single-class (3.1.7c) | Primary metric is **pooled macro F1**, computed by accumulating TP/FP/FN across all folds before averaging per-class F1 — identical in construction to MEGC's UF1 (See et al., 2019). Per-fold averaged macro F1 is explicitly rejected, since fold composition bounds it below the study's own target | Ch. 4, Ch. 5 |
| Majority class 99/156 (3.1.3) | Accuracy is never reported alone; the always-Negative reference (0.6346 accuracy, 0.2588 macro F1) is quoted alongside every result as a floor | Ch. 5 |
| Severe imbalance (3.1.3) | Minority oversampling via a balanced sampler, with focal loss; inverse-frequency class weighting in the loss deliberately **disabled**, since applying both corrections simultaneously induced single-class collapse | Ch. 4, Ch. 5 |
| 156 clips total (3.1.7a) | Preference for **parameter-free or low-parameter components** wherever a choice exists — motivating the use of SimAM (Yang et al., 2021), which adds zero learnable parameters — and for shallow rather than deep spatial backbones, following the model-shrinking argument of Xia et al. (2020b) | Ch. 4 |
| Single corpus, laboratory conditions (3.1.7e, 3.1.7f) | No cross-dataset or in-the-wild claim is made anywhere in this thesis; generalisation is listed as future work | Ch. 6 |
| Baseline hyper-parameter sensitivity (3.1.4); N = 156 (3.1.7a) | Effect sizes below the resolution of the corpus are reported as unresolved rather than as findings; the 95 % confidence interval at N = 156 is stated wherever two configurations are compared | Ch. 5, Ch. 6 |

*Table 3.7 — Corpus properties and the design decisions they determine.*

Three consequences deserve to be drawn out, because they shape the interpretation of Chapter 5's results rather than merely the construction of the pipeline.

**First, the corpus determines the metric, and the metric determines the conclusion.** Because 10 of the 25 folds contain a single class, a macro F1 averaged over folds is capped by fold composition alone at a value below this study's own target — no classifier, however good, can reach the target on that quantity. It is not a defect of any model; it is arithmetic imposed by how CASME II distributes clips across subjects. Reporting the pooled quantity instead is not a stylistic preference but the correction that MEGC 2019 already standardised, and Chapter 5 shows that the two quantities rank configurations differently.

**Second, the corpus is small enough that the evaluation protocol, not the architecture, dominates the variance.** This is the most significant methodological finding of the present work, and it is anticipated entirely by the properties reviewed above. With 25 subjects distributed as unevenly as §3.1.7(c) describes, a single held-out split is not a measuring instrument: which subjects fall in the test set can move the result by more than any architectural change under study. Chapter 5 demonstrates this empirically by evaluating one fixed set of twelve configurations under five different protocols and observing that the ranking inverts.

**Third, the small-sample regime predicts which components should help.** A corpus of 156 clips cannot supply the data needed to learn a general spatio-temporal feature extractor from scratch, particularly when the input has already been reduced to optical flow and strain — that is, when much of the spatio-temporal feature extraction has been performed analytically before the network sees the data. Conversely, a component that models the temporal arc of an expression — onset, apex, relaxation — is exploiting exactly the structure that the corpus's onset/apex/offset annotation guarantees is present and correctly delimited. Chapter 5 tests this prediction directly through a full component ablation.

---

### 3.1.9 Limitations of the existing corpus literature, and how this work differs

*(This paragraph fulfils the requirement that the literature review close by stating the gap; a corresponding paragraph appears at the end of each subsequent section of this chapter.)*

The corpus literature reviewed above establishes three things well and one thing poorly. It establishes the phenomenon's definitional boundaries, an elicitation methodology that produces genuinely spontaneous samples, and a capture standard — 200 fps, flicker-free illumination, FACS coding with onset/apex/offset annotation — that makes fine-grained motion analysis feasible. What it does not establish is a stable basis for comparing methods. The original CASME II baseline reports accuracy only, on a five-class problem whose majority class is 41 % of the data, and its own hyper-parameter sweep moves the headline figure by several points with no change to the model. Sample counts, label granularity and class groupings differ across papers using the same database — 247, 255 and 145 all appear in the literature under the name "CASME II" — and are frequently not stated. The community's strongest standardisation effort, the MEGC 2019 CDE protocol, fixed the metric problem convincingly by mandating pooled macro F1 and balanced recall, but did so within a composite training regime that makes its per-database columns non-comparable to single-corpus work, a distinction that subsequent papers routinely elide. Above all, no study in this group quantifies how much of a reported difference between methods is attributable to the evaluation protocol rather than to the methods themselves, despite the fact that the corpus's subject distribution — one subject supplying a fifth of all clips, ten of twenty-five folds containing a single class — makes that question unavoidable.

This thesis differs in three respects. It runs the **complete** 25-fold leave-one-subject-out protocol over **every** valid cell of a four-component ablation matrix rather than reporting a single proposed configuration, so that each component's contribution is measured from matched pairs differing in exactly one factor. It adopts MEGC's pooled macro F1 as its primary metric while explicitly declining the composite training regime, so that its numbers describe single-corpus performance and are labelled as such. And it treats the evaluation protocol as an experimental variable in its own right, re-evaluating one fixed set of configurations under five successive protocols and reporting the resulting instability as a finding rather than suppressing it. The intended contribution is therefore not only a recognition result on CASME II, but a quantified account of how much of that result — and of results already in the literature — is attributable to how the corpus is split rather than to what is computed on it.

---

---

## 3.2 Motion Magnification: Eulerian Video Magnification as a Pre-Processing Stage

> **Scope of this section.** This section reviews Eulerian Video Magnification (EVM) through the four review-corpus papers that apply it — Bai et al. (2021), Li et al. (2018), and Li, Huang and Zhao (2018, 2021). The original formulations it builds on, Wu et al.'s amplitude-based method and Wadhwa et al.'s phase-based method, lie outside the corpus and are described here only as attributed through those four papers.

---

### 3.2.1 The problem magnification addresses

The corpus design (§3.1.1) responds to the *short-duration* half of the micro-expression definition but not to the *low-intensity* half: CASME II's 200 fps sampling represents a movement completing in under 200 ms across dozens of frames (Yan et al., 2014), yet a micro-expression may still occupy only a few grey levels of change across a small patch of skin, and Yan et al. (2014) report discarding samples outright because the movement was too faint for coders to place an onset or offset.

Li et al. (2018) state the difficulty directly: "the intensity levels of facial movements are too low to be distinguishable." Their proposed remedy is to amplify the motion itself before any descriptor is computed, so that the *difference between expression categories is enlarged* rather than the classifier being made more sensitive. This is a data-level intervention: the model is untouched, and the modification is entirely upstream. That property is what makes EVM cleanly separable as an ablation factor in this thesis, and it is why §3.1.8 and Chapter 4 treat it as part of the data pipeline — the baseline — rather than as a network component.

---

### 3.2.2 How Eulerian magnification works

EVM is *Eulerian* in the fluid-dynamics sense: rather than tracking features as they move (a Lagrangian approach, as optical flow does), it fixes attention on each spatial location and amplifies the temporal variation observed *at that location*. Motion is amplified as a side effect of amplifying intensity change, without any explicit correspondence being computed.

Bai et al. (2021) give the clearest account in the review corpus of the two variants and their difference.

**Amplitude-based magnification (AMM)** proceeds in four steps. Each frame is decomposed into spatial frequency bands by a full **Laplacian pyramid**; a **Butterworth temporal band-pass filter** extracts the frequency range of interest at each band; the band-passed signal is multiplied by a **magnification factor α**; and the amplified signal is added back to the original image. This is the formulation implemented in this thesis.

**Phase-based magnification (PMM)** decomposes each frame with **octave complex steerable pyramids** over four orientations, and applies the temporal filter to the *local phase* rather than the amplitude. Bai et al. (2021) explain the advantage in signal terms: the steerable pyramid "has impulse response with spatial support," which makes it easier to isolate the intended temporal frequencies while suppressing the remainder. The practical consequences they report are that the phase-based method **supports larger amplification factors** and is **notably less sensitive to noise**.

Three parameters are therefore free in any EVM application, and all three recur below:

1. the **temporal pass-band**, which decides *which* motions are amplified;
2. the **magnification factor α**, which decides *how much*;
3. the **pyramid depth**, which decides at which spatial scales amplification occurs.

**Setting the temporal band.** Bai et al. (2021) report the rule given by the original EVM authors: the pass-band should correspond to the duration of the phenomenon being amplified — for micro-expressions, 1/5 s to 1/25 s, equivalently a band of **5 Hz to 25 Hz**. They also report that applying this full band produced "noticeable noise" in their setting, and that they narrowed it to 1/15–1/25 s (15–25 Hz) accordingly. This is the only explicit guidance on band selection anywhere in the review corpus, and it is the basis for the band used in this thesis (§3.2.7).

---

### 3.2.3 Evidence that magnification improves recognition

The strongest empirical case in the review corpus is Li et al. (2018), who sweep the magnification factor systematically rather than fixing it. Clips are magnified at what the paper describes as ten levels, listing α ∈ {1, 2, 4, 8, 12, 16, 20, 24, 30} — nine values, an inconsistency in the source rather than in this account — where α = 1 denotes no magnification; each magnified clip is then temporally interpolated to ten frames (TIM10), and LBP, HOG and HIGO descriptors are extracted and classified. The experiment is run on both SMIC and CASME II under leave-one-subject-out.

Two findings emerge, and both are directly relevant here.

**Magnification helps, consistently.** The improvement holds "for all three kinds of features on all four testing datasets." On CASME II the effect is substantial:

| Descriptor | No magnification | With magnification | Δ |
|---|:--:|:--:|:--:|
| LBP | 55.87 % | 60.73 % | **+4.86** |
| HOG | 57.49 % | 63.97 % | **+6.48** |
| HIGO | 57.09 % | 67.21 % | **+10.12** |

*Table 3.8 — Effect of Eulerian motion magnification on CASME II under leave-one-subject-out, after Li et al. (2018, Table 6).*

The best configuration of their full framework — magnification, TIM10 and HIGO — reaches 67.21 % on CASME II and 68.29 % on SMIC-HS under LOSO, which they show to be competitive with or better than the contemporaneous state of the art. Li et al. (2018) flag a problem in that comparison: several published CASME II figures, including the 63.41 % baseline of Yan et al. (2014), were obtained under **leave-one-sample-out** validation, "which is much easier" than the leave-one-subject-out protocol used here (§3.1.4).

**The relationship between α and accuracy is non-monotonic.** Li et al. (2018) describe the resulting curves as "rainbow-shaped," with best performance "generally achieved when the motion is magnified in the range of **[8, 16]**." Their explanation of the two failure modes is mechanistic: "magnification at lower levels might not be enough to reveal the ME motion progress; on the other hand, magnification at higher levels degrades the performance because too many artifacts are induced." Magnification is therefore not a monotone improvement to be maximised, but a parameter with an interior optimum that must be selected.

Bai et al. (2021) provide the second quantitative demonstration, using a fundamentally different downstream model — a pre-trained VGGFace2 (ResNet-50) spatial encoder feeding a bidirectional LSTM. On SMIC they report **60.60 % without magnification against 75.76 % with phase-based magnification**, and the phase-based variant outperforms the amplitude-based one. Because the pipeline shares nothing with Li et al.'s hand-crafted descriptors, the two results together establish that the benefit is not an artefact of a particular feature family — at least when the downstream representation is appearance-based.

---

### 3.2.4 Magnification in apex-frame pipelines, and the α discrepancy

Two further papers in the review corpus use EVM in a different regime, and the contrast is informative.

Li, Huang and Zhao (2018) address whether a micro-expression can be recognised from a *single* apex frame. Their argument for magnification is specific to that setting: the apex frame carries the most emotional information, but "the intensity level of apex frame is not distinguished due to subtle change, especially for deep model that describes high-layer abstract information." They magnify the spotted apex frame using EVM, set **α = 30**, and fine-tune a VGG-Face descriptor on the result. Because a single frame per clip is too little data to train on, they exploit the high frame rate to augment: the apex frame plus its two preceding and two following frames are all retained, expanding the training set fivefold. They restate the same trade-off Li et al. (2018) quantified — "bigger values of motion amplification level lead to larger scale of motion amplification, but also can cause bigger displacement and artifacts."

Li, Huang and Zhao (2021) extend this into a joint local–global learning framework with frequency-domain apex detection, and carry the same setting forward: motion magnification level "set as 30 in our framework," with the same ±2-frame augmentation producing what they name the Extended Magnified ME (EMME) database.

**The α discrepancy is worth dwelling on**, because it is easy to misread as a contradiction. Li et al. (2018) find the optimum at α ∈ [8, 16] and observe degradation beyond it; Li, Huang and Zhao (2018, 2021) adopt α = 30, which sits at the extreme end of that same sweep. The regimes differ in a way that plausibly explains the gap. The [8, 16] optimum is measured on **full magnified sequences** described by spatio-temporal descriptors, where inter-frame artefacts accumulate along the temporal axis and corrupt exactly the dynamic texture the descriptor encodes. The α = 30 setting is applied to a **narrow five-frame window around the apex**, consumed by a deep face encoder as largely static appearance, where the temporal artefacts that penalise a spatio-temporal descriptor have far less opportunity to express themselves — and where the encoder needs a strong appearance signal to distinguish categories at all.

The implication for any new system is that α cannot be transferred between pipelines. It is a parameter of the *combination* of magnification and downstream representation, not of the corpus.

---

### 3.2.5 Known limitations of magnification

The review corpus is unusually candid about EVM's failure modes, and four are documented.

**It amplifies everything in band, not only expressions.** This is the most important limitation and the most easily overlooked. Li et al. (2018) decline to use magnification for micro-expression *spotting* at all, "because it magnifies unwanted motions (e.g. head movements) at the same time." EVM has no notion of which motion is of interest; a rigid head translation whose temporal frequency falls inside the pass-band is amplified exactly as a brow raise is. In a recognition pipeline this is mitigated by prior face alignment and cropping — Li et al. (2018) magnify only after normalising and cropping to the eye-defined rectangle — but it is mitigated, not eliminated.

**Artefacts grow with α.** All four papers report this. Li et al. (2018) show it as the descending limb of the rainbow curve; Li, Huang and Zhao (2018, 2021) state it as a design caution; Bai et al. (2021) observe it directly as noise that forced them to narrow their temporal band.

**There is no principled procedure for choosing α.** Every value in the corpus is obtained empirically — by exhaustive sweep (Li et al., 2018) or by citation of a prior sweep (Li, Huang & Zhao, 2018, 2021). No paper offers a way to select α from properties of the data.

**Evaluation is confined to appearance-based downstream representations.** LBP, HOG and HIGO in Li et al. (2018); VGGFace2 features in Bai et al. (2021); VGG-Face in both Li, Huang and Zhao papers. Every one of these descriptors is computed on **magnified pixel intensities**. No study in the review corpus measures what magnification contributes when the representation handed to the classifier is itself an explicit motion field — an optical flow and strain volume — rather than an appearance signal. §3.2.8 argues that this is the gap this thesis occupies, and §3.2.6 explains why the distinction is likely to matter.

---

### 3.2.6 Why magnification may behave differently in front of a motion representation

This subsection sets out a hypothesis, not a review finding, because the review corpus does not test it. It is stated here because it frames the interpretation of Chapter 5's EVM result.

EVM amplifies the *amplitude* of intensity variation in a temporal band. An appearance descriptor computed on the magnified frames sees a directly stronger signal: LBP codes flip that previously did not, gradient orientations become better determined, and a deep encoder's activations move further from the decision boundary. The mechanism by which magnification helps is straightforward.

An optical flow field is a different object. Flow estimates *displacement*, and amplifying the underlying intensity change by α does not simply scale the recovered displacement by α — it changes the conditioning of the estimation problem, potentially making faint motion recoverable where it previously was not, while simultaneously introducing displacement artefacts of the kind all four reviewed papers warn about. Any residual global scaling of the flow magnitude is then further attenuated by the per-channel normalisation that a learned pipeline applies before training (Chapter 4). The plausible outcome is that magnification in front of a flow-and-strain representation retains the part of its benefit that consists of *making otherwise-unrecoverable motion recoverable*, while losing the part that consists of *increasing contrast for an appearance descriptor*.

If that is correct, EVM's measured contribution should be (i) smaller on average than the +5 to +10 percentage points reported in Table 3.8, and (ii) concentrated in configurations where the downstream model is capable of exploiting finer spatial deformation. Chapter 5 reports exactly this pattern, and §3.2.7 records the numbers.

---

### 3.2.7 Implications for this research

The review above determines this thesis's magnification settings and the way EVM is positioned within the experimental design.

| Finding from the review | Decision taken in this thesis |
|---|---|
| Pass-band should match the phenomenon's duration, 1/5–1/25 s ≡ 5–25 Hz (reported by Bai et al., 2021) | Temporal band-pass set to **5–25 Hz**, matched to CASME II's 200 fps sampling |
| Accuracy against α is rainbow-shaped, optimum in **[8, 16]** for full-sequence pipelines (Li et al., 2018) | **α = 10**, inside the reported optimum; the α = 30 of the apex-frame papers is explicitly *not* adopted, since this thesis consumes full sequences |
| Amplitude-based magnification uses a Laplacian pyramid plus temporal band-pass (Bai et al., 2021) | Amplitude-based EVM over a **4-level Laplacian pyramid**, temporal filtering by FFT band-pass |
| Magnification amplifies head motion as well as expression (Li et al., 2018) | Magnification applied only after the clip has been trimmed onset-to-offset; the risk is acknowledged as a residual threat rather than claimed to be eliminated |
| Magnification is a purely upstream, data-level operation | Stage 1 is run **twice**, producing a magnified and a non-magnified tensor set; the EVM switch selects a directory and leaves the network unchanged, making it a clean ablation factor |
| Magnification is part of the proposed data pipeline, not of the network | EVM is treated as the **baseline condition** (`config_4`), not as a component to be added; the non-magnified arm (`config_1`) is the control that isolates it — see §3.1.8 |

*Table 3.9 — Review findings and the magnification settings they determine.*

**What the measurement gives.** Under the 25-fold LOSO protocol (§3.1.4), EVM is measured across six matched pairs of configurations that differ in nothing but the tensor directory. Its mean effect is **+0.015 macro F1** — an order of magnitude smaller than the +0.05 to +0.10 accuracy gains reported in Table 3.8 for appearance descriptors, and small enough to sit below this study's resolution. The distribution behind that mean is more informative than the mean itself: four of the six pairs are positive and two are negative, and the two largest gains (**+0.080** and **+0.049**) both occur in configurations that contain a 3D convolutional spatial stem, while the largest loss occurs in the configuration with no spatial stem at all. The single clearest instance is a configuration whose Surprise-class F1 rises from **0.300 to 0.655** when magnification is enabled. This is the pattern §3.2.6 predicts: the benefit survives where the downstream model can exploit finer spatial deformation, and largely vanishes where it cannot.

**A defect found and repaired, which the review made detectable.** In every experimental run of this project prior to the one reported here, each magnification-enabled configuration produced results *bit-identical* to its magnification-disabled counterpart — identical accuracies, identical macro F1 scores, and identical confusion matrices to four decimal places. Given that all four reviewed papers report clear and repeatable effects from magnification, an effect of exactly zero across six independent pairs is not a plausible null result; it is a symptom. The cause was that both arms were reading the same tensor directory, so half of the ablation matrix was duplicated and the magnification hypothesis had never in fact been exercised. The defect is repaired in the run reported in Chapter 5, in which all six pairs differ. The results in this thesis are therefore the first genuine measurement of EVM in the project. Chapter 6 recommends that the existing verification helper be wired in as a start-up assertion so that the failure cannot recur silently.

---

### 3.2.8 Limitations of the existing work, and how this study differs

What the reviewed literature does not establish is threefold. First, every evaluation in the corpus places magnification in front of an **appearance-based** representation (§3.2.5), so the reported gains (§3.2.3) conflate two distinct mechanisms — making faint motion recoverable, and increasing contrast for a texture or gradient descriptor — and no experiment separates them. Second, the magnification factor is selected empirically in every case and transferred between studies without re-validation, despite the α discrepancy noted in §3.2.4. Third, and most consequentially, magnification is always evaluated as an *addition to a fixed pipeline* — the comparison is a two-cell one, with and without — so its interaction with the other components of a system is never measured.

This thesis differs on all three counts. It places magnification in front of an explicit **motion** representation — dense optical flow and optical strain — rather than an appearance representation, isolating the recoverable-motion mechanism from the appearance-contrast one. It selects α from the full-sequence optimum rather than the apex-frame value (§3.2.7), with the choice justified rather than inherited. And it evaluates magnification as one factor in a **full factorial ablation** (§3.1.4), so that its effect is reported not as a single delta but as the six matched-pair measurements of §3.2.7, whose spread across architectures is itself the finding.

---

---

## 3.3 Motion Representation: Dense Optical Flow

> **Scope of this section.** This section reviews dense optical flow — the horizontal and vertical displacement fields that form two of the three input channels used in this thesis — through the four review-corpus papers that apply it: Liong et al. (2019a, 2019b), Xu et al. (2017), and Zhao et al. (2021). The estimators themselves, Farnebäck's polynomial-expansion method and the TV-L1 variational method, lie outside the corpus and are described here only as attributed through those papers.

---

### 3.3.1 Why motion rather than appearance

The decision to hand a classifier a motion field rather than pixels is the single most consequential representational choice in this thesis, and the review corpus supports it from three independent directions.

**The argument from mechanism.** Xu et al. (2017) state the case most directly in their critique of the prevailing approaches. Most existing methods, they observe, "utilize textural features like Gabor filters or LBP descriptors. These features cannot shed light on the mechanism of microexpressions intuitively. A facial-dynamics-based descriptor has the potential to expose the nature of microexpressions." A micro-expression *is* a movement — a brief contraction of specific facial muscles — so a representation that encodes displacement is describing the phenomenon itself, whereas a texture descriptor is describing a side effect of it.

**The argument from confounding.** Given the corpus's demographic and illumination narrowness (§3.1.7), appearance is exactly the wrong thing to model: pixel intensities encode who the subject is and how the scene is lit far more strongly than they encode a few grey levels of transient muscle movement, and a network with sufficient capacity will exploit the former. A displacement field discards absolute intensity by construction, and is therefore invariant to identity and illumination in a way no appearance descriptor is.

**The argument from convergence.** §3.1.6 noted the MEGC 2019 organisers' observation that the top three submissions to the challenge "all used optical flow as their choice of" feature representation (See et al., 2019). Independent teams competing on a common protocol converged on the same representation, which is stronger evidence than any single paper's ablation.

---

### 3.3.2 The brightness-constancy formulation

Liong et al. (2019a) give the fullest derivation in the corpus, and it is worth restating because its assumptions become the limitations discussed in §3.3.6.

Optical flow estimation rests on four stated assumptions: that the apparent brightness of moving objects is unchanged between source and target frames, so that "the noises generated by a large variety of imaging variables such as the shadows, highlights, illumination and surface translucency phenomena are entirely neglected"; that movement between consecutive frames is small; that the flow field is continuous and differentiable in space and time; and that the scene is static and its objects rigid.

The constraint equation that follows from these assumptions is derived in §2.2.2 and is not repeated here; Zhao et al. (2021) reach the same equation and add the flow vector and its amplitude, $V = [u, v]^{\mathsf T}$ with $m = \sqrt{u^2 + v^2}$.

A single scalar equation cannot determine two unknowns per pixel — the aperture problem — so every practical estimator adds an assumption to close the system. **The choice of that assumption is the difference between flow algorithms**, and it is the subject of §3.3.3.

Both papers arrive at the same output format. Liong et al. (2019a) summarise each video as the field $O_i = \{(u(x,y), v(x,y))\}$ over the image domain, giving two representations per clip: the horizontal component $u$ and the vertical component $v$. These are the first two channels of the input tensor used in this thesis; the third, optical strain, is derived from them and is the subject of §3.4.

---

### 3.3.3 Choice of estimator: Farnebäck and TV-L1

The corpus contains two estimator choices, made for stated but different reasons, and never compared against each other.

**Farnebäck.** Zhao et al. (2021) write: "We follow the classic Farnebäck method to implement the OF estimation, which has been implemented and integrated into the OpenCV library and can be used easily." The method approximates the local neighbourhood of each pixel by a quadratic polynomial and solves for the displacement that maps one polynomial expansion onto the next, computed over an image pyramid so that both large and small displacements are recoverable. The justification offered is practical — it is classical, well understood, and available as a maintained implementation. This is the estimator used in this thesis.

**TV-L1.** Liong et al. (2019a) select TV-L1 instead, and give a technical justification: it is "better in preserving the flow discontinuities and is more robust compared to the classical optical flow method." Total-variation regularisation with an L1 data term penalises the *magnitude* of flow gradients rather than their square, which permits sharp discontinuities at motion boundaries instead of smoothing across them, and the L1 data term tolerates brightness-constancy violations better than a quadratic one. Liong et al. (2019b) build STSTNet on the same onset–apex flow construction, though that paper does not name the estimator it uses.

**No paper in the corpus compares the two on micro-expression data.** The claim that TV-L1 preserves discontinuities better is a property of the estimators in general rather than a measured result on this task, and the Farnebäck choice is justified by availability rather than by accuracy. This is a genuine gap: on a corpus where a purely descriptive hyper-parameter has been shown to move the headline figure by several points (§3.1.4), an unexamined estimator choice is not obviously safe (§3.3.8).

---

### 3.3.4 What the flow is computed between: two frames or a sequence

The corpus divides on a question that turns out to be decisive for this thesis, because §3.1 established that temporal modelling is the object under study.

**The two-frame position.** Liong et al. (2019a) compute a single flow field between the onset frame and the apex frame, and use it to represent the entire video. Their reasoning is that the apex "portrays the highest intensity of facial motion among all frames," so the onset-to-apex displacement captures the expression's full excursion; and that because the interval is short — "less than 0.2 seconds" — the brightness-constancy assumption is comfortably satisfied over it. The resulting $u$ and $v$ maps are resized to 28 × 28 and fed to a small CNN. Liong et al. (2019b) adopt the identical scheme for STSTNet, adding optical strain as a third channel computed from the same pair, and it is worth noting for §3.6 that this three-channel construction — **horizontal flow, vertical flow, optical strain, computed from onset and apex** — is the direct precedent for the input tensor used in this thesis.

The economy of the two-frame approach is considerable: an entire video collapses to two images, a substantial defence against overfitting on a corpus this size (§3.1.3).

**The sequence position.** Zhao et al. (2021) reject the collapse. They construct an eleven-frame key-frame sequence from the three annotated frames — onset, apex and offset — by adaptively interpolating eight intermediate transition frames, distributed between the onset–apex and apex–offset intervals in proportion to their durations. Optical flow is then computed between each adjacent pair, yielding a **ten-frame flow sequence** rather than a single field. Their stated principles for the construction are that the sequence must be sufficient to summarise the original video, must contain as few noisy frames as possible, and must highlight the movement of the apex frame, "since it has been proven to contribute major information for facial-expression recognition".

**The trade-off is explicit.** A single onset-to-apex field records *how far* the face moved but nothing about *how* it got there; the temporal arc that defines a micro-expression — neutral, rise, apex, relaxation — is discarded. A flow sequence retains that arc at the cost of more data per sample and therefore more capacity required downstream. Since the central question of this thesis is whether explicit temporal modelling improves recognition, only the second option permits the question to be asked at all; §3.3.7 records the consequence.

---

### 3.3.5 Flow as an alignment tool, not only as a feature

Xu et al. (2017) use optical flow for a second purpose that the other papers do not, and their argument for it is one of the strongest statements in the corpus about why micro-expression analysis is unlike ordinary expression analysis.

They observe that existing methods "place less emphasis on fine face alignment," either assuming the face is already aligned or relying on landmark locators such as the Active Shape Model. Their objection is quantitative in character: "a fine face alignment is more crucial in microexpression recognition than in conventional expression recognition because microexpression is a subtle movement, and thus a rather small misalignment may result in considerably large performance degradation." Where the signal of interest is a few pixels of skin displacement, an alignment error of a few pixels is not a small perturbation — it is the same order as the signal.

Their remedy is a two-stage alignment. Landmarks are located and the face coarsely aligned and cropped; then optical flow between consecutive frames is used to estimate pixel-level movement, and a **fine-scale in-sequence alignment** is performed from those fields before any descriptor is computed. Flow here is not the feature but the instrument that removes residual motion, so that what remains is expression rather than head displacement.

This sharpens the concern raised in §3.2.5 about magnification amplifying head motion indiscriminately: Xu et al. (2017) show that residual head movement is large enough relative to the signal to require dedicated correction in its own right. A pipeline that magnifies before computing flow — as this thesis does — is therefore exposed on both counts, and the exposure is only partly mitigated by the coarse alignment and cropping that CASME II's own release procedure applies (§3.1.2).

---

### 3.3.6 Limitations

**The assumptions are known to be false.** Liong et al. (2019a) are explicit that brightness constancy causes shadows, highlights, illumination change and surface translucency to be "entirely neglected," and that the scene is assumed static and rigid. A human face is neither rigid nor free of translucency, and the assumption of small inter-frame motion sits awkwardly beside a magnification step whose declared purpose is to make motion larger (§3.2).

**Two-frame pipelines inherit the apex-detection error.** Liong et al. (2019a) report this against themselves. On SMIC, where no ground-truth apex is annotated, an automatic spotter must supply it, and "the average of frame difference between the detected and ground-truth apex is 13 frames"; they attribute their weaker SMIC result directly to "extracting the features from imprecise apex frame." Any representation that depends on locating one frame correctly is only as good as the spotter. It is worth recalling from §3.1.6 that one MEGC submission found mid-position frames to be an adequate substitute for annotated apex frames, which suggests the dependence is real but the precision requirement may be looser than assumed.

**The estimator is chosen, not evaluated.** As §3.3.3 noted, no paper in the corpus measures the effect of the estimator on recognition, despite two different choices being in circulation.

**Flow inherits everything upstream.** It is computed on whatever the alignment stage produced, and Xu et al. (2017) demonstrate that landmark-based alignment alone is insufficient for this task. Errors do not merely pass through; a misalignment of a few pixels produces a spurious displacement field of the same magnitude as the signal.

---

### 3.3.7 Implications for this research

| Finding from the review | Decision taken in this thesis |
|---|---|
| Motion descriptors expose the mechanism where textural ones do not (Xu et al., 2017); flow was the convergent choice of the top MEGC entries (See et al., 2019) | The network never sees pixels. Every clip is converted to a motion tensor before training |
| Horizontal and vertical flow fields $u, v$ are the standard pair (Liong et al., 2019a) | Channels 1 and 2 of the input tensor are flow-$u$ and flow-$v$; channel 3 is optical strain (§3.4) |
| Farnebäck is classical and available in OpenCV (Zhao et al., 2021) | Dense **Farnebäck** estimation via OpenCV, matching Zhao et al.'s choice; the TV-L1 alternative of Liong et al. (2019a) is noted as untested here and listed in future work |
| Onset→apex two-frame flow discards the temporal arc (Liong et al., 2019a, 2019b); an interpolated key-frame sequence retains it (Zhao et al., 2021) | Flow is computed **across a full resampled sequence**, not a single onset–apex pair, following the sequence position — since the temporal arc is precisely the object of study |
| Two-frame pipelines are limited by apex-spotting error of up to ~13 frames (Liong et al., 2019a) | The pipeline trims onset→offset using CASME II's annotations and resamples to a fixed length; **no single apex frame must be located correctly**, removing that failure mode entirely. It introduces a different one: three clips are shorter than the 33-frame sampling window, so a few of their 32 pairs are duplicate-frame pairs yielding zero flow (§3.5.7) |
| Fine alignment matters more for micro- than macro-expressions (Xu et al., 2017) | Acknowledged as a residual threat: this pipeline relies on the corpus's coarse alignment and does not perform flow-based fine alignment. Listed in Chapter 6 as the most promising unexploited improvement |

*Table 3.10 — Review findings and the motion-representation decisions they determine.*

**One consequence deserves separate statement, because it explains the most surprising result in Chapter 5.** Optical flow is not a passive recording of the video; it is an *analytical spatio-temporal feature extractor*. Solving the brightness-constancy constraint at every pixel across every adjacent frame pair already performs, in closed form, a large part of the work that a learned spatio-temporal convolutional stem would otherwise have to discover from data. When the input to a network is a flow-and-strain volume rather than raw frames, a 3D convolutional backbone is therefore being asked to re-derive, from 156 clips, structure that has already been computed. Chapter 5 measures that backbone's marginal contribution at **−0.031 macro F1** against 97 % of the study's compute budget (§3.6) — the negative sign is exactly what an analytically-precomputed input predicts a redundant learned stem should produce. The same reasoning predicts the opposite for a temporal encoder, which is given something the flow computation does *not* provide: a model of how the displacement field evolves across the sequence.

---

### 3.3.8 Limitations of the existing work, and how this study differs

The reviewed literature establishes that dense optical flow is an effective and now-conventional representation for micro-expression recognition (§3.3.1–§3.3.2), and that flow can serve as both feature and alignment instrument (§3.3.5). Four things it does not establish. First, the estimator choice is never evaluated on this task (§3.3.3), despite the corpus elsewhere showing that a purely descriptive hyper-parameter can move the headline figure by several points (§3.1.4). Second, the two-frame and sequence positions (§3.3.4) are never contrasted under a common protocol, so no study measures what the collapse costs. Third, the interaction between flow and what consumes it is unexamined: every paper pairs a flow representation with one fixed architecture, so nothing indicates whether a learned spatial backbone adds anything once the input is already a displacement field, or whether it duplicates work already done in closed form. Fourth, alignment quality is treated as either solved or as a contribution in its own right (Xu et al., 2016) but never as a controlled variable alongside the representation.

This thesis differs on the second and third counts, which are the ones its research question requires. It computes flow across a **full temporally resampled sequence** rather than an onset–apex pair (§3.3.4), so that the temporal arc survives into the representation and a temporal model has something to model — and, as a by-product, removes the apex-spotting failure mode documented in §3.3.6. And it holds the motion representation fixed while varying the components that consume it across a **full factorial ablation** under complete leave-one-subject-out (§3.1.4), which makes it possible to ask, and answer, whether a learned spatio-temporal backbone contributes anything on top of an analytically computed motion field. The answer reported in Chapter 5 — that it does not, at measurable negative cost (§3.3.7) — is a statement about the *interaction* between representation and architecture that no single-architecture study in the corpus is positioned to make. On the first and fourth counts this thesis inherits the field's limitations rather than resolving them: the Farnebäck estimator is adopted without comparison against TV-L1, and no flow-based fine alignment is performed. Both are stated as limitations in Chapter 6.

---

---

## 3.4 Deformation Representation: The Optical Strain Tensor

> **Scope of this section.** §3.3 covered dense optical flow; this section covers the third channel it feeds, optical strain, through the four review-corpus papers that develop it — Shreve et al. (2011) and the Liong et al. strain series — together with STSTNet's use of it as an input channel. The continuum-mechanics results the strain tensor rests on are attributed through those same papers rather than to primary mechanics texts, which sit outside the review corpus.

---

### 3.4.1 What strain adds that flow does not

Optical flow answers *where each point moved*. Optical strain answers a different question: *how much the surface deformed*. The distinction is not cosmetic, and Shreve et al. (2011) motivate their method on precisely this ground — that facial expressions are legible in "the strain impacted on the facial skin due to the non-rigid motion caused during expressions."

The practical consequence follows from what the strain operator does. Strain is built from *spatial derivatives* of the displacement field. A spatially uniform displacement — every pixel moving by the same amount, which is what a small rigid head translation produces — has zero spatial derivative and therefore contributes nothing to the strain field. A localised muscle contraction, in which neighbouring skin patches move by different amounts, produces a large one. Strain is thus a filter that suppresses rigid motion and preserves non-rigid deformation, which is exactly the discrimination micro-expression analysis requires and which §3.3.5 established that flow alone does not provide.

Shreve et al. (2011) make two separate sets of claims. The first concerns strain itself, which they hold to be "robust to moderate amounts of head translations" and which "has been shown in [8] [12] to be robust to adverse lighting conditions and heavy make-up". The second is a list of advantages of their overall method, two items of which bear on strain: that its reliability has been demonstrated "even under adverse illumination and heavy make-up", and that "optical strain can be quickly and accurately calculated from optical flow fields", so it costs almost nothing once flow has been computed. It is worth recording that both robustness claims are attributed by Shreve et al. to earlier work — their references [8] and [12] — rather than measured in that paper.

---

### 3.4.2 The finite strain tensor

Shreve et al. (2011) give the derivation that every subsequent paper in the corpus reuses.

The tensor itself, its normal and shear components and its scalar magnitude are defined in §2.2.3. What matters here is the discretisation, which differs between the reviewed papers.

Because each component is a function of the continuous displacement field but only discrete flow is available, the derivatives are approximated from the estimated flow $(p, q)$, where $u = p\,\Delta t$ and $v = q\,\Delta t$ for a fixed inter-frame interval $\Delta t$. Shreve et al. (2011) evaluate the resulting spatial derivatives by the **central difference method**, for example

$$\frac{\partial u}{\partial x} \;=\; \frac{u(x + \Delta x) - u(x - \Delta x)}{2\,\Delta x} \;\doteq\; \frac{p(x + \Delta x) - p(x - \Delta x)}{2\,\Delta x}$$

with $(\Delta x, \Delta y) \approx 2$–3 pixels. Liong et al. (2014a) reproduce the same construction using finite difference approximation with $(\Delta x, \Delta y)$ "preset distances of 1 pixel."

**The strain magnitude.** Shreve et al. (2011) reduce the tensor to a scalar per pixel by summing the component contributions, optionally normalised to 0–255 for display as a *strain map*. Liong et al. (2014a) — and, in identical form, Liong et al. (2014b) — state the scalar reduction explicitly as the Frobenius norm of the tensor:

$$\varepsilon = \sqrt{\varepsilon_{xx}^{\,2} + \varepsilon_{yy}^{\,2} + \varepsilon_{xy}^{\,2} + \varepsilon_{yx}^{\,2}}$$

This scalar discards the *direction* of deformation and retains only its intensity — a deliberate reduction, since the direction information is already carried by the $u$ and $v$ channels alongside it.

---

### 3.4.3 A declared divergence in this thesis's implementation

The formulation implemented in this thesis (`Stage1_DataPipeline/step2_extraction/flow_strain_extractor.py`) computes

$$\varepsilon_{\text{os}} = \sqrt{\varepsilon_{xx}^{\,2} + \varepsilon_{yy}^{\,2} + \varepsilon_{xy}^{\,2}}$$

with the shear term counted **once**, where Liong et al. (2014a) count it twice as $\varepsilon_{xy}$ and $\varepsilon_{yx}$. Since the tensor is symmetric by construction, the published form is equivalent to $\sqrt{\varepsilon_{xx}^2 + \varepsilon_{yy}^2 + 2\varepsilon_{xy}^2}$, so the two differ by the weight given to shear relative to normal strain.

**The most directly relevant precedent, however, uses the same three-term form.** STSTNet (Liong et al., 2019b), which §3.4.4 identifies as the origin of this thesis's three-channel input, prints its own strain magnitude as $|\varepsilon_{x,y}| = \sqrt{(\partial u/\partial x)^2 + (\partial v/\partial y)^2 + [\tfrac{1}{2}(\cdot)]^2}$ — three squared terms, with the shear contribution appearing once. The implementation used here therefore follows the paper whose input construction it adopts, and diverges only from the earlier four-term convention of Liong et al. (2014a, 2014b, 2016). This is stated rather than glossed because the two conventions coexist in the literature without either being flagged. Its practical significance is limited: both expressions are monotone in the same underlying quantities, both are strictly positive functions of the same two derivatives, and the channel is subsequently min–max normalised to $[0,1]$ across the whole sequence before training, which removes any global scaling. What is not removed is the *relative* weighting of shear against normal strain, which differs by a factor of $\sqrt{2}$ on the shear contribution. No experiment in this thesis isolates that difference, and it is recorded in Chapter 6 as a small unexamined implementation choice rather than claimed to be equivalent.

Spatial derivatives are computed with NumPy's `gradient`, which applies central differences over the interior of the array with a one-pixel spacing — matching the discretisation of Liong et al. (2014a) rather than the 2–3 pixel spacing of Shreve et al. (2011).

---

### 3.4.4 Strain as a feature, and strain as a weight

The corpus develops two distinct uses of the strain field, and Liong et al. (2016) evaluate both together.

**Optical Strain Features (OSF).** The strain map is computed for each consecutive frame pair, filtered, and then **temporally pooled** into a single composite strain map. The pooling operator changed across the series: Liong et al. (2014a, 2014b) sum the per-pixel magnitudes across the sequence, while Liong et al. (2016) — the source of the results in §3.4.5 — revise this to temporal *mean* pooling, on the grounds that it "ensures that the optical strain magnitudes are normalized based on their respective sequence lengths". The composite map that serves as "a temporal representative image for the respective video." The pooled map is max-normalised "to increase the significance of the strain information," resized, and used directly as the feature. Liong et al. (2014a) apply "two types of filters, which are the Wiener and Gaussian filters, to all the strain map images in order to suppress the background noise and interfering signals" — an explicit acknowledgement that the strain field is noisy.

**Optical Strain Weighted features (OSW).** Rather than being classified directly, the strain field is used to decide *which parts of the face matter*. Liong et al. (2014b) partition the image into $N \times N$ non-overlapping blocks and pool the strain magnitudes within each block to a single value; the resulting normalised coefficients form a weight matrix that is multiplied with the LBP-TOP histograms extracted from the corresponding blocks on the XY plane. Facial regions undergoing strong muscle contraction are thereby up-weighted and quiescent regions suppressed, before any classifier is trained.

The pooling that produces those weights is performed **separably**: Liong et al. (2016) "consider spatio-temporal pooling in a separable fashion; spatial mean pooling is performed first, followed by temporal mean pooling," with mean rather than sum pooling chosen so that magnitudes remain comparable across sequences of different length.

**OSF + OSW.** The third variant concatenates the two feature histograms. This combination is the one that performs best in their experiments.

Liong et al. (2019b) later reduce the idea to its simplest form for STSTNet, where optical strain is neither pooled nor used as a weight but supplied as **a third input channel alongside horizontal and vertical flow**, computed from the onset and apex frames. That three-channel construction is the direct precedent for the input tensor used in this thesis.

---

### 3.4.5 What strain measurably buys — and the protocol caveat

Liong et al. (2016) report the clearest quantification, and it is more equivocal than the strain literature's framing suggests.

| Method | Recog-SMIC (micro / macro avg.) | Recog-CASME II |
|---|:--:|:--:|
| Baseline | 45.73 / 47.76 | 61.94 |
| OSF | 41.46 / 46.00 | 51.01 |
| OSW | 49.39 / 50.71 | 61.94 |
| **OSF + OSW** | **52.44 / 58.15** | **63.16** |

*Table 3.11 — Recognition performance of the strain-based methods, after Liong et al. (2016, Table 8). SMIC results use LOSO cross-validation; CASME II results use leave-one-video-out.*

Three observations follow, and all three bear on this thesis.

**The gain is dataset-dependent and much smaller on CASME II.** Liong et al. (2016) summarise their own improvement as "+5% and +10% on micro-expression recognition performance for the SMIC dataset using 5 × 5 and 8 × 8 block partitions," against "+0.41% and +1.22% in 5 × 5 and 8 × 8 block partitions respectively" on CASME II. An improvement of under half a percentage point on the corpus used in this thesis is not a robust effect.

**The CASME II figures are obtained under a weaker protocol.** Their table footnotes SMIC results as leave-one-subject-out and CASME II results as leave-one-*video*-out — the same non-subject-disjoint protocol issue §3.1 raises for both the CAS(ME)² and original CASME II baselines. Comparison of any strain result on CASME II against the LOSO figures in this thesis is therefore not like-for-like.

**Strain alone underperforms; strain as a weight does not.** OSF used directly is *worse* than baseline on CASME II (51.01 against 61.94); OSW matches it exactly; only the concatenation improves on it. The evidence that strain carries independent discriminative power is weaker than the evidence that it usefully re-weights something else.

A fourth observation is methodological and supports §3.1.6 directly. Liong et al. (2016) distinguish macro-averaged from micro-averaged accuracy under LOSO — the former averaging subject-wise accuracies, the latter computed over all samples — and argue for reporting F1, precision and recall because these "provide a more meaningful perspective than accuracy rates when the datasets used are naturally imbalanced since each subject has a different number of video samples." That is the same argument, from a different direction, that this thesis makes for pooled macro F1 over per-fold averaging.

---

### 3.4.6 Limitations

**Strain is a derivative of an estimate, so it amplifies noise.** The flow field is itself the solution of an ill-posed problem (§3.3.2); differentiating it spatially amplifies whatever estimation error it contains. Every reviewed paper responds with filtering — Liong et al. (2014a) apply Wiener and Gaussian filters explicitly for this purpose — and Liong et al. (2016) close by recommending "better noise filtering techniques and masking of different face regions" to alleviate instability of illumination and intensity changes on the face, which they in turn expect to "reduce the erroneous optical flow/strain computation."

**Strain adds no information beyond the flow field.** It is a deterministic, differentiable function of $u$ and $v$. In an information-theoretic sense a model given $u$ and $v$ has everything needed to compute strain itself. Its value is therefore one of *representation* — making deformation intensity directly available at the input rather than requiring the network to learn a differential operator from 156 clips — not one of added evidence. No paper in the corpus tests whether a sufficiently expressive model given only flow recovers the benefit.

**Every strain method in the corpus collapses the temporal axis.** OSF pools the strain sequence into one composite map; OSW pools it into one weight matrix; STSTNet computes it from a single onset–apex pair. In all three the temporal evolution of the deformation is discarded before classification.

**The robustness claims are largely inherited.** Shreve et al.'s (2011) illumination- and make-up-robustness claims are attributed to prior work; the reviewed papers do not measure them.

**The magnitude discards direction.** Whether the skin is stretching or compressing, and along which axis, is not recoverable from the scalar. This is defensible when $u$ and $v$ accompany it, and lossy when the strain map is used alone.

---

### 3.4.7 Implications for this research

| Finding from the review | Decision taken in this thesis |
|---|---|
| Strain suppresses rigid translation and preserves non-rigid deformation (Shreve et al., 2011) | Optical strain is included as **channel 3**, alongside flow-$u$ and flow-$v$ — the same three-channel construction STSTNet uses (Liong et al., 2019b) |
| Strain is derived from flow at negligible cost (Shreve et al., 2011) | Strain is computed in the same pass as the flow field, from its spatial gradients, adding no separate estimation stage |
| Tensor components from central differences of the flow (Shreve et al., 2011; Liong et al., 2014a) | `np.gradient` central differences at 1-pixel spacing, matching Liong et al. (2014a) |
| Two magnitude conventions coexist: four terms in Liong et al. (2014a, 2014b, 2016), three in STSTNet (Liong et al., 2019b) | The **three-term** form is implemented, matching STSTNet — the paper this thesis's input construction follows. Declared in §3.4.3; the difference from the four-term convention is a √2 weighting on shear and is untested |
| Strain has a numerical range unlike raw flow (implicit in the max-normalisation of Liong et al., 2016) | Each channel is **min–max normalised independently** to $[0,1]$ across the sequence, so the strain channel cannot dominate training |
| Every reviewed method pools strain over time (Liong et al., 2014a, 2014b, 2016; Liong et al., 2019b) | **Temporal pooling is deliberately not applied.** The strain field is retained as a per-frame-pair sequence, since the temporal evolution of deformation is what the temporal encoder under study is meant to model |
| Strain maps are noisy and are routinely filtered (Liong et al., 2014a, 2016) | **No dedicated strain post-filter is applied**; the only smoothing is that internal to the Farnebäck estimator. Recorded in Chapter 6 as an untested omission |
| Strain's measured benefit on CASME II is +0.41 % to +1.22 %, under leave-one-video-out (Liong et al., 2016) | Expectations are set accordingly: a small effect on this corpus is the *predicted* result, not an anomaly |

*Table 3.12 — Review findings and the deformation-representation decisions they determine.*

**Two consequences deserve statement.**

The first concerns what the ablation in Chapter 5 can and cannot say about strain. Because strain is present in all twelve configurations — it is part of the input tensor, not one of the four switched components — this thesis does **not** measure its marginal contribution. That is a scope limitation, and an examiner is entitled to raise it. The justification is that §3.4.5 shows the published effect on CASME II to be under 1.5 percentage points under an easier protocol, so a factorial arm devoted to it would very probably have returned an unresolved result at the cost of doubling an already 50-GPU-hour sweep. Chapter 6 lists a strain-on/strain-off arm as the natural next ablation.

The second extends §3.3.7's conclusion that dense flow already performs, in closed form, much of the spatio-temporal feature extraction a learned 3D backbone would otherwise have to discover. Strain adds a further layer to that claim rather than merely inheriting it: the strain channel is itself a hand-specified spatial differential operator applied to the flow field, so what reaches the network has already had **two** levels of the convolutional hierarchy — displacement estimation, then first-order spatial differentiation — computed for it in closed form, rather than one. The 3D-CNN's measured marginal contribution of −0.031 macro F1 (§3.6) is consistent with this.

---

### 3.4.8 Limitations of the existing work, and how this study differs

The reviewed literature establishes the finite strain tensor as a principled deformation descriptor for facial analysis, provides a complete and reproducible discretisation of it from optical flow, demonstrates that strain suppresses rigid head motion by construction, and shows two workable uses — as a feature in its own right and as a spatial weighting on another descriptor. What it does not establish is fourfold, and each point was flagged above in §3.4.5–§3.4.6: strain's independent contribution on CASME II is small and equivocal, and measured moreover under leave-one-video-out rather than a subject-disjoint protocol, so the headline claim for strain rests largely on SMIC; no paper isolates strain against a matched control, so what it adds *over flow alone* remains unmeasured; every strain method reviewed collapses the temporal axis before classification, discarding what may be the most expression-specific thing strain could encode; and the noise sensitivity of differentiating an estimated field is handled by filtering choices that are stated but never evaluated.

This thesis differs on the third count and inherits the limitations of the others. It retains strain as a **full temporal sequence** rather than a pooled composite, so that the deformation trajectory is available to the temporal encoder under study — the first of the four listed gaps that this project's research question actually requires closing. On the first and second counts it does not improve on the literature, for the reasons §3.4.7 already gives; Chapter 6 records the strain-on/strain-off ablation as the most obvious extension. On the fourth it is weaker than the reviewed work, applying no dedicated strain filtering where Liong et al. (2014a) apply Wiener and Gaussian filters, and this is likewise declared rather than defended. The contribution of this section is therefore primarily interpretive: it establishes that the input representation used in this thesis already embeds a hand-specified differential operator, which becomes the principal explanation offered in Chapter 5 for why an additional learned convolutional stem fails to pay for itself.

---

---

## 3.5 Temporal Normalisation: The Temporal Interpolation Model

> **Scope of this section.** §3.3 and §3.4 covered *what* is computed from each frame pair; this section covers *which* frame pairs, through the two review-corpus papers that treat temporal interpolation directly plus several others that use it as an unremarked preprocessing step. The Temporal Interpolation Model itself was introduced by Pfister et al., outside the review corpus, and is described here only through the reviewed papers that implement and evaluate it.

---

### 3.5.1 The problem: sequences of unequal length

Micro-expressions vary in duration, and a corpus reflects that variation directly: the three-class CASME II working set spans roughly a four-fold range of sequence length, from tens of frames to over a hundred (§3.1.7h).

This creates two distinct difficulties.

**The architectural one.** Any batched network requires a fixed input shape. A model consuming a spatio-temporal volume must be told how many frames that volume contains, and the corpus does not supply a constant.

**The descriptor one, which is older and better documented.** Li et al. (2018) set it out for the low-frame-rate case: "when recording at a standard speed of 25 fps, some MEs only last for four to five frames, which limit the application of some spatial-temporal descriptors, e.g., if we use LBP-TOP we can only use the radius $r = 1$." A descriptor with a temporal radius cannot be applied at all if the sequence is shorter than that radius, so short clips constrain the descriptor for the whole dataset.

Lu et al. (2015) add a third motivation that is about signal rather than mechanics: temporal normalisation removes "temporal fluctuations of micro-expressions, which may introduce noise irrelevant to classification of micro-expressions." On this view, unequal duration is not merely an inconvenience but a nuisance variable — a clip is not more *surprised* for lasting longer, so length is variance the classifier should not have to model.

---

### 3.5.2 What TIM is

Lu et al. (2015) give the most compact description in the corpus: "TIM is a manifold-based interpolation method, which builds a low-dimensional embedding of an image sequence, and then interpolates a curve in the low-dimensional space. The interpolated frames are mapped back to the original high-dimensional space to form the temporally normalized image sequence."

Three properties follow from that construction and matter for §3.5.7.

First, TIM operates on **images**, not on features: it embeds whole frames, interpolates in the embedded space, and reconstructs frames. Second, it can **synthesise frames that were never recorded** — Lu et al. (2015) note that Pfister et al. originally proposed TIM "to generate sufficient frames from an image sequence," i.e. as an *upsampler* for short clips, not only as a length-standardiser. Third, because it interpolates along a curve in a learned low-dimensional space rather than by pixel-wise blending, it assumes the appearance variation across a clip is smooth and low-dimensional — an assumption that holds well for a face moving slightly and less well for a noisy or artefact-laden sequence.

---

### 3.5.3 TIM in practice: the convergence on ten frames

TIM appears throughout the corpus as an unremarked preprocessing step, almost always at ten frames.

- **SMIC's baseline** (Li et al., 2013) normalises clips with TIM to ten frames before LBP-TOP, reaching 48.78 % on the three-class HS task under leave-one-subject-out (§3.1.5).
- **Li et al. (2018)** apply TIM10 after motion magnification, in the pipeline whose magnification results §3.2.3 reported.
- **Lu et al. (2015)** apply TIM as the first stage of their Delaunay-based coding model, and follow the protocol of Pfister et al. so as to compare directly against TIM10 baselines — reported at 74.3 % for separating micro from non-micro expressions and 71.4 % for positive-versus-negative classification, both under leave-one-subject-out.
- **CAS(ME)²'s baseline** (Qu et al., 2016) likewise combines TIM with LBP-TOP.

The convergence is striking and, in the corpus, essentially unjustified: ten frames became the default because the SMIC baseline used ten frames. Only one study interrogates the choice.

---

### 3.5.4 How many frames? The one systematic comparison

Ben et al. (2021) run the only controlled experiment on the question in the review corpus. Holding alignment, descriptor and classifier fixed — JCFDA alignment, LBP-TOP, SVM with an RBF kernel — they interpolate micro-expression sequences to **10, 20, 30, 40, 50, 60, 70, 80, 90, 100 and 110 frames** using two algorithms, **Newton interpolation** and **TIM**, and record the best recognition rate at each length on the MMEW dataset.

Two results follow.

**TIM outperforms Newton interpolation.** Newton reaches its best recognition rate of **33.3 %** at 60 interpolated frames; TIM reaches **38.9 %**, its highest, "when the micro-expression sequence is respectively interpolated to 30 or 60 frames." A 5.6-point margin between two interpolation algorithms, with everything else held constant, is a larger effect than most component-level differences reported elsewhere in this review. On that basis Ben et al. adopt TIM at 60 frames for their subsequent experiments.

**The relationship between sequence length and accuracy is non-monotonic.** "As the number of frames increases, the recognition rates of the two interpolation algorithms increase initially and then decrease towards the end." Too few frames under-describe the temporal arc; too many appear to add nothing but interpolated redundancy and, presumably, dimensionality.

The structural parallel with §3.2.3 is worth naming. The magnification factor α and the interpolated sequence length are both *preprocessing* parameters, both exhibit an interior optimum, both have that optimum located only by exhaustive sweep, and in neither case does the corpus offer a principled way to choose the value in advance. Two of the three preprocessing decisions in this thesis's Stage 1 are of this character.

**Two caveats limit how far this transfers.** The experiment is run on MMEW with LBP-TOP, not on CASME II and not with a learned model. It is also conducted almost entirely in the *upsampling* regime — MMEW's longest sequence is about 108 frames, so interpolating to 30, 60 or 110 adds frames for most clips. This thesis operates in the opposite regime: only 3 of its 156 clips fall below the 33-frame target while 81 are downsampled by a factor of two or more (§3.5.7). An optimum measured while synthesising frames need not be the optimum when discarding them. Whether the 30–60 frame optimum holds for a different corpus, a different descriptor, or a network that consumes motion rather than appearance is untested.

---

### 3.5.5 Alternatives to uniform interpolation

The corpus spans a remarkably wide range of answers to "how many frames", and two alternatives to plain interpolation are worth recording.

**Adaptive, annotation-aware construction.** Zhao et al. (2021) do not interpolate uniformly. They build an eleven-frame key-frame sequence from the three annotated frames — onset, apex and offset — by generating eight intermediate transition frames, allocated between the onset–apex and apex–offset intervals **in proportion to the duration of each sub-interval**, so that the sampling density follows the expression's own phase structure. Their stated principles include that the sequence must summarise the original video, contain as few noisy frames as possible, and highlight the movement of the apex frame "since it has been proven to contribute major information for facial-expression recognition." Optical flow is then computed between adjacent key-frames, yielding a ten-frame flow sequence.

**Collapse to two frames.** Liong et al. (2019a, 2019b) take the limiting case: the entire clip is represented by the onset and apex frames alone, so the temporal dimension is removed rather than normalised (§3.3.4).

Taken together, the corpus contains temporal representations of length 2, 10, 11, 30, 60 and 110 frames, chosen variously by convention, by sweep, and by architecture. There is no consensus, and the one systematic comparison is single-corpus.

---

### 3.5.6 Limitations

**Ten frames is a convention, not a finding.** The most-used setting in the corpus traces to a single baseline paper's choice, and the only study to test it finds a substantially higher optimum — 30 to 60 frames rather than 10.

**The one systematic result is narrow.** Ben et al. (2021) test on one dataset with one descriptor and one classifier. Interpolation length interacts with the temporal receptive field of whatever consumes it, and their consumer is LBP-TOP.

**Frame synthesis and motion analysis are in tension.** TIM generates frames that were not recorded, by interpolating along a manifold of appearances. Where the downstream representation is appearance-based, this is benign. Where the downstream representation is *motion* — flow computed between adjacent frames — synthesised intermediate frames produce synthesised displacements: motion that no muscle produced. No paper in the corpus examines this interaction, because those using TIM extract appearance descriptors and those extracting flow either use two frames or, in Zhao et al.'s case, generate transition frames and accept the consequence without discussing it.

**Downsampling forfeits the corpus's principal advantage.** §3.1 establishes why CASME II was recorded at 200 fps rather than its predecessor's 60 fps; any pipeline that reduces a clip to a few tens of frames gives back part of that advantage. The corpus never discusses this trade-off, because the papers that use TIM10 do so on 100 fps SMIC data as readily as on 200 fps CASME II data.

---

### 3.5.7 Implications for this research

| Finding from the review | Decision taken in this thesis |
|---|---|
| Fixed-length input is required, and unequal duration is nuisance variance (Lu et al., 2015) | Every clip is normalised to a fixed temporal length before training |
| TIM beats Newton interpolation by 5.6 points, with a joint optimum at 30 or 60 frames (Ben et al., 2021) | **L = 33 frames sampled, giving T = 32 flow pairs** — placed at the lower of the two reported optima rather than at the field-conventional 10 |
| Sequence length has an interior optimum found only by sweep (Ben et al., 2021) | Acknowledged: T = 32 is fixed throughout this study and is **not** swept. Listed in Chapter 6 as an untested parameter |
| Onset and offset are annotated in CASME II (§3.1.2) | Sampling is bounded by the annotated onset and offset, so the fixed-length window spans the expression rather than surrounding neutral frames. This holds on the image-folder path used for CASME II; the loader's `.avi` branch samples across a whole file and is not exercised here |
| TIM synthesises frames; flow between synthesised frames is synthesised motion (§3.5.6) | **No frames are synthesised.** See the declaration below |

*Table 3.13 — Review findings and the temporal-normalisation decisions they determine.*

**A declaration about what this pipeline actually implements.** The module performing this step is named for the Temporal Interpolation Model and its documentation describes it as such, but what it computes is **uniform index sampling**, not manifold-based interpolation: it selects 33 evenly spaced indices between the annotated onset and offset and loads those existing frames. No embedding is built, no curve is interpolated, and no frame is synthesised. This is a materially different operation from the TIM of Pfister et al. described in §3.5.2, and it is named accurately here rather than in the pipeline's own terms.

The choice is nonetheless defensible on the grounds of §3.5.6. Because the downstream representation is optical flow between *adjacent sampled frames*, interpolating intermediate images would manufacture displacement fields corresponding to motion that never occurred. Subsampling has the opposite failure mode — it discards temporal resolution but never fabricates it — which is the safer error for a motion-based pipeline. What the thesis gives up is TIM's ability to *lengthen* short clips.

**Two quantitative consequences follow, and both are stated rather than buried.**

*Effective temporal resolution is reduced, by a clip-dependent factor.* Sampling 33 frames from a clip of $N$ frames recorded at 200 fps yields an effective sampling rate of $6600/N$ fps. For the median clip ($N = 66$) this is **100 fps** — exactly SMIC's recording rate, and half of what CASME II was built to provide. For the longest clip ($N = 126$) it falls to roughly **52 fps**, below the 60 fps of the original CASME database that Yan et al. (2014) judged inadequate. Eighty-one of the 156 clips are downsampled by a factor of two or more. This is a real cost of fixed-length normalisation and it is not one the reviewed literature discusses.

*Magnification is applied after subsampling, which shifts the effective pass-band.* The pipeline loads the 33 sampled frames first and applies Eulerian magnification to that sequence, passing the corpus's nominal 200 fps to the magnifier. But the sampled sequence spans the clip's full duration in 33 frames, so its true inter-frame interval is $N/(200 \times 32)$ seconds rather than $1/200$. For the median clip the sequence's effective rate is 100 fps, so a band-pass specified as 5–25 Hz against a nominal 200 fps corresponds to roughly 2.5–12.5 Hz in physical terms; for the longest clip the discrepancy is larger still. The band is therefore not the 5–25 Hz that §3.2.7 derives from the duration rule reported by Bai et al. (2021), and it is clip-dependent rather than constant. This is recorded here as a discrepancy between the intended and realised pass-band; whether magnifying before rather than after subsampling changes the measured EVM effect is untested and is listed in Chapter 6.

*Three clips are shorter than the sampling window.* The shortest clips contain 31 frames against a target of 33, so uniform index sampling necessarily repeats indices, producing a small number of duplicated frames and hence near-zero flow at those positions. This affects **3 of 156 clips** and is noted for completeness rather than as a material threat.

---

### 3.5.8 Limitations of the existing work, and how this study differs

The reviewed literature establishes that temporal normalisation is necessary, that manifold-based interpolation outperforms polynomial interpolation where the two have been compared, and that the number of frames has an interior optimum. What it does not establish is fourfold, and each was flagged above in §3.5.3–§3.5.6: the field's most-used setting, ten frames, rests on convention rather than evidence, with the one controlled study finding an optimum three to six times higher, confined to one dataset, one hand-crafted descriptor and one classifier that need not transfer to a learned model consuming motion; no paper examines the interaction between frame *synthesis* and motion extraction, for the reason §3.5.6 gives; and the trade-off between temporal normalisation and acquisition frame rate goes unremarked, the same TIM10 setting applied to 100 fps and 200 fps corpora alike.

This thesis differs on the third and fourth counts and inherits the limitations of the first two. It **does not synthesise frames**, sampling only recorded ones, so that every flow field it computes corresponds to displacement that physically occurred — the appropriate choice for a pipeline whose entire representation is motion, and one the corpus does not make explicitly. And, as §3.5.7 quantifies, it states the resulting cost rather than leaving it unremarked — a price no reviewed paper reports. On the first two counts it improves on convention only partially: T = 32 is placed at the lower of the two optima Ben et al. (2021) identify rather than at the conventional 10, but the value is fixed rather than swept, and the corpus on which that optimum was measured is neither CASME II nor evaluated with a learned temporal model. A sweep over T under the full leave-one-subject-out protocol is listed in Chapter 6 as a direct extension of the ablation reported here.

---

---

## 3.6 The Learned Spatial Backbone: Shallow 3D Convolutional Networks

> **Scope of this section.** This section covers the convolutional backbone — switched on and off as Variable C of this thesis's ablation — through four papers from the review corpus. The general-purpose architectures cited for comparison in those papers — GoogLeNet, VGG16, ResNet-101 and DenseNet-169 — are outside the review corpus and appear only as figures quoted from the reviewed work.

---

### 3.6.1 The case for, and against, a learned spatial stem

By 2019 the field had converged on optical flow as the input representation (§3.3.1). The open question was what should consume it. Liong et al. (2019b) frame the motivation for a learned backbone in terms of what hand-crafted descriptors cannot do: "the robustness of deep learning has yielded promising performance beyond that of traditional handcrafted approaches."

Against that stands the small-data constraint already established (§3.1.3, §3.1.7): a general-purpose image classifier's parameter count runs into the tens of millions, against a training set of at most a few hundred clips — a ratio orders of magnitude beyond anything such architectures were designed to fit. Xia et al. (2020b) state the resulting failure mode precisely: "the important subtle dynamics are prone to disappearing in the domain shift such that the models greatly degrade their performance, especially for deep models."

The literature's answer is not to abandon the learned backbone but to shrink it, and §3.6.3 shows how far that shrinking goes.

---

### 3.6.2 What "3D CNN" means in this literature

The term is used for three materially different architectures, and the distinction matters directly for how this thesis's ablation result should be worded.

**A convolution over spatial and modality axes, with no temporal axis at all.** STSTNet (Liong et al., 2019b) is the canonical shallow model in this field and is described as a "Shallow Triple Stream Three-dimensional CNN". Its input is a single **28 × 28 × 3** tensor computed from the onset and apex frames, where the third dimension holds the three motion modalities — horizontal flow, vertical flow, optical strain — not time. The image "is passed through three parallel streams, each consists of a convolutional layer (each stream has a different number of kernels, i.e. 3, 5, 8) followed by a max pooling layer," with the stated purpose of supplementing "the small scale input data by utilizing different number of 3 × 3 kernels on each stream to avoid the problem of underfitting the data." Stream outputs are merged channel-wise, average-pooled 2 × 2, and passed to a 400-node fully connected layer. The reported architecture table gives filter size 3 × 3 × 3 with 3, 5 and 8 filters and output sizes 28 × 28 × 3, 28 × 28 × 5 and 28 × 28 × 8. **There is no time dimension in this network**; the entire clip has already been collapsed to a single frame pair upstream (§3.3.4).

**A convolution genuinely extended across time.** STRCN (Xia et al., 2020a) takes the opposite approach, proposing "several recurrent convolutional layers for extracting visual features" and explicitly "two types of extending the connectivity of convolutional networks across temporal domain," modelling spatiotemporal deformation "in views of facial appearance and geometry separately."

**A convolution over a short flow sequence.** Zhao et al. (2021) apply a C3D-style network to the ten-frame optical-flow sequence built from their eleven adaptively interpolated key-frames (§3.5.5), so the temporal axis is present but short.

The label "3D CNN" therefore covers architectures with no temporal extent, with recurrent temporal extent, and with a ten-frame temporal extent. §3.6.6 declares which of these describes the component ablated in this thesis, because the answer is not the one the component's name implies.

---

### 3.6.3 The central finding: shrink the model *and* the input

Xia et al. (2020b) provide the most useful experiment in the corpus for a study of this kind, because they treat architectural depth and input resolution as controlled variables rather than as design choices.

They construct four models of increasing depth — from a single convolutional layer alone (Model 1), through one convolutional plus one recurrent convolutional layer (Model 2), up to one convolutional plus three recurrent convolutional layers (Model 4) — and evaluate each at nine input resolutions: **20, 40, 60, 80, 100, 150, 200, 250 and 300 pixels**, using optical flow as input and UAR as the metric. Three results follow.

**Deeper is worse.** "Using deeper models, e.g., Model 3 and 4, have worse recognition performances compared to shallower … models, e.g., Model 1 and Model 2." They emphasise that even these are far shallower than conventional networks such as ResNet-101 or DenseNet-169, and are "still very easy to be distracted by the domain shift."

**Higher resolution is worse, and there is a threshold.** "The model performance will be degraded with larger input resolutions… Especially, for the deeper models (Model 3 and 4), the performance begins to decrease dramatically when the resolution is bigger than **100 × 100**." Shallow models are "basically robust to the change of input resolution", with performance improving toward lower resolution above a floor of about 40 × 40.

**The conclusion they draw is a design rule.** "Suitable input resolutions (flow-map resolutions) and model architectures (network layers) are helpful to ease the degradation." They select their four-layer Model 2 as the backbone, "which pursues a trade-off between powerful representation ability and domain-shift overfitting."

STSTNet is the same argument taken to its limit and validated competitively. Liong et al. (2019b) report parameter counts alongside those of the models they compare against: **STSTNet 0.00167 M parameters across 2 layers at 28 × 28 × 3 input**, against OFF-ApexNet's 2.77 M across 5 layers and GoogLeNet's 7 M across 22. Roughly seventeen hundred learnable parameters — three orders of magnitude below a general-purpose classifier — and, as §3.1.6 recorded, that network posts a UF1 of 0.8382 and the highest UAR of any MEGC 2019 entry on the CASME II subset.

---

### 3.6.4 Coping with limited and imbalanced training data

Three distinct strategies appear, and all three have counterparts in this thesis's training configuration.

Xia et al. (2020a) address "the shortcomings of limited and imbalanced training samples" with "temporal data augmentation strategies as well as a balanced loss … jointly used for our deep network" — the same pairing of augmentation and loss rebalancing that §3.1.8 recorded as a source of difficulty in this project, where applying two imbalance corrections simultaneously caused single-class collapse.

Zhao et al. (2021) borrow from few-shot learning, splitting training into a prior stage that learns generic features from same/different sample pairs and a target stage that learns high-level discriminative features, on the analogy of a child generalising from comparison before naming.

Xia et al. (2020b) take a third route: adding capability without adding capacity. They "develop three parameter-free modules … integrate with RCN without increasing any learnable parameters," so that representational power grows while the parameter count — and therefore the overfitting risk — does not. That principle is the direct justification for the attention mechanism reviewed in §3.7.

---

### 3.6.5 Limitations

**The terminology obscures what is being compared.** Papers titled as three-dimensional CNNs include networks with no temporal axis at all. A reader comparing "3D-CNN" results across the corpus is not comparing like with like.

**Input resolution is almost never controlled.** Xia et al. (2020b) are the exception; elsewhere the resolution is stated as an implementation detail — 28 × 28 in STSTNet and OFF-ApexNet — with no indication of sensitivity, despite Xia et al. showing sensitivity strong enough to reverse a comparison.

**The evidence base is composite-database.** Both Xia et al. papers and STSTNet are developed and evaluated for the MEGC composite setting, which supplies substantially more training data per fold than a single-corpus study (§3.1.6), so conclusions about how much model capacity is affordable do not transfer directly here — if anything they understate the constraint.

**No study isolates the backbone.** Each paper proposes a backbone and evaluates the whole system. None reports what happens if the learned spatial stem is removed entirely and the motion representation is passed straight to a classifier or temporal model.

---

### 3.6.6 Implications for this research

The component ablated as Variable C in this thesis is a **three-stream shallow convolutional backbone**. Each of the three input channels — flow-$u$, flow-$v$, optical strain — is processed by its own unshared stream of two convolutional layers (16 then 32 channels, BatchNorm, ReLU, Dropout), followed by a spatial max-pool; the three outputs are concatenated channel-wise to give 96 feature maps.

| Finding from the review | Decision taken in this thesis |
|---|---|
| Shallow beats deep on small micro-expression data (Xia et al., 2020b); STSTNet uses 2 layers and ~1,670 parameters (Liong et al., 2019b) | Two convolutional layers per stream, **14,544 parameters** in the backbone (15,027 including the classifier head) — shallow by the corpus's standards, though roughly nine times STSTNet's 1,670 |
| Triple-stream design supplements small-scale input (Liong et al., 2019b) | Three unshared streams, one per input modality. Note the difference: STSTNet's streams differ in *filter count* over the same input, whereas these differ in *input channel* |
| Parameter-free modules add capability without capacity (Xia et al., 2020b) | Motivates the parameter-free attention evaluated as Variable B (§3.7) |
| Limited and imbalanced data require augmentation plus loss correction (Xia et al., 2020a) | Balanced sampling with focal loss; loss class-weighting deliberately disabled after single-class collapse (§3.1.8) |
| **Performance degrades above 100 × 100 input, dramatically for deeper models (Xia et al., 2020b)** | **Not followed.** This pipeline feeds the backbone at **224 × 224**, more than twice that threshold and eight times STSTNet's 28 × 28. See below |

*Table 3.14 — Review findings and the backbone decisions they determine.*

**A declaration about what this component actually is.** As §2.4.1 sets out, every kernel in this backbone is spatial only: no filter spans two time steps, and the temporal axis passes through unmixed. Implemented with `Conv3d` operations, the component is nevertheless a **purely spatial** feature extractor applied independently to every frame.

This has a direct consequence for how Chapter 5's result must be worded. The ablation does not test spatio-temporal convolution; it tests a **learned spatial stem**. The finding that Variable C contributes −0.031 macro F1 is therefore evidence against *this* stem — spatial-only, at 224 × 224, on flow and strain input — and **not** evidence that 3D convolution over time is unhelpful for micro-expression recognition, which this study never evaluated. Chapter 5 and Chapter 6 are worded accordingly.

Two mitigating observations, and one aggravating one, follow.

*In the field's own terms the choice is defensible.* STSTNet, the strongest shallow baseline on the CASME II subset, likewise performs no temporal convolution: its third kernel axis spans modalities, not time (§3.6.2). A spatial stem feeding a separate temporal encoder is a coherent division of labour, and it is precisely the division this thesis's factorial design is built to test.

*The stem is competing against a representation that has already done comparable work.* §3.3.7 and §3.4.7 argued that dense flow performs spatio-temporal feature extraction analytically, and that the strain channel is a hand-specified first-order spatial differential operator. A two-layer spatial CNN is being asked to add something on top of both, from 156 clips.

*The input resolution is the clearest unforced departure from the literature.* Xia et al. (2020b) report performance decreasing dramatically above 100 × 100 and shallow models improving as resolution falls toward 40 × 40; STSTNet and OFF-ApexNet both operate at 28 × 28. This pipeline uses 224 × 224. That single choice also explains the compute figures reported in Chapter 5 — the eight configurations containing this backbone consume 48.9 of the study's 50.6 GPU-hours, because the cost of convolving over a 224 × 224 × 32 volume dominates everything else. **The measured −0.031 may therefore be a resolution result as much as an architecture result**, and re-running the backbone arm at 28 × 28 or 56 × 56 is the single cheapest and most informative experiment left undone. Chapter 6 lists it first.

---

### 3.6.7 Limitations of the existing work, and how this study differs

The reviewed literature establishes that shallow convolutional backbones outperform deep ones on micro-expression data, that low input resolution helps and high resolution actively harms, that parameter-free additions are preferable to parameterised ones under sample scarcity, and that augmentation with loss rebalancing is necessary for imbalanced corpora. The four gaps set out in §3.6.5 — unreliable "3D-CNN" terminology, resolution left uncontrolled outside one study, no backbone ever isolated from the system built around it, and an evidence base that is almost entirely composite-database — define what that literature does not establish.

This thesis differs on the third count, which is the one its research question requires. It removes the convolutional backbone entirely in four of its twelve configurations — the matrix is unbalanced because the four cells pairing attention with no backbone are architecturally invalid (§3.1.3) — replacing it with a fixed 4 × 4 spatial pooling and a linear projection, and measures the difference under complete leave-one-subject-out with every other factor held constant — the matched-pair measurement the corpus does not contain. The result, that the backbone's mean marginal contribution is negative while it consumes 97 % of the compute budget, is a finding the existing literature is not positioned to produce. On the first count this thesis inherits the field's ambiguity and answers it by declaring precisely what its own component is (§3.6.6), so the negative result is not misread as a verdict on spatio-temporal convolution. On the second count it knowingly departs from the literature's guidance rather than extending it, at the 224 × 224 resolution already flagged above as the clearest such departure — identified only through this review, and the first item of proposed further work.

---

---

## 3.7 Parameter-Free Attention: SimAM

> **Scope of this section.** This section covers the attention module applied to the backbone's feature maps — switched on and off as Variable B of this thesis's ablation — through one paper that introduces the mechanism in full and a second that supplies the case for parameter-free modules in micro-expression recognition specifically. The competing attention modules discussed for contrast — SE, CBAM, ECA, GC and SRM — and the neuroscience findings the method rests on are **outside** the review corpus, and appear here only as they are reported in the reviewed paper.

---

### 3.7.1 Why attention, and why it must be free

Attention modules re-weight a network's feature maps so that informative regions contribute more to the representation than uninformative ones — for micro-expression recognition, a small, localised signal (a brow, a lip corner) against an otherwise quiescent frame. The obstacle is that attention normally costs parameters, already established as the scarcest resource in this problem (§3.1.7, §3.6.3). Yang et al. (2021) tabulate the cost of the standard modules, and it is not small:

| Module | Operators | Learnable parameters | Design |
|---|---|:--:|---|
| SE | GAP, FC, ReLU | $2C^2/r$ | handcrafted |
| CBAM | GAP, GMP, FC, ReLU, CAP, CMP, BN, C2D | $2C^2/r + 2k^2$ | handcrafted |
| GC | C1D, Softmax, LN, FC, ReLU | $2C^2/r + C$ | handcrafted |
| ECA | GAP, C1D | $k$ | handcrafted |
| SRM | GAP, GSP, CFC, BN | $6C$ | handcrafted |
| **SimAM** | GAP, /, ⊙, + | **0** | Eqn (5) |

*Table 3.15 — Attention modules compared, after Yang et al. (2021, Table 1). $C$ is the channel count and $r$ the reduction ratio. The Design column reproduces the source's own entries: every competitor is labelled "handcrafted", while SimAM's points to the energy equation it is derived from.*

On a 156-clip corpus a module costing $2C^2/r$ parameters is a material addition to a backbone that has fewer than fifteen thousand in total (§3.6.6). Xia et al. (2020b) reach the same conclusion independently from within micro-expression research itself, for the same reason (§3.6.4). A zero-parameter attention module is therefore not merely convenient here; it is the only kind whose cost is unambiguously justified.

The second column of Table 3.15 matters as much as the third. Every competing module is marked "handcrafted" — its structure is a design choice validated after the fact. SimAM's is described by Yang et al. (2021) as following from a defined objective, which they present as an advantage in its own right: "most of the operators are selected based on the solution to the defined energy function, avoiding too many efforts for structure tuning."

---

### 3.7.2 The premise: spatial suppression

SimAM's derivation begins from visual neuroscience rather than from architecture search. Yang et al. (2021) argue that "the most informative neurons are usually the ones that show distinctive firing patterns from surrounding neurons," and that "an active neuron may also suppress the activities of surrounding neurons, a phenomenon termed as spatial suppression." From this they take the operational rule that "the neurons displaying clear spatial suppression effects should be given higher priority (i.e., importance) in visual processing."

The engineering translation is that importance can be measured as *distinctiveness from context* — and the simplest way to measure that is "the linear separability between one target neuron and other neurons."

A second neuroscientific result determines *how* the resulting weights are applied. Because "attention modulation in mammalian brain typically manifests as a gain (i.e., scaling) effect on neuronal responses," Yang et al. "use a scaling operator rather than an addition for feature refinement."

---

### 3.7.3 The energy function and its closed-form solution

For a target neuron $t$ and the other neurons $x_i$ in the same channel, Yang et al. (2021) define an energy that is minimised when a linear transform separates the target from its context:

$$e_t(w_t, b_t, y, x_i) = (y_t - \hat{t})^2 + \frac{1}{M-1}\sum_{i=1}^{M-1}(y_o - \hat{x}_i)^2$$

with $\hat{t} = w_t t + b_t$ and $\hat{x}_i = w_t x_i + b_t$, and $M = H \times W$ the number of neurons in the channel.

Solving for $w_t$ and $b_t$ in closed form, and assuming all neurons in a channel share a distribution so that the mean and variance can be computed once and reused rather than recomputed per position, gives the minimal energy

$$e_t^* = \frac{4(\hat{\sigma}^2 + \lambda)}{(t - \hat{\mu})^2 + 2\hat{\sigma}^2 + 2\lambda}$$

where $\hat{\mu}$ and $\hat{\sigma}^2$ are the channel mean and variance. Yang et al. state the interpretation directly: "the lower energy $e_t^*$, the neuron $t$ is more distinctive from surround neurons, and more important for visual processing. Therefore, the importance of each neuron can be obtained by $1/e_t^*$."

Refinement is then $\tilde{X} = \text{sigmoid}(1/E) \odot X$, with the sigmoid present only "to restrict too large value in $E$" — and, because it is monotonic, it "will not influence the relative importance of each neuron."

The reason this matters practically is that $1/e_t^*$ simplifies to a form requiring only a channel mean, a channel variance and element-wise arithmetic. Yang et al. report that the solution "can be implemented in less than ten lines of code", and give the whole module as a PyTorch function:

```python
n = X.shape[2] * X.shape[3] - 1
d = (X - X.mean(dim=[2,3])).pow(2)
v = d.sum(dim=[2,3]) / n
E_inv = d / (4 * (v + lambda)) + 0.5
return X * sigmoid(E_inv)
```

Two details of this implementation are load-bearing. The variance divides by $n = HW - 1$ rather than $HW$, a Bessel-style correction. And the constant $+\,0.5$ is not a heuristic offset: it is what $1/e_t^*$ reduces to algebraically once the shared-distribution assumption is applied.

---

### 3.7.4 Three-dimensional weights

Yang et al.'s (2021) structural argument against the existing modules is that they attend along one axis at a time. Refinement "is usually operated along either the channel dimension or the spatial dimension. As a result, those methods generate 1-D or 2-D weights and treat neurons in each channel or spatial location equally, which may limit their capability of learning more discriminative cues."

SimAM instead produces a weight for every neuron — a full three-dimensional tensor over channel, height and width — at no parameter cost, because the weight is computed from the feature map's own statistics rather than predicted by a learned sub-network.

---

### 3.7.5 What the evidence shows

Yang et al. (2021) evaluate on CIFAR and ImageNet classification against SE, CBAM, ECA, GC and SRM, and report SimAM as competitive with or better than modules that cost parameters. They note in particular that it holds up on networks that are already large — remarking that one baseline "contains nearly 2.4 M parameters, which is even bigger than ResNet-110 with ∼1.17 M parameters," and demonstrating results on a WideResNet with "about 36 M free parameters", concluding that "the effectiveness of our parameter-free SimAM is not confined to some specific networks".

The single hyper-parameter $\lambda$ is swept from $10^{-1}$ to $10^{-6}$, five repetitions per value. Two conclusions are drawn: "our module can significantly boost performance using a wide range of $\lambda$", and "$\lambda = 10^{-4}$ provides a good balance between the top-1 accuracy and the standard deviation." That value is adopted for all CIFAR networks. Notably, for ImageNet they "still employ cross-validation to find a good $\lambda$" rather than transferring the CIFAR value — an indication that the authors themselves do not treat $10^{-4}$ as universal.

---

### 3.7.6 Limitations

**The evidence base is generic large-scale vision, not micro-expression recognition.** Every result in Yang et al. (2021) comes from CIFAR or ImageNet classification, with millions of training images against this corpus's few hundred clips (§3.1.3). Nothing in the paper speaks to behaviour at that scale, where the statistics from which SimAM computes its weights are estimated from far fewer and far more homogeneous feature maps.

**The input is appearance, not motion.** SimAM's premise is that informative neurons are those distinctive from their spatial surroundings. On a natural image that is a strong prior. On an optical-flow map — where large regions are legitimately near-zero because nothing moved, and the non-zero region is the signal — the prior may be either unusually well matched or subtly wrong, and no reviewed work tests which.

**$\lambda$ is dataset-tuned, and the paper says so.** The widely reused $10^{-4}$ was selected on CIFAR, and the authors re-tuned it for ImageNet.

**There is no temporal dimension in the original formulation.** The module is defined over a channel's $H \times W$ plane. Any application to video requires a decision about what constitutes "surrounding neurons" across time, and the paper offers no guidance because the question does not arise for it.

---

### 3.7.7 Implications for this research

The component ablated as Variable B is a **5-D adaptation of SimAM**, applied independently to each of the three convolutional streams (§3.6.6) immediately before their outputs are concatenated.

| Finding from the review | Decision taken in this thesis |
|---|---|
| Attention normally costs $2C^2/r$ or more parameters; SimAM costs 0 (Yang et al., 2021) | SimAM chosen over SE/CBAM/ECA specifically because it adds **zero learnable parameters**, the only defensible choice at ~15 k backbone parameters (§3.6.6) |
| Parameter-free modules are the right trade in micro-expression recognition (Xia et al., 2020b) | Corroborates the choice from within the task domain rather than from generic vision |
| $\lambda = 10^{-4}$ balances accuracy against variance (Yang et al., 2021) | $\lambda = 10^{-4}$ adopted unchanged. **Not re-tuned for this corpus**, although the authors re-tuned it when moving from CIFAR to ImageNet — declared as an untested inheritance |
| Refinement by scaling, gated by a monotonic sigmoid (Yang et al., 2021) | Implemented exactly as specified: `x * sigmoid(energy)` |
| Statistics computed over a channel's $H\times W$ plane with an $n = HW-1$ correction (Yang et al., 2021) | Extended to $n = D \times H \times W - 1$ over the spatio-temporal volume — see the declaration below |
| SimAM requires a feature map to attend over (Yang et al., 2021) | Creates the architectural dependency that prunes 4 of the 16 factorial cells (§3.1.3): attention without a backbone is undefined |

*Table 3.16 — Review findings and the attention decisions they determine.*

**A declaration about the 5-D extension.** The implementation reproduces the reference module's arithmetic exactly — the same $n-1$ correction, the same $d / (4(v + \lambda)) + 0.5$ energy, the same sigmoid gate — but computes the channel mean and variance over the **three-dimensional** volume $(D, H, W)$ rather than the two-dimensional plane $(H, W)$, so that $M = DHW$ rather than $HW$.

This is a faithful generalisation of the formula, and it is also a substantive modelling choice that the source paper does not make and does not discuss. Under the 2-D formulation each frame would be normalised against itself, so a neuron's importance would be judged relative to its own frame. Under the 3-D formulation a neuron is judged relative to the entire clip. The consequence is that frames in which nothing happens — the neutral onset and offset regions that every CASME II clip contains by design (§3.1.2) — receive uniformly low attention, while the apex region stands out against the clip as a whole. For a phenomenon defined by a brief peak against a neutral baseline, that is arguably the more appropriate behaviour, but it is an argument rather than a result: no experiment in this thesis compares the 2-D and 3-D pooling variants.

**What the measurement gives.** Across the four matched pairs of the ablation, SimAM's mean marginal contribution is **+0.003 pooled macro F1** — the metric defined in §3.1.6, and worth naming explicitly here, because the same four pairs computed on the mean-of-folds column give −0.004 and would reverse the sign of the conclusion — statistically indistinguishable from zero at this sample size, and consistent with §3.7.6's caution that a module validated on ImageNet need not transfer to 156 clips of optical flow. Two observations qualify that null result and are reported in Chapter 5 alongside it.

The first is that a zero mean effect from a zero-parameter module is not the same finding as a zero mean effect from a costly one. SimAM adds no parameters, negligible computation, and no risk of overfitting; a component that is free and neutral has a different disposition than one that is expensive and neutral, and Chapter 6 recommends retaining it wherever the backbone survives.

The second is that the average conceals a large effect in one place. In the configuration with the weakest Surprise-class performance among those where attention is applicable at all — the four that carry a convolutional backbone — adding SimAM raises the Surprise-class F1 from **0.300 to 0.590**. (One backbone-free configuration scores lower still at 0.246, but SimAM cannot be applied there, so it is not part of any matched pair.) That the largest observed benefit falls on the rarest class, in the configuration that handles it worst, is at least consistent with the module's stated premise — that it up-weights neurons distinctive from their context — although with 25 Surprise clips in total this is a single observation and is reported as such.

---

### 3.7.8 Limitations of the existing work, and how this study differs

The reviewed work establishes that attention weights can be derived in closed form from a neuroscience-motivated energy function rather than learned, that doing so costs no parameters where competing modules cost $2C^2/r$ or more, that the resulting three-dimensional weights are competitive with or better than one- and two-dimensional learned alternatives on large-scale image classification, and that the single hyper-parameter is robust across five orders of magnitude. The three gaps set out in §3.7.6 — evidence confined to generic large-scale vision rather than micro-expression data, a distinctiveness prior validated on appearance rather than motion, and no temporal dimension in the original formulation — are the ones this thesis must respond to.

This thesis contributes to the first and third of these. It measures SimAM as an isolated factor in a full factorial ablation under complete leave-one-subject-out on a 156-clip corpus of motion tensors — a matched-pair measurement in exactly the small-data, motion-input regime the source evidence does not cover — and reports a mean effect of +0.003 with a large single-class exception. And it makes the temporal-context decision explicitly, pooling statistics over the whole spatio-temporal volume so that neurons are judged against the clip rather than the frame, with the reasoning stated in §3.7.7. It does not close the second gap: the module is applied to flow-derived feature maps without any test of whether its distinctiveness prior suits that distribution, and $\lambda$ is inherited at $10^{-4}$ without re-tuning despite the source authors re-tuning it across datasets. Both are recorded in Chapter 6, together with the untested 2-D-versus-3-D pooling variant, as the natural follow-up experiments — all three of which are cheap, since the module has no parameters to train.

---

---

## 3.8 The Temporal Encoder: Transformers for Micro-Expression Recognition

> **Scope of this section.** This section covers the component that Chapter 5 identifies as responsible for essentially all of this study's measured performance: the self-attention encoder applied over the temporal axis, switched as Variable D. Two papers are covered — SLSTT, the architecture this component is named after, and the ViT it derives from — with the substantial divergence between published SLSTT and this implementation set out in full in §3.8.6.

---

### 3.8.1 Why self-attention should suit this problem

Given the temporal structure established in §3.1.1–§3.1.2 — a movement with an annotated onset, apex and offset — a model that can relate any frame to any other frame is matched to the phenomenon in a way that a local operator is not.

Zhang et al. (2022) frame the gap this addresses: "the problem of capturing both local and global spatio-temporal patterns remains challenging." Their solution is "the first purely transformer based approach (i.e. void of any convolutional network use) for micro-expression recognition," comprising "a spatial encoder which learns spatial patterns, a temporal aggregator for temporal dimension analysis, and a classification head."

The alternative this thesis measures against is mean pooling over time — an operator with no parameters and no notion of order, which averages the apex into the baseline. The contrast between the two is the whole of Variable D.

---

### 3.8.2 The transformer's route into vision

Dosovitskiy et al. (2021) established the mechanism SLSTT builds on. An image is split into fixed-size patches; each flattened patch $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $N = HW/P^2$ is both the patch count and "the effective input sequence length for the Transformer", is mapped to $D$ dimensions "with a trainable linear projection"; learnable 1-D position embeddings are added; and the resulting sequence is fed to a standard transformer encoder. Patches are "treated the same way as tokens (words) in an NLP application."

The paper's central caveat matters more here than its headline result. Dosovitskiy et al. report that "when trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies," and give the reason: "Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data." Their resolution is scale — "large scale training trumps inductive bias" — with strong results obtained only after pre-training on ImageNet-21k or JFT-300M.

Taken at face value this is a discouraging prior for a 156-clip corpus, and §3.8.6 returns to why the result reported in Chapter 5 does not in fact contradict it.

---

### 3.8.3 SLSTT: architecture

Zhang et al. (2022) assemble three components.

**Long-term optical flow.** Before any network is applied, they change how the motion input is computed, and the argument is worth quoting because it bears directly on §3.3. Optical flow "is inherently temporally local, i.e. … it is computed between consecutive frames of sequence," which they identify as a problem for micro-expressions: consecutive-frame flow fields "are rather similar up to the apex frame … with a similar trend thereafter but in the opposite direction." They therefore "calculate optical flow between each sample frame and the onset frame instead of consecutive frames." The resulting long-term field "exhibits a much more structured pattern, always being in the same direction, increasing in magnitude up to the apex frame and declining in magnitude thereafter," which they argue yields "much more stable and discriminative features."

**A ViT spatial encoder.** Each long-term flow field is treated as a sequence of patches and passed through transformer encoder layers for "long-term spatial feature extraction". Implementation uses "the official ViT-B/16 model pre-trained on ImageNet", with 12 encoder layers, and inputs resized to 384 × 384. Note that in SLSTT the transformer's role is **spatial**: it relates patches within a frame.

**An LSTM temporal aggregator.** Temporal relations are handled separately. Zhang et al. reason that "since facial movement during micro-expressions is almost imperceptible, all frames from a single video sample are rather similar one to another," yet "it is still possible to identify reliably a number of salient frames, such as the apex frame," and "therefore, we propose an LSTM architecture for temporal aggregation," whose inputs are the per-frame spatial encoder outputs.

---

### 3.8.4 SLSTT: results, and the one internal ablation that matters here

Evaluated under the composite-database protocol with LOSO and MEGC's UF1/UAR metrics (§3.1.6), SLSTT reports the strongest CASME II figures in the review corpus:

| Method | Composite UF1 | CASME II UF1 | CASME II UAR |
|---|:--:|:--:|:--:|
| LBP-TOP | 0.588 | 0.703 | 0.743 |
| Bi-WOOF | 0.630 | 0.781 | 0.803 |
| OFF-ApexNet (2019) | 0.720 | 0.876 | 0.868 |
| STSTNet (2019) | 0.735 | 0.838 | 0.869 |
| EMR (2019) | 0.789 | 0.829 | 0.821 |
| RCN (2020) | 0.705 | 0.809 | 0.856 |
| **SLSTT-Mean** | 0.788 | 0.844 | 0.830 |
| **SLSTT-LSTM** | **0.816** | **0.901** | **0.885** |

*Table 3.17 — Composite-database evaluation, after Zhang et al. (2022, CDE results table). Selected rows.*

Zhang et al. describe this as "the first framework in the published literature on micro-expression recognition to achieve the unweighted F1-score greater than 0.9 on any of the aforementioned data sets."

**The two SLSTT rows constitute a direct ablation of a decision this thesis also had to make.** SLSTT-Mean aggregates the per-frame features by averaging; SLSTT-LSTM aggregates them recurrently. The difference on CASME II is **0.844 against 0.901 UF1** — 5.7 points from the temporal aggregator alone, with the spatial encoder, the input representation and the protocol all held fixed. This thesis uses mean aggregation (§3.8.6), so that measurement is a direct estimate of what is being left on the table.

---

### 3.8.5 Limitations

**The evidence rests on large-scale pre-training.** SLSTT's spatial encoder is ViT-B/16 pre-trained on ImageNet. Dosovitskiy et al. (2021) are explicit that transformers "do not generalize well when trained on insufficient amounts of data", and SLSTT's design conforms to that finding rather than contradicting it. Nothing in the corpus establishes what a transformer contributes when trained from scratch on micro-expression data alone.

**Results are composite-database** — trained under the richer regime §3.1.6 describes, not on a single corpus fold — so a CASME II UF1 of 0.901 is not a single-corpus result.

**The transformer's role is spatial, not temporal.** In SLSTT self-attention relates patches *within* a frame while an LSTM relates frames. A reader who takes "spatio-temporal transformer" to mean attention over time has misread the architecture — and §3.8.6 declares that this thesis inverts the arrangement.

**Label noise bounds what any architecture can achieve.** Zhang et al. make an observation the field rarely states: because "the mapping between facial action unit activations and emotions … is not a bijection", and because only visual data is used, "the theoretical highest accuracy of automated micro-expression recognition on the MER corpora currently used for research purposes is not 100%."

---

### 3.8.6 Implications for this research, and a full declaration of divergence

The component ablated as Variable D is a **pre-norm transformer encoder applied to the temporal axis**: the sequence of 32 per-frame feature vectors produced by the spatial stem (§3.6.6) or by the fixed 4 × 4 pooling that replaces it, with $d_{\text{model}} = 96$, 8 heads, 4 layers, feed-forward width 256, dropout 0.1, fixed sinusoidal positional encoding, and mean pooling over the output sequence. When the switch is off, the same sequence is collapsed by plain mean pooling over time.

**It is named for SLSTT, and it differs from SLSTT in six material respects.** These are set out in full because the component carries this study's headline result, and because a reader who assumes the published architecture was reproduced would be misled.

| | Published SLSTT (Zhang et al., 2022) | This thesis |
|---|---|---|
| **What attention operates over** | Patches within a frame — a **spatial** encoder | Frames within a clip — a **temporal** encoder |
| **Temporal aggregation** | LSTM | Mean pooling (SLSTT's own weaker variant) |
| **Initialisation** | ViT-B/16 pre-trained on ImageNet | Trained from scratch |
| **Capacity** | 12 encoder layers, hidden size 768, input 384 × 384 | 4 layers, $d_{\text{model}} = 96$, 32 tokens |
| **Positional encoding** | Learnable 1-D embeddings (ViT) | Fixed sinusoidal |
| **Input flow** | Long-term: each frame against the **onset** frame | Short-term: each frame against its **predecessor** |

*Table 3.18 — Divergences between the published SLSTT and the component evaluated here.*

The honest description of Variable D is therefore **a small transformer encoder over the temporal axis, trained from scratch**, not a reproduction of SLSTT. Chapter 5's conclusion should be read accordingly: it establishes that self-attention over time is decisive *in this pipeline*, not that SLSTT was validated.

**Why the ViT small-data warning does not apply straightforwardly.** Dosovitskiy et al.'s (2021) argument is that transformers lack *locality and translation equivariance*, inductive biases that matter when modelling the two-dimensional spatial structure of images from limited data. The component here does no spatial modelling at all: it receives a 32-token sequence of 96-dimensional vectors and models only the order among them. The sequence is short, one-dimensional, and has strong known structure — rise, peak, decay — that the corpus's onset/apex/offset annotation guarantees is present and correctly delimited. That is a far easier problem than learning image structure from scratch, which is a plausible reason why a component the vision literature warns against under-performing on small data is the one component here that clearly works.

**What the measurement gives.** Across six matched pairs, the transformer's mean marginal contribution is **+0.217 pooled macro F1**, positive in every pair, with a minimum of +0.158. The configurations containing it score 0.583–0.712 and those without it 0.419–0.448, two ranges separated by an empty gap of 0.135. Against components whose measured effects are +0.015, +0.003 and −0.031 (§3.2, §3.7, §3.6), this is not a marginal difference in degree.

The mechanism is the one §3.8.1 anticipated. Self-attention is the only component in the study that sees all 32 frames simultaneously and can relate an early frame to a late one directly. Every alternative collapses the temporal axis by averaging, which removes the apex — the single most informative frame, as both Liong et al. (2019a) and Zhang et al. (2022) argue — into the neutral baseline surrounding it.

**Three cheap improvements follow directly from the divergence table**, and Chapter 6 lists them in this order. Replacing mean pooling with an LSTM aggregator is worth 5.7 UF1 points in SLSTT's own ablation (§3.8.4). Recomputing the input as long-term onset-referenced flow rather than consecutive-frame flow is a Stage 1 change costing no training time and is the modification Zhang et al. argue most strongly for. And the fixed sinusoidal encoding could be replaced with learned embeddings at a cost of $32 \times 96$ parameters. None of the three was tested here, and all three are independent of the ablation's conclusions.

---

### 3.8.7 Limitations of the existing work, and how this study differs

The reviewed work establishes that a purely transformer-based architecture can outperform convolutional approaches, that onset-referenced flow yields more discriminative motion fields than consecutive-frame flow, that recurrent aggregation outperforms mean aggregation by a measurable margin, and that transformers need either large-scale pre-training or convolution's inductive biases to generalise from limited data (§3.8.2–§3.8.4). Three gaps remain. First, no result in the corpus isolates the transformer as a factor — the closest is SLSTT's own Mean-versus-LSTM comparison, which varies the aggregator while keeping the transformer in both arms. Second, every transformer result in the corpus depends on ImageNet pre-training and composite-database training, so the field has no evidence about self-attention trained from scratch on a single small corpus. Third, the transformer is applied to the spatial axis with a recurrent network handling time, and no reviewed work tests the inverse arrangement in which attention models the temporal axis directly.

This thesis addresses all three, at the cost of not reproducing the published architecture. It measures the temporal encoder as an isolated binary factor across six matched pairs under complete leave-one-subject-out, with every other component held constant — the ablation the corpus does not contain (§3.8.6). It does so **from scratch on a single 156-clip corpus**, with no pre-training of any kind, which is precisely the regime the vision literature identifies as unfavourable to transformers and in which the corpus offers no prior evidence. And it applies self-attention to the temporal axis rather than the spatial one, so the arrangement tested is the inverse of SLSTT's. What it does not do is validate SLSTT: the six divergences of Table 3.18 (§3.8.6) are large enough that the result should be read as a finding about temporal self-attention in this pipeline, not as a validation of the published architecture.

---

---

## 3.9 Training Under Severe Class Imbalance

> **Scope of this section.** The four ablated components are covered in §3.2, §3.6, §3.7 and §3.8. This section covers the training configuration that is held constant across all twelve of them, and which §3.1.8 identified as forced by the corpus rather than chosen freely: the handling of a 4 : 1.3 : 1 class imbalance.
>
> **A deliberate exclusion.** The project's reading list groups two further topics under this heading — Grad-CAM++ visual auditing and adversarial identity disentanglement. Neither is part of the implemented system. The codebase states this explicitly: the loss module records that "the Cross-Batch Memory (XBM) queue, and the adversarial identity loss — are intentionally removed", and the trainer likewise removes the adversarial trainer of an earlier project stage. No gradient-based visualisation is implemented anywhere. Following the principle applied throughout this chapter, those papers are therefore **not reviewed**, because reviewing techniques the system does not use would misrepresent what was built. They are noted in Chapter 6 as proposed future work.

---

### 3.9.1 The imbalance is a property of the phenomenon, not of the sampling

§3.1.3 and §3.1.4 established the working-set class counts and the resulting always-Negative floor; what this section adds is why that skew exists at all. It is not an artefact of careless collection: Yan et al. (2014) are explicit that it follows from what can be elicited in a laboratory, noting "some types of facial expressions are difficult to elicit in laboratory situations, thus the samples in different categories distributed unequally, e.g., there are 60 disgust samples but only 7 sadness samples." Nor is it specific to CASME II — the MEGC composite exhibits the same pattern (§3.1.6). Any method for this task must cope with skew; it cannot be collected away.

---

### 3.9.2 Three places to intervene

The reviewed corpus corrects for imbalance at three distinct points in the pipeline, and it is useful to separate them because this thesis intervenes at all three and the interaction between two of them turned out to matter.

1. **At the data**, by changing how often each class is presented during training — resampling or augmentation.
2. **At the loss**, by changing how much each example contributes to the gradient — class weighting or focal re-weighting.
3. **At the metric**, by changing how performance is scored — macro-averaged rather than sample-averaged measures.

---

### 3.9.3 Loss-level correction: focal loss

Zhao et al. (2021) bring focal loss into micro-expression recognition, introducing it "to alleviate the inefficient model training caused by the class imbalance in the MEs Datasets". Their motivation is stated plainly: cross-entropy is "usually used for back propagation to update model parameters. However, there is an imbalanced distribution of samples in the spontaneous MEs datasets. This could be biased toward particular emotions that constitute a larger portion of the training set. Therefore, applying a fairer loss function is critical."

They adopt the formulation developed for one-stage object detection, where the imbalance is between foreground and background regions:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma}\log p_t$$

where $\alpha_t$ is "the weight balance factor for samples", $\gamma$ is "the balance factor for loss", and $p_t$ is the model's probability for the true class. The mechanism is set out in §2.5.2; under skew, the examples it up-weights are disproportionately the minority classes.

Zhao et al. adapt it to multi-class classification and report two parameter choices: "$\gamma$ is set as 2 in practice, and $\alpha$ is treated as a hyper-parameter to set by cross validation."

Note the division of labour in that sentence. $\gamma$ controls *difficulty* re-weighting and is treated as a constant; $\alpha$ controls *class-frequency* re-weighting and is treated as dataset-specific. The two do different jobs, and §3.9.7 shows that conflating them is where this project went wrong.

---

### 3.9.4 Data-level correction: resampling and augmentation

Xia et al. (2020a) address the same problem at the data end, and pair the two interventions. Their stated aim is "to overcome the shortcomings of limited and imbalanced training samples", for which "temporal data augmentation strategies as well as a balanced loss are jointly used for our deep network" — augmentation to enrich a small dataset, and a balanced loss "for counterweighing imbalanced classes".

The word **jointly** is the one to note. Xia et al. apply a data-level and a loss-level correction simultaneously and report no difficulty arising from the combination. §3.9.7 records that this thesis did encounter one.

---

### 3.9.5 Metric-level correction

The third intervention is metric-level; §3.1.6 defines UF1 and the case for macro-averaged metrics in full. Liong et al. (2016) add the argument from the subject side rather than the class side, distinguishing macro- from micro-averaged accuracy under LOSO and reporting F1, precision and recall because these "provide a more meaningful perspective than accuracy rates when the datasets used are naturally imbalanced since each subject has a different number of video samples." The two arguments compound on CASME II, where classes are skewed **and** per-subject clip counts range from 1 to 33 (§3.1.7c).

---

### 3.9.6 Limitations

**No study reports the interaction between corrections.** Xia et al. (2020a) apply augmentation and a balanced loss together; Zhao et al. (2021) apply focal loss without resampling. Nothing in the corpus examines whether combining a data-level and a loss-level correction over-corrects — which is the failure this project encountered and documents in §3.9.7.

**$\gamma = 2$ is inherited rather than tuned.** Zhao et al. adopt the value from the object-detection literature "in practice", without a sweep on micro-expression data. Every subsequent use in this project inherits it.

**No ablation of focal loss against cross-entropy exists in the corpus.** Zhao et al. introduce it as one component of a larger system and do not isolate it, so the field has no measurement of what focal loss is worth on this task.

**Label smoothing is absent from the corpus entirely.** It is used in this thesis (§3.9.7) but is not discussed in any reviewed paper, and is therefore declared as an out-of-corpus choice rather than a literature-supported one.

---

### 3.9.7 Implications for this research

| Finding from the review | Decision taken in this thesis |
|---|---|
| Focal loss addresses class-frequency bias in MER (Zhao et al., 2021) | Focal loss adopted as the training objective, with **$\gamma = 2.0$**, matching Zhao et al.'s reported setting |
| $\alpha$ is a dataset-specific class-weight term (Zhao et al., 2021) | Implemented as an optional per-class $\alpha$ vector, **switched off at run time** — see the declaration below |
| Balanced presentation of classes during training (Xia et al., 2020a) | Minority oversampling via a weighted sampler, active in every configuration |
| Imbalance-aware metrics are mandatory (See et al., 2019; Liong et al., 2016) | Pooled macro F1 — identical in construction to MEGC's UF1 — as the primary metric, with accuracy quoted only alongside the always-Negative floor |
| *(not from the corpus)* | Label smoothing of 0.05 applied inside the focal loss. **No reviewed paper uses or discusses this**; it is an out-of-corpus choice and is declared as such |

*Table 3.19 — Review findings and the imbalance-handling decisions they determine.*

**The finding this project contributes: two corrections are one too many.** The configuration is resolved at run time by a single rule — class weighting in the loss is enabled only when the balanced sampler is *not* active — and the run log records the decision explicitly each fold: *"Class weights in loss disabled (balanced sampler already active)."*

That rule exists because of a failure. In this project's earlier runs, inverse-frequency class weights were applied in the loss **and** no balanced sampler was used; in a later configuration both were active. The documented consequence was that the proposed model predicted **zero** Positive clips — an entire minority class abandoned, scoring 0.000 F1 on it — which was initially misdiagnosed as over-capacity overfitting. With the sampler active and loss weighting stood down, the same configuration recovers 15 of 32 Positive clips and no configuration in the final study abandons any class (§3.1, Chapter 5).

This is worth stating as a contribution rather than an incident, because §3.9.6 established that the corpus offers no guidance on it: Xia et al. (2020a) apply both corrections jointly and report no problem, and no reviewed paper warns that doing so can invert the intended effect. On a 156-clip corpus with a 4 : 1 skew, over-correction is evidently as damaging as no correction, and the safe configuration is to correct at exactly one point in the pipeline.

**A second declaration.** The severity of the imbalance also constrains what the ablation can conclude. Because every configuration shares the same sampler, loss and $\gamma$, this thesis measures **no** effect attributable to the imbalance treatment itself — it is a constant, not a factor. The claim supported is that the four architectural components were compared under a training regime that does not abandon minority classes, not that this regime is optimal.

---

### 3.9.8 Limitations of the existing work, and how this study differs

The reviewed literature establishes that spontaneous micro-expression corpora are irreducibly imbalanced because the imbalance originates in what can be elicited rather than in how data was gathered; that the imbalance biases cross-entropy training toward majority classes; that focal loss and balanced losses are effective responses at the loss level; that augmentation and resampling are effective at the data level; and that accuracy must be replaced by macro-averaged metrics at the evaluation level. Three gaps remain. First, the corrections are never composed under controlled conditions: one paper applies a data-level and a loss-level correction jointly without examining the combination, another applies only a loss-level correction, and no study reports what happens when both are applied to a corpus as small and as skewed as this one. Second, focal loss is introduced into micro-expression recognition as a component of a larger system and never isolated, so its contribution over plain cross-entropy on this task is unmeasured, and its $\gamma$ is imported from object detection without validation. Third, no reviewed work reports a failure mode from over-correction, so a practitioner following the literature has no warning that combining two standard remedies can suppress a minority class entirely.

This thesis contributes to the first and third. It documents an over-correction failure in which combining inverse-frequency loss weighting with a balanced sampler drove the proposed model to predict none of the 32 Positive clips, and it reports the resolution — correcting at exactly one point in the pipeline, enforced by an explicit run-time rule — together with the recovery that followed. That is a negative result the corpus does not contain, and it is directly actionable for anyone reproducing this class of pipeline. On the second gap it offers no improvement: focal loss with $\gamma = 2.0$ and label smoothing of 0.05 is held constant across all twelve configurations, so this study measures neither focal loss against cross-entropy nor the value of either hyper-parameter. Both are recorded in Chapter 6 as cheap additions to the ablation matrix, alongside the Grad-CAM++ auditing and identity-disentanglement work that this project's earlier stages removed and that this section therefore excludes.

---

---

## 3.10 Synthesis: What the Literature Establishes, What It Leaves Open, and Where This Thesis Sits

> **Purpose of this section.** This section consolidates the gap statements closing §3.1–§3.9: what the corpus collectively establishes, the four limitations recurring across every topic, the resulting research gap, and precisely which parts of it this thesis closes and which it does not. It also collects, in one place, every declared divergence between the system built here and the literature it draws on.

---

### 3.10.1 What the corpus establishes

Read as a whole rather than topic by topic, the reviewed literature settles a good deal.

**On data.** Spontaneous micro-expression corpora are small, demographically narrow, laboratory-bound and irreducibly imbalanced by what can be elicited rather than by how collection was conducted, and CASME II remains the primary single-corpus benchmark under its native leave-one-subject-out protocol (§3.1, §3.9).

**On representation.** Motion rather than appearance is the appropriate input; amplifying it before extraction, deriving a strain field from it, and normalising unequal clip durations by manifold rather than polynomial interpolation each carry an independently supported interior optimum (§3.2–§3.5).

**On architecture.** Shallow, low-resolution, parameter-free capability outperforms capability bought with parameters on data of this size, and self-attention is the strongest-performing mechanism in the corpus (§3.6–§3.8).

**On evaluation.** Accuracy alone is inadequate under skew and must be replaced by macro-averaged measures accumulated across folds before averaging (§3.1.6, §3.9.5).

---

### 3.10.2 Four limitations that recur across every topic

The per-section gap statements are not nine independent observations. Four structural problems generate most of them.

**(a) No component is ever isolated.** Every paper in the corpus proposes a complete system and evaluates it against other complete systems. STSTNet is compared with OFF-ApexNet, not with STSTNet-minus-its-backbone. SLSTT is compared with STSTNet, not with SLSTT-minus-its-transformer. Where an internal ablation exists it varies one part while holding the rest of the proposed design fixed — SLSTT's Mean-versus-LSTM comparison keeps the transformer in both arms (§3.8.4). The consequence is that the field has no matched-pair measurement of what any of its standard components contributes over an otherwise identical pipeline, and therefore no basis for deciding which of them are load-bearing.

**(b) The evaluation protocol is unstable, and under-reported.** Sample counts, label-set size, and the choice among leave-one-subject-out, leave-one-video-out and leave-one-sample-out protocols all vary across the corpus in ways that are inconsistently reported — even the two primary sources for the field's most-quoted baseline contradict each other about which protocol was used (§3.1.4, §3.1.9). Given the fold composition this corpus actually produces (§3.1.7), these are not bookkeeping differences: they change what the numbers mean.

**(c) The strongest results depend on regimes a single-corpus study cannot access.** The best CASME II figures in the corpus come from composite-database training (§3.1.6) and, in SLSTT's case, ImageNet pre-training (§3.8.5) — neither a property of the architecture being credited. The literature offers almost no evidence about what any of these components does when trained from scratch on one small corpus, which is the situation most practitioners are actually in.

**(d) Preprocessing parameters have interior optima, and are transferred without re-validation.** The magnification factor, the interpolated sequence length and the input resolution all exhibit rise-then-fall behaviour, all have their optima located only by exhaustive sweep, and none has a principled selection rule (§3.2.3, §3.5.4, §3.6.3). Values are nonetheless carried between pipelines whose downstream models differ entirely — α = 30 moves from a single-frame appearance encoder to other settings, TIM10 propagates from one baseline paper to the whole field, and λ = 10⁻⁴ transfers from CIFAR even though its own authors re-tuned it for ImageNet (§3.7.5).

---

### 3.10.3 The research gap

Taking (a)–(d) together:

> **The field has established a standard pipeline for micro-expression recognition — magnify, compute motion, normalise duration, extract spatially, model temporally — but has never measured which of its stages are responsible for the result. Because components are only ever evaluated inside complete systems, under protocols that vary in unstated ways, and predominantly in training regimes richer than a single small corpus provides, it is not currently possible to say which parts of that pipeline earn their place and which are inherited convention.**

This matters practically as well as scientifically. The corpus's own evidence shows that the most computationally expensive stage is not obviously the most valuable, that two components with near-zero average effect nevertheless rescue a specific failure, and that the choice of evaluation protocol can reverse a ranking. A practitioner assembling this pipeline today has no basis for allocating effort.

---

### 3.10.4 How this thesis addresses the gap — and what it does not

**What it does.** This thesis treats the pipeline as a factorial experiment rather than a proposal: four components varied independently across every architecturally valid cell of a 2⁴ matrix, twelve configurations, each evaluated under **complete 25-fold leave-one-subject-out on CASME II** with every non-varied factor held identical. That gives the matched-pair measurement (a) says the corpus lacks; reporting the protocol in full addresses (b) for this study at least; and training from scratch with no pre-training is the regime (c) identifies as unevidenced.

The result is a ranking with an unusually large separation:

| Component | Mean effect (pooled macro F1) | Matched pairs | Consistency |
|---|:--:|:--:|---|
| **SLSTT Transformer** | **+0.217** | 6 | positive in **6 of 6**, min +0.158 |
| EVM | +0.015 | 6 | positive in 4 of 6, range −0.054 to +0.080 |
| SimAM | +0.003 | 4 | positive in 3 of 4, range −0.029 to +0.034 |
| **3D-CNN** | **−0.031** | 4 | positive in 2 of 4, min −0.129 |

*Table 3.20 — Marginal contribution of each ablated component, measured from matched pairs under 25-fold LOSO, N = 156.*

One component accounts for essentially the entire effect; one is actively harmful on average while consuming 97 % of the compute; two are indistinguishable from noise on average while each rescuing a specific minority-class failure. That distribution is the substantive answer to the gap, and it is not one that could have been obtained without the factorial design.

Beyond the ranking, three further contributions follow from the review rather than from the experiment:

- **A protocol-instability result.** The same twelve configurations, evaluated under five successive protocols across this project's life, produce **four different winning configurations**, and the two extremes are architecturally opposite — SimAM + 3D-CNN with no transformer under holdout, the transformer alone under full LOSO (§3.1). This is direct evidence for limitation (b) and, on a corpus this size, arguably a more transferable finding than the accuracy figure.
- **A repaired measurement.** The EVM switch was inert in every earlier run of this project, producing bit-identical results across all six matched pairs; the review's evidence that magnification has a clear, repeatable effect is what made that detectable as a defect rather than a null result (§3.2.7).
- **An over-correction failure.** Combining loss-level and data-level imbalance correction suppressed an entire minority class, a failure mode no reviewed paper reports despite one applying both corrections jointly (§3.9.7).

**What it does not do.** Three of the four limitations are only partially addressed, and one is not addressed at all. On (b), this study reports its own protocol completely but cannot repair the field's inconsistency, and its own N = 156 differs from MEGC's 145. On (c), it establishes what these components do from scratch on one corpus, but that is a different question from — not an answer to — how they behave under composite training, and no cross-dataset claim is made anywhere. On (d), the thesis inherits parameter values from the literature rather than sweeping them: α = 10, T = 32, λ = 10⁻⁴ and γ = 2.0 are all fixed throughout, so the ablation measures architecture at one point in preprocessing space, not across it.

---

### 3.10.5 A consolidated ledger of declared divergences

Reviewing the literature closely enough to write this chapter surfaced ten points at which the implemented system departs from the work it draws on. Several were discovered only through the review. They are collected here so that no result in Chapter 5 rests on an unstated assumption.

| § | Divergence | Consequence |
|---|---|---|
| 3.1.3 | Working set is N = 156, including sadness and fear in Negative; MEGC's CASME II subset is N = 145 | Comparisons against challenge figures are not exactly like-for-like |
| 3.4.3 | Strain magnitude counts shear once, not twice as in the published Frobenius norm | Shear weighted lower by √2 relative to normal strain; untested |
| 3.4.7 | No dedicated strain filtering, where the literature applies Wiener and Gaussian filters | Strain channel is noisier than in the reviewed work |
| 3.5.7 | Temporal normalisation is uniform index sampling, **not** manifold-based TIM; no frames are synthesised | Avoids fabricating motion; forfeits the ability to lengthen short clips |
| 3.5.7 | Magnification is applied **after** subsampling, at a nominal 200 fps | Realised pass-band is clip-dependent and below the intended 5–25 Hz |
| 3.6.6 | Backbone kernels are (1, 3, 3) — **no temporal mixing** | The ablation tests a learned *spatial* stem, not spatio-temporal convolution |
| 3.6.6 | Backbone input is 224 × 224, against literature guidance of ≤ 100 × 100 | The −0.031 result may be a resolution effect as much as an architecture effect |
| 3.7.7 | SimAM statistics pooled over (D, H, W), not the paper's (H, W); λ not re-tuned | Neurons judged against the clip, not the frame; defensible but untested |
| 3.8.6 | Six divergences from published SLSTT (temporal not spatial attention, mean not LSTM aggregation, from scratch, smaller, sinusoidal encoding, short-term flow) | Variable D is a temporal transformer encoder, **not** a reproduction of SLSTT |
| 3.9.7 | Label smoothing of 0.05 has no source in the review corpus | An out-of-corpus choice, held constant and unmeasured |

*Table 3.21 — Every declared departure from the reviewed literature, with its consequence.*

Four of these — the strain formula, the magnification ordering, the absence of temporal mixing, and the input resolution — were identified by the review itself rather than being known design decisions, and are recorded in Chapter 6 as proposed further work; none invalidates the ablation, since every configuration shares the same choices. What they constrain is the *wording* of the conclusions: Chapter 5 reports what a spatial stem at 224 × 224 contributes, and what temporal self-attention contributes in this pipeline, not verdicts on 3D convolution or on SLSTT.

---

### 3.10.6 What this chapter contributes to the thesis

Three things carry forward.

First, the **methodology of Chapter 4 is largely determined by this review, not chosen freely**: the corpus dictates the class grouping, representation, fold structure and metric, and each component's parameters come from published sweeps (§3.1.8, §3.2–§3.9).

Second, the **interpretation of Chapter 5's central result is prepared here, not improvised there**: flow and strain already perform the spatio-temporal feature extraction (§3.3.7, §3.4.7), so a negative contribution from the spatial-only, oversized-resolution backbone is expected, not anomalous (§3.6.6).

Third, the **limits of what can be claimed are established before any number is reported**: a small, single-seed, demographically narrow corpus, measured against richer published regimes, with the ten divergences of Table 3.21 standing throughout — restated in Chapter 6 as consequences of this chapter, not concessions made after the fact.

---

---

## References

*Every source listed is a paper held in the project's `docs/` corpus. Works cited only **inside** those papers — Ekman and Friesen (1969) on nonverbal leakage; the Active Shape Model and Local Weighted Mean registration methods; the original Eulerian magnification formulations of Wu et al. (amplitude) and Wadhwa et al. (phase); the Farnebäck polynomial-expansion and TV-L1 flow estimators; the Temporal Interpolation Model of Pfister et al.; Newton interpolation; the focal loss of Lin et al.; the competing attention modules SE, CBAM, ECA, GC and SRM; the general-purpose architectures GoogLeNet, VGG16, ResNet-101 and DenseNet-169; the CASME (2013), Polikovsky (2009) and Dual-Inception (2019) papers; and the CIFAR, ImageNet-21k and JFT-300M datasets — are attributed in the text to the reviewed source that reports them, and are deliberately not listed here. Label smoothing (§3.9.7) has no source in the corpus and is declared as an out-of-corpus choice at the point of use.*

---

Bai, M., Goecke, R., & Herath, D. (2021). Micro-expression recognition based on video motion magnification and pre-trained neural network. *IEEE International Conference on Image Processing (ICIP 2021)*, 549–553.

Ben, X., Ren, Y., Zhang, J., Wang, S.-J., Kpalma, K., Meng, W., & Liu, Y.-J. (2021). Video-based facial micro-expression analysis: A survey of datasets, features and algorithms. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. https://doi.org/10.1109/TPAMI.2021.3067464

Davison, A. K., Lansley, C., Costen, N., Tan, K., & Yap, M. H. (2018). SAMM: A spontaneous micro-facial movement dataset. *IEEE Transactions on Affective Computing*, 9(1), 116–129.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16 × 16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.

Li, X., Pfister, T., Huang, X., Zhao, G., & Pietikäinen, M. (2013). A spontaneous micro-expression database: Inducement, collection and baseline. *10th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2013)*.

Li, X., Hong, X., Moilanen, A., Huang, X., Pfister, T., Zhao, G., & Pietikäinen, M. (2018). Towards reading hidden emotions: A comparative study of spontaneous micro-expression spotting and recognition methods. *IEEE Transactions on Affective Computing*, 9(4), 563–577.

Li, Y., Huang, X., & Zhao, G. (2018). Can micro-expression be recognized based on single apex frame? *IEEE International Conference on Image Processing (ICIP 2018)*.

Li, Y., Huang, X., & Zhao, G. (2021). Joint local and global information learning with single apex frame detection for micro-expression recognition. *IEEE Transactions on Image Processing*, 30, 249–263.

Liong, S.-T., Gan, Y. S., Yau, W.-C., Huang, Y.-C., & Tan, L. K. (2019a). OFF-ApexNet on micro-expression recognition system. *Signal Processing: Image Communication*, 74, 129–139.

Liong, S.-T., See, J., Wong, K., & Phan, R. C.-W. (2018). Less is more: Micro-expression recognition from video using apex frame. *Signal Processing: Image Communication*, 62, 82–92.

Liong, S.-T., Gan, Y. S., See, J., Khor, H.-Q., & Huang, Y.-C. (2019b). Shallow triple stream three-dimensional CNN (STSTNet) for micro-expression recognition. *14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019)*.

Liong, S.-T., Phan, R. C.-W., See, J., Oh, Y.-H., & Wong, K. (2014a). Optical strain based recognition of subtle emotions. *International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS)*.

Liong, S.-T., See, J., Phan, R. C.-W., Le Ngo, A. C., Oh, Y.-H., & Wong, K. (2014b). Subtle expression recognition using optical strain weighted features. *Asian Conference on Computer Vision (ACCV) Workshops*.

Liong, S.-T., See, J., Phan, R. C.-W., Oh, Y.-H., Le Ngo, A. C., Wong, K., & Tan, S.-W. (2016). Spontaneous subtle expression detection and recognition based on facial strain. *Signal Processing: Image Communication*, 47, 170–182.

Liu, Y., Du, H., Zheng, L., & Gedeon, T. (2019). A neural micro-expression recognizer. *14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019)*.

Lu, Z., Luo, Z., Zheng, H., Chen, J., & Li, W. (2015). A Delaunay-based temporal coding model for micro-expression recognition. In C. V. Jawahar & S. Shan (Eds.), *ACCV 2014 Workshops, Part II*, LNCS 9009 (pp. 698–711). Springer.

Qu, F., Wang, S.-J., Yan, W.-J., & Fu, X. (2016). CAS(ME)²: A database of spontaneous macro-expressions and micro-expressions. In M. Kurosu (Ed.), *Human-Computer Interaction, HCI 2016, Part III*, LNCS 9733 (pp. 48–59). Springer.

See, J., Yap, M. H., Li, J., Hong, X., & Wang, S.-J. (2019). MEGC 2019 — The second facial micro-expressions grand challenge. *14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019)*.

Shreve, M., Godavarthy, S., Goldgof, D., & Sarkar, S. (2011). Macro- and micro-expression spotting in long videos using spatio-temporal strain. *IEEE International Conference on Automatic Face and Gesture Recognition (FG 2011)*, 51–56.

Xia, Z., Hong, X., Gao, X., Feng, X., & Zhao, G. (2020a). Spatiotemporal recurrent convolutional networks for recognizing spontaneous micro-expressions. *IEEE Transactions on Multimedia*, 22(3), 626–640.

Xia, Z., Peng, W., Khor, H.-Q., Feng, X., & Zhao, G. (2020b). Revealing the invisible with model and data shrinking for composite-database micro-expression recognition. *IEEE Transactions on Image Processing*, 29, 8590–8605.

Xu, F., Zhang, J., & Wang, J. Z. (2017). Microexpression identification and categorization using a facial dynamics map. *IEEE Transactions on Affective Computing*, 8(2), 254–267.

Yan, W.-J., Li, X., Wang, S.-J., Zhao, G., Liu, Y.-J., Chen, Y.-H., & Fu, X. (2014). CASME II: An improved spontaneous micro-expression database and the baseline evaluation. *PLoS ONE*, 9(1), e86041.

Yang, L., Zhang, R.-Y., Li, L., & Xie, X. (2021). SimAM: A simple, parameter-free attention module for convolutional neural networks. *38th International Conference on Machine Learning (ICML)*.

Zhang, L., Hong, X., Arandjelović, O., & Zhao, G. (2022). Short and long range relation based spatio-temporal transformer for micro-expression recognition. *IEEE Transactions on Affective Computing*, 13(4), 1973–1985.

Zhao, G., & Pietikäinen, M. (2007). Dynamic texture recognition using local binary patterns with an application to facial expressions. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 29(6), 915–928.

Zhao, S., Tao, H., Zhang, Y., Xu, T., Zhang, K., Hao, Z., & Chen, E. (2021). A two-stage 3D CNN based learning method for spontaneous micro-expression recognition. *Neurocomputing*, 448, 276–289.
