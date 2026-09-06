# Chapter 2 — Background

Draft chapter for the MSc thesis *Micro-Expression Recognition on CASME-II*.

## Files

| File | Section | Words |
|---|---|:--:|
| `00_Chapter2_Complete.md` | **assembled chapter — read or submit this** | ~7,500 |
| `01_2.1_The_Phenomenon.md` | 2.1 The phenomenon | ~880 |
| `02_2.2_From_Video_To_Motion.md` | 2.2 Optical flow and optical strain | ~1,070 |
| `03_2.3_Motion_Magnification.md` | 2.3 Motion magnification | ~1,010 |
| `04_2.4_Network_Building_Blocks.md` | 2.4 Network building blocks | ~1,290 |
| `05_2.5_Learning_Under_Scarcity_And_Skew.md` | 2.5 Training machinery | ~1,255 |
| `06_2.6_Evaluating_On_A_Small_Corpus.md` | 2.6 Evaluation apparatus | ~1,300 |
| `07_References.md` | the chapter's single reference list (11 entries) | — |

Section files carry no reference lists of their own; the rebuild script appends `07_References.md` to the assembled chapter.

## Editing

```bash
./rebuild_complete.sh
```

## Conventions

- **Background gives mechanism; Chapter 3 gives literature, evidence and argument.** Where both need a fact, Background owns it and Chapter 3 cross-references. Chapter 3 was edited during this work to defer to §2.1 (the definition of the phenomenon and the half-second bound), §2.2.2 (the optical flow constraint equation), §2.2.3 (the strain tensor), §2.4.1 (the spatial-only kernel), §2.5.2 (the focal-loss mechanism) and §2.6.3 (pooled versus per-fold estimators).
- **Only what the project uses.** Quantisation, pruning and distillation are absent from the codebase and are excluded, with the absence stated in §2.5.
- **No invented citations.** Standard ML background with no source in `docs/` is stated uncited.

## How each section was produced

Every section passed a four-stage pipeline before being accepted:

1. **Scope check** — an agent confirmed, against the code and data, that each proposed topic has a real footprint in the project, and reported the exact implementation facts.
2. **Writing** — an agent wrote from those verified facts, under an explicit no-duplication constraint against Chapter 3.
3. **Fact and reference audit** — an agent verified every claim against the code and every citation against the PDF, checking for ghost references, wrong author order, and orphaned citations.
4. **Duplication audit** — an agent compared the section against the assembled Chapter 3, in both directions.

### What the checks caught

- **§2.1** — the pipeline never reads the apex frame, and Action Units are never read by the model; both are now stated explicitly rather than implied. Two quotations had drifted from their sources and were restored. Four duplication overlaps with §3.1 were removed.
- **§2.2** — a parameter name was wrong (`normalize`, not `normalize_inputs`) and one reference lacked its year suffix. Normalisation is **two-stage** — min–max at extraction and z-score at load — which an earlier reading had missed.
- **§2.3** — clean. Confirmed the temporal filter is an ideal FFT mask rather than the Butterworth filter of the cited method, that all four pyramid bands share one α, and that there is no chromatic or spatial-frequency attenuation.
- **§2.4** — clean. Confirmed there are no author-written residual connections, and that `Ablation_Study/models.py` rather than the `Stage2_Architecture/` prototype is the file the experiments used.
- **§2.5** — clean. Confirmed the most consequential fact in the section: because the balanced sampler is active, **the focal loss's class-weighting term is inactive in the default runs**, so it operates through difficulty-focusing and label smoothing alone.
- **§2.6** — clean. Confirmed that the pooled macro F1 is never computed by the training code, that UAR is never computed at all, and that **there is no inner validation split** — the held-out fold serves both as the checkpoint-selection signal and as the final scored set, which is optimistically biased and is flagged as a limitation.

## Outstanding

- `docs/` holds two near-identical PDFs of the Xia et al. STRCN paper (one hyphenated filename, one not) — worth de-duplicating.
- Bibliographic details for references whose `docs/` copies are preprints or author manuscripts remain unconfirmed; the list is in `../Thesis_Chapter3_LiteratureReview/Tech-wise/README.md`.
