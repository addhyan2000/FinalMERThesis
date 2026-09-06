# Handover — MER Thesis (read this first)

**Purpose.** This file lets a fresh Claude Code session pick up where the previous one stopped, without re-deriving anything. Read it end to end before touching any file.

**Project.** MSc thesis: micro-expression recognition on CASME-II. A four-component ablation (EVM · SimAM · 3D-CNN · SLSTT Transformer) evaluated under complete 25-fold leave-one-subject-out.

---

## 1. READ THIS BEFORE QUOTING ANY NUMBER

`Ablation_Study/results/config_*/final_results.json` contains **two different macro-F1 quantities, and they can differ in sign.**

| Quantity | Where | Status |
|---|---|---|
| **Pooled macro F1** = mean of the `per_class_f1` array | must be computed; the code never stores it | ✅ **the thesis's primary metric** — identical in construction to MEGC's UF1 |
| Mean-of-folds macro F1 = the `macro_f1` key | stored directly | ❌ **rejected** — structurally capped at 0.6267 by fold composition |
| Pooled accuracy = the `micro_f1` key | stored | ✅ quote this as "accuracy" (micro-F1 ≡ accuracy here) |
| Mean-of-folds accuracy = the `accuracy` key | stored | ⚠️ inflates config_8 by 6.3 points |

A verification agent once reported the SimAM effect as an *error* because it used the `macro_f1` key. On pooled macro F1 SimAM is **+0.003**; on the rejected column it is **−0.004**. Always use pooled.

```python
# correct way to read a result
import json
o = json.load(open('Ablation_Study/results/config_8_.../final_results.json'))
pooled_macro_f1 = sum(o['per_class_f1']) / len(o['per_class_f1'])
```

## 2. Verified headline numbers (recomputed on pooled macro F1)

| Component | Mean effect | Pairs | Consistency |
|---|:--:|:--:|---|
| SLSTT Transformer | **+0.217** | 6 | positive in 6/6, min +0.158 |
| EVM | +0.015 | 6 | 4/6 positive, −0.054 to +0.080 |
| SimAM | +0.003 | 4 | 3/4 positive, −0.029 to +0.034 |
| 3D-CNN | **−0.031** | 4 | 2/4 positive, min −0.129 |

Matched pairs — Transformer: 1→2, 4→12, 3→9, 13→7, 5→6, 16→8 · EVM: 1→4, 3→13, 9→7, 6→8, 5→16, 2→12 · SimAM: 3→5, 9→6, 7→8, 13→16 · 3D-CNN: 1→3, 2→9, 4→13, 12→7.

Other verified figures: config_8 pooled accuracy 0.7500, pooled macro F1 0.6659, 117/156 correct · config_2 0.7436 / 0.7122, 116/156 · always-Negative floor 0.6346 / 0.2588 · transformer-on range 0.583–0.712, off 0.419–0.448, gap 0.135 · 8 of 12 configs contain the CNN and consume 48.9 of 50.6 GPU-hours (97 %) · backbone 14,544 params (15,027 with head) · transformer 348,736 params.

**Corpus:** 255 rows in `Processed_Data/master_thesis_labels.csv` → 156 after dropping "others". Negative 99 / Positive 32 / Surprise 25. 25 usable subjects (subject 18's three clips are all "others"). Subject 17 = 33 clips. 10 folds single-class, 8 two-class, 7 three-class. Clip lengths 31–126 frames at 200 fps.

## 3. What exists

| Path | State |
|---|---|
| `Thesis_Chapter2_Background/` | ✅ **complete** — 6 sections + consolidated refs, all verified |
| `Thesis_Chapter3_LiteratureReview/Tech-wise/` | ✅ **complete** — 10 sections + consolidated refs, all verified |
| `MER_Thesis_Final_Presentation.pptx` | ✅ 13 slides, speaker notes embedded |
| `SPEAKER_NOTES.md` | ✅ per-slide script, timing plan, Q&A prep |
| `THESIS_PRESENTATION*.md` | 3 variants of the deck source |
| `tools/build_presentation_pptx.js` | regenerates the .pptx — `node tools/build_presentation_pptx.js` |
| Chapters 1, 4, 5, 6 | ❌ **not started** |

Each chapter folder has a `README.md` (conventions, verification log, outstanding items) and a `rebuild_complete.sh` that reassembles the complete file after editing any section.

## 4. Conventions that must be preserved

- **Chapter 2 gives mechanism; Chapter 3 gives literature, evidence and argument.** A fact is established once and cross-referenced thereafter. Chapter 3 already defers to §2.1, §2.2.2, §2.2.3, §2.4.1, §2.5.2 and §2.6.3.
- **Cite only PDFs in `docs/`.** Works cited only *inside* those papers (Ekman, FACS, Wu et al. EVM, Wadhwa, Farnebäck, TV-L1, Pfister TIM, Newton interpolation, Lin et al. focal loss, SE/CBAM/ECA/GC/SRM, Vaswani, GoogLeNet/VGG16/ResNet/DenseNet) are attributed in text to the reviewed paper that reports them and get **no reference entry**. Standard ML background with no `docs/` source is stated **uncited** — never invent a citation.
- **One reference list per chapter.** Section files carry none.
- **Only write about what the project implements.** Grad-CAM++, adversarial identity disentanglement, quantisation, pruning and distillation are absent from the codebase and are excluded, with the exclusion stated.
- British spelling. Each Chapter 3 section closes with a gap statement.

## 5. The workflow that produced these chapters

Per section, four agent stages — repeat this for Chapters 4–6:

1. **Scope check** — verify against code and data that each proposed topic has a real footprint; report exact implementation facts.
2. **Write** — from the verified facts only, under an explicit no-duplication constraint naming what the other chapter owns.
3. **Fact + reference audit** — every claim against code, every quotation against the PDF, every citation checked for ghost references and author order.
4. **Duplication audit** — against the other chapter, in both directions.

PDF text extraction (rebuild if the scratchpad is gone):
```bash
python3 -m venv /tmp/lit/venv && /tmp/lit/venv/bin/pip install pypdf
# then a 6-line script using pypdf.PdfReader(path).pages[i].extract_text()
```

## 6. Implementation facts that surprised us — do not re-derive

- The **3D-CNN performs no temporal convolution**: kernel `(1,3,3)`, pooling `(1,2,2)`, T unchanged. It is a learned *spatial* stem. Word conclusions accordingly.
- **EVM is applied after temporal subsampling**, with the nominal 200 fps passed to the magnifier, so the realised pass-band is clip-dependent and below the intended 5–25 Hz.
- The temporal step is **uniform index sampling of existing frames**, not manifold TIM. No frames are synthesised. L = 33 → T = 32.
- **Focal loss's α term is inactive by default** — `use_loss_weights = use_class_weights and not use_balanced_sampler`, and the sampler is on. It operates through difficulty-focusing plus label smoothing only.
- **No inner validation split** — the held-out fold is used both for checkpoint selection and as the final scored set. Optimistically biased; flagged in §2.6.5.
- Strain magnitude is **three-term** (shear once), matching STSTNet's own printed equation but not the four-term Frobenius form of Liong et al. (2014a).
- Normalisation is **two-stage**: min–max at extraction, z-score at load. Both active.
- Augmentation flips horizontally **and negates the u channel**.
- The input is **224×224**, against literature guidance of ≤100×100 — possibly a resolution result as much as an architecture result.
- Stale docstrings: `run_ablation_experiments.py` and `ablation_config.py` describe an "8-cell" matrix; the code builds 12.

## 7. Known errors already corrected — do not reintroduce

- The winner across five protocols is **four distinct configurations**, not five (config_16 wins or ties three times).
- **Four** of the eleven other configurations score below the config_4 baseline, not five. Prose in `ALL_RESULTS_LOSO.md` still says five and contradicts its own rank column.
- `config_4` (EVM only) is **the baseline**; `config_1` is the EVM-off control, not a second baseline.
- N = 156 here vs MEGC's 145 — an **eleven**-clip difference, of which nine are sadness + fear; the residual two are unexplained in MEGC's own count.
- MEGC's CASME II subset figures reach UF1 0.83–0.90, but those models train on the 442-sample composite. Single-corpus comparison is not like-for-like.

## 8. Outstanding

- Bibliographic details for references whose `docs/` copies are arXiv preprints or author manuscripts (STSTNet, STRCN, Xia 2020b, SLSTT, OFF-ApexNet, Bi-WOOF, Li 2018, Li/Huang/Zhao 2018 & 2021, Liong 2014a/b, Shreve 2011, SMIC, Xu 2017).
- Xu et al.: PDF says 2016, cited volume 8(2) implies 2017. Chapter uses 2017.
- The Delaunay paper's real pagination is 698–711; the filename says 703–716.
- `docs/` holds two copies of the Xia et al. STRCN paper.
- Suggested next experiments, in order: save per-clip predictions (enables McNemar), rerun the CNN arm at 28×28 or 56×56, magnify before subsampling, multi-seed runs.

## 9. Next task

**Chapter 4 (Methodology)** is the natural next piece — §3.10.6 argues it is largely determined by the review already, and the parameters are all in `ablation_config.py` and this file. Chapter 5 can draw on `ALL_RESULTS_LOSO.md` and `LOSO_Validation_Report.md`, but re-verify their prose against the JSON: those documents contain at least the "five of eleven" error noted above.
