# HANDOVER: MER master thesis, defence preparation

Written 14 October 2026 for a new agent continuing this work in another account.
Read this whole file before doing anything. It is self-contained: it records the
project, the current state of every file, the rules the author has set, and every
task given so far.

---

## 0. TL;DR

- This is the repository of a **master's thesis on micro-expression recognition (MER)**,
  written in LaTeX and compiled on **Overleaf**. The thesis was **submitted on 2 October 2026**.
  The manuscript is **final**: do not edit the chapter files unless the author explicitly asks
  (e.g. corrections after examination).
- The current focus is the **defence**. The project root now contains only the defence files:
  - `MER_Thesis_Defence.pptx`: 40-slide deck (24 main + 16 backup), speaker notes on every slide.
  - `Presentation_Speech_Script.docx`: read-aloud script for all 40 slides, about 20 to 24 minutes.
  - `Defence_Questions_and_Answers.docx`: 404 examiner questions with model answers, no repeats.
  - `HANDOVER.md` (this file) and `.gitignore`.
- Everything else that used to be loose in the root, and every `.md` file in the repo, was moved into
  `legacy presentations/`. The author wants the root kept clean.
- **Every fact in the thesis and the defence material must come from the thesis** (or, for paper
  questions, from the cited papers). Never invent numbers, claims or references.
- Working tree is clean; branch `full-loso-17July`; latest commit `0a1923c moved more files`.
  Commit or push only when the author asks.

---

## 1. The author and how they work

- Master's student; thesis on micro-expression recognition. Name not recorded here; the title slide
  of the deck still has `[Your name] | Supervisor: [name] | [University, date]` placeholders
  for them to fill.
- Writes requests briefly, often with typos; infer intent, act, then report clearly.
- Wants things **academic, concise, verified**. Gets annoyed by repetition, colloquial phrasing and
  unverified claims. Asked for plagiarism checks and reference verification.
- Prefers that you **just do the task** and report what changed (with counts and file names),
  flagging judgement calls and anything you could not verify.
- For defence material asked for answers "short and simple using /caveman skill". No caveman skill is
  installed; the style was applied by hand (terse fragments, no filler, facts exact). Used in Parts 2
  and 3 of the Q&A document.
- Asked for a per-slide reading script that fits **20 to 30 minutes**, followed by questions.
- Wants only the defence files in the project root; everything else archived in `legacy presentations/`
  (except the `docs/` folder, which stays where it is).

---

## 2. The study (what the thesis is about)

**Title:** "Improving Micro-Expression Recognition through Temporal and Motion Modeling".
**Subtitle used on slides:** "A factorial ablation of a standard recognition pipeline on CASME II".

**Task description** (from `docs/topic.pdf`, copied verbatim into `contents/task_description.tex`):
investigate whether improved temporal modelling and motion representations can enhance MER.
Central research question: *how can modelling of temporal phases and motion cues improve recognition
accuracy for subtle and short-lived expressions?* Expected outcome mentions improving sensitivity to
fine-grained dynamics while maintaining robustness to noise (robustness was **not** measured; the
thesis says so).

**Design.** A factorial ablation of four components of a standard MER pipeline:

| Group | Component | When disabled |
|---|---|---|
| Motion modelling | Eulerian video magnification (EVM), alpha = 10, 5–25 Hz, 4-level Laplacian pyramid, hard Fourier mask | no magnification |
| Motion modelling | Convolutional spatial stem: 3 unshared streams (u, v, strain), Conv3d kernels (1,3,3) 16 then 32 filters, BN, ReLU, Dropout3d 0.3, MaxPool (1,2,2) | per-frame 4x4 average pooling + Linear 48 to 96 (4,704 params) |
| Motion modelling | SimAM parameter-free attention (only with the stem), lambda = 1e-4, statistics over T x H x W, weight in about [0.62, 1) | unmodified stream outputs |
| Temporal modelling | Temporal transformer: sinusoidal PE, 4 pre-norm layers, 8 heads, d = 96, FF 256, GELU, dropout 0.1, final LayerNorm, mean over time | plain mean over time (no params, no order) |

16 combinations, 4 invalid (SimAM without stem) gives **12 configurations**, named **C1 to C16**
(C10, C11, C14, C15 do not exist). Never use the old code names `config_N_...` in text.

| Config | EVM | SimAM | CNN | TF |
|---|---|---|---|---|
| C1 | - | - | - | - |
| C2 | - | - | - | yes |
| C3 | - | - | yes | - |
| C4 (baseline) | yes | - | - | - |
| C5 | - | yes | yes | - |
| C6 | - | yes | yes | yes |
| C7 | yes | - | yes | yes |
| C8 (all four) | yes | yes | yes | yes |
| C9 | - | - | yes | yes |
| C12 | yes | - | - | yes |
| C13 | yes | - | yes | - |
| C16 | yes | yes | yes | - |

**Matched pairs** (differ in exactly one component): transformer 6, EVM 6, SimAM 4, CNN stem 4 = 20.
Effect of a component = mean difference over its pairs.

**Data.** CASME II, original 640x480 frames, 200 fps, no alignment/cropping. Released coding 255 clips,
26 subjects; exclude "Others" (99 clips; also removes subject 18) gives **156 clips, 25 subjects**.
Classes by valence: **Negative 99** (disgust 63, repression 27, sadness 7, fear 2), **Positive 32**
(happiness), **Surprise 25**. MEGC 2019 uses 145 CASME II clips (Negative = disgust + repression = 88);
the thesis has 11 more (9 sadness/fear, 2 unexplained).

**Stage 1 (preprocessing, once per clip):** greyscale, resize 224x224, uniformly sample **33 frames**
onset to offset (effective rate 6600/N fps: 100 fps at median N = 66, about 52 fps for the longest clip;
81/156 clips downsampled 2x or more; 3 clips have 31 frames so indices repeat), optional EVM (applied
**after** sampling, so the real pass-band is clip-dependent, about 2.5–12.5 Hz at median length),
Farnebäck flow on 32 pairs (pyr scale 0.5, 3 levels, window 15, 3 iterations, poly 5, sigma 1.2),
strain magnitude sqrt(exx^2 + eyy^2 + exy^2) (shear counted once; Liong et al. count it twice),
stack [u, v, strain], min-max per channel per clip. Tensor **3 x 32 x 224 x 224**.

**Stage 2 (training/eval per configuration x fold):** per-sample standardisation, random h-flip
(negate u), stem/SimAM/pool/transformer/head (LayerNorm, Dropout 0.3, Linear 96 to 3).
Focal loss gamma = 2 + label smoothing 0.05, **no class weights**, class-balanced sampler (weights 1/n_c,
with replacement), AdamW lr 1e-4 wd 1e-4, warmup 5 epochs from 1e-5 then cosine to 1e-7, 50 epochs,
batch 8, grad clip 1.0, mixed precision, seed 42 reset per configuration and fold, Kaiming/Xavier init.
**Best epoch selected on the held-out subject (no inner split): declared limitation.**
Per-clip predictions were **not** stored (no McNemar test possible).

**Evaluation:** 25-fold leave-one-subject-out. Confusion matrices summed over folds; **primary metric
pooled macro F1** (= MEGC 2019 UF1). Mean-of-folds macro F1 is capped at 0.627 because 10 folds have one
class, 8 two, 7 three. Pooled accuracy reported alongside.

**Parameters:** stem 14,544; transformer 348,736; replacement projection 4,704; head 483. Totals:
5,187 (neither), 15,027 (stem only), 353,923 (transformer only), 363,763 (both). Transformer about 96 %.

### Results (never change these numbers)

| Rank | Config | Macro F1 | Accuracy | F1 Neg | F1 Pos | F1 Sur |
|---|---|---|---|---|---|---|
| 1 | C2 | 0.7122 | 0.7436 | 0.8068 | 0.6506 | 0.6792 |
| 2 | C8 | 0.6659 | 0.7500 | 0.8458 | 0.5556 | 0.5965 |
| 3 | C7 | 0.6625 | 0.7051 | 0.7845 | 0.5479 | 0.6552 |
| 4 | C12 | 0.6581 | 0.6795 | 0.7485 | 0.5591 | 0.6667 |
| 5 | C6 | 0.6171 | 0.7308 | 0.8325 | 0.4286 | 0.5902 |
| 6 | C9 | 0.5830 | 0.7308 | 0.8491 | 0.6000 | 0.3000 |
| 7 | C13 | 0.4480 | 0.4359 | 0.4127 | 0.5278 | 0.4035 |
| 8 | C4 | 0.4386 | 0.4808 | 0.5641 | 0.5055 | 0.2462 |
| 9 | C1 | 0.4337 | 0.4615 | 0.5600 | 0.3714 | 0.3696 |
| 10 | C5 | 0.4302 | 0.4231 | 0.3840 | 0.5227 | 0.3838 |
| 11 | C3 | 0.4252 | 0.4167 | 0.3607 | 0.5227 | 0.3922 |
| 12 | C16 | 0.4192 | 0.4038 | 0.3500 | 0.5143 | 0.3934 |

- Mean effects: **transformer +0.2173** (6/6 positive, +0.1578 to +0.2786); **EVM +0.0152** (4/6,
  -0.0541 to +0.0795); **SimAM +0.0034** (3/4, -0.0288 to +0.0341); **CNN stem -0.0310** (2/4,
  -0.1292 to +0.0094).
- Pair values: TF 1→2 +0.2786, 16→8 +0.2467, 4→12 +0.2195, 13→7 +0.2146, 5→6 +0.1869, 3→9 +0.1578.
  EVM 9→7 +0.0795, 6→8 +0.0488, 3→13 +0.0228, 1→4 +0.0049, 5→16 -0.0109, 2→12 -0.0541.
  SimAM 9→6 +0.0341, 3→5 +0.0050, 7→8 +0.0034, 13→16 -0.0288. CNN 4→13 +0.0094, 12→7 +0.0044,
  1→3 -0.0085, 2→9 -0.1292.
- With transformer 0.5830–0.7122, without 0.4192–0.4480, gap **0.1350**.
- Best C2 beats all-four C8 by **0.0463**; C8 has more accuracy (117 vs 116 clips correct).
- All-Negative reference: accuracy 0.6346, macro F1 0.2588.
- Confusion matrices (rows true Neg/Pos/Sur): C1 [[42,22,35],[4,13,15],[5,3,17]];
  C8 [[85,3,11],[13,15,4],[4,4,17]]; C2 [[71,20,8],[3,27,2],[3,4,18]] (recall 0.72/0.84/0.72).
- Predicted counts (Neg, Pos, Sur): C2 77/51/28, C8 102/22/32, C7 82/41/33, C12 72/61/23, C6 110/10/36,
  C9 113/28/15, C13 27/40/89, C4 57/59/40, C1 51/38/67, C5 26/56/74, C3 23/56/77, C16 21/38/97.
  No-transformer configs over-predict minority classes (Surprise 40–97 times; Negative recall 0.21–0.44).
- Published MEGC 2019 CASME II-subset UF1: OFF-ApexNet 0.8764, Zhou et al. 0.8621, STSTNet 0.8382,
  EMR 0.8293, Bi-WOOF 0.7805, Quang et al. 0.7068, LBP-TOP 0.7026 (thesis 0.7122 sits above the last two).
  Not like-for-like: composite training (442 samples, 68 subjects), 145 vs 156 clips, checkpoint bias.
- Raw results: `Ablation_Study/results/config_*/final_results.json` (`metrics.confusion_matrix`,
  `per_class_f1`). Pooled macro F1 = mean of `per_class_f1`; the stored `macro_f1` key is the rejected
  mean-of-folds value.

**Contributions (same order in Ch 1 and Ch 6):** (1) factorial evaluation of a standard pipeline;
(2) measured account: temporal modelling carries almost all of the effect; (3) the full pipeline is not
the best configuration.

**Main limitations:** single seed; epoch chosen on held-out subject; no per-clip predictions; fixed
untuned settings (alpha 10, T 32, lambda 1e-4, gamma 2); 12 declared divergences from literature; one
dataset, one ethnicity, lab recording; transformer effect confounded with added capacity.

---

## 3. Repository layout (current)

```
FinalMERThesis/
  MER_Thesis_Defence.pptx              <- defence deck (final)
  Presentation_Speech_Script.docx      <- read-aloud script
  Defence_Questions_and_Answers.docx   <- question bank
  HANDOVER.md                          <- this file
  .gitignore                           <- leave in place
  contents/            thesis .tex files (active + many superseded; see section 4)
  bib/thesis.bib       68 entries, all cited
  report_figures_thesis/   figure PNGs used by the thesis
  docs/                source PDFs (topic.pdf, papers); author said keep where they are
  tools/               verification and figure scripts; tools/txt/ = text of 30 cached papers
  Ablation_Study/      model code + results/config_*/final_results.json
  Stage1_DataPipeline/, Stage2_Architecture/, Processed_Data/   experiment code/data
  report_figures*/     older figure outputs
  Thesis_Chapter1..6_*/  old Markdown chapter drafts (their .md files were moved to legacy)
  tmp/                 old working files
  legacy presentations/  archive (see below)
```

**`legacy presentations/`** (moved 14 Oct 2026, at the author's request; relative subpaths kept):
- all 100 `.md` files that were in the repo, including **`HANDOVER_CURRENT.md`** (the detailed
  manuscript handover of 24–25 Sep 2026: structure, labels, style rules, tool list), `README.md`,
  old reports, the `THESIS_PRESENTATION*.md` drafts, `SPEAKER_NOTES.md`, chapter Markdown drafts,
  `contents/*README.md`, `tmp/` reports;
- the old deck `MER_Thesis_Final_Presentation.pptx`;
- `MER_Experiment_Report.pdf`, `MER_Thesis_Presentation.pdf`, `CLIENT_FEEDBACK_RESPONSE.txt`,
  `IMPLEMENTATION_GUIDE.txt`, `LOSO_PILOT_GUIDE.txt`;
- `setup and GPU scripts/`: `*.ps1`, `requirements.txt`, `README_GPU.txt`, `gui_settings.json`.
  To re-run experiments, move `requirements.txt` and `gui_settings.json` back to the root
  (`tools/run_gui.py` reads `gui_settings.json` from the root; `setup_gpu.ps1` expects `requirements.txt`
  in the root).

Do not move things back, and do not add files to the root, unless the author asks.

---

## 4. The thesis manuscript

**Active files (the only ones `main.tex` should include):**

| Chapter | File |
|---|---|
| Task description | `contents/task_description.tex` (verbatim copy of `docs/topic.pdf`, own hand-typed [1]–[7] reference list, no `\cite`, no effect on the bib) |
| 1 Introduction | `contents/introduction.tex` |
| 2 Background | `contents/chapter2_shortened.tex` |
| 3 Literature Review | `contents/chapter3_cut.tex` |
| 4 Methodology | `contents/chapter4_shortened.tex` |
| 5 Results | `contents/chapter5_shortened.tex` |
| 6 Conclusion | `contents/chapter6.tex` |
| Declaration (after bibliography) | `contents/declaration.tex` (AI-use declaration, author's wording verbatim; says "essay") |
| Bibliography | `bib/thesis.bib` (68 entries) |

**Superseded, never include** (they share labels, causing hundreds of duplicate-label errors):
`chapter2.tex`, `chapter3.tex`, `chapter3_shortened.tex`, `chapter3_short*.tex`, `chapter4.tex`,
`chapter5.tex`, every `*_short_half*`/`*_part*` file, `appendix_formulas.tex`, `formula_guide_*.tex`,
`chapter2_pipeline_overview.tex`.

**`main.tex` exists only on Overleaf** (not in this repo). Required changes the author was told to make:
`\usepackage{float}` (the Ch 4 pipeline table and flowchart use `[H]`);
`\graphicspath{{figures/}{report_figures_thesis/}}`; include the six active chapter files above;
`\bibliography{bib/thesis}` (not `bib/example`) and remove `\nocite{*}`; `\include{contents/task_description}`
after the title page; `\include{contents/declaration}` after the bibliography. Class is KOMA `scrbook`;
`unsrt` bibliography style (numbers in citation order); TikZ libraries `arrows.meta, positioning, fit, backgrounds`.

**Chapter structure (final):**
- Ch 1: hook; 1.1 Research Motivation; 1.2 Aim and Objectives (central RQ, 3 sub-RQs, 4 objectives,
  robustness aim); 1.3 Contributions (3); 1.4 Structure.
- Ch 2: 2.1 Micro-Expressions (Fig 2.1 CASME II sequence, cited `yan2014`); 2.2 Datasets (table of
  datasets; 2.2.5 "Benchmark Protocols" = original baseline + MEGC 2019 paragraph; 2.2.6 dataset
  limitations); 2.3 Data acquisition/preprocessing; 2.4 EVM; 2.5 Flow and strain; 2.6 NN fundamentals;
  2.7 Building blocks; 2.8 Limited/imbalanced data; 2.9 Evaluation; 2.10 Summary. 22 numbered equations,
  each followed by a "where ..." definition.
- Ch 3: no figures (supervisor request); 3.1–3.8 one section per component; 3.9 Synthesis and Research
  Gap (four limitations a–d, the gap, how addressed, 12-row divergence ledger); 3.10 Summary.
- Ch 4: 4.1 Design; 4.2 Dataset (fold-composition figure moved here from Ch 2, `[tbp]`); 4.3
  Configurations (table C1–C16, matched pairs); 4.4 Architecture (4.4.1 pipeline overview: data-shape
  table then TikZ flowchart, both pinned `[H]` at the end of 4.4.1); 4.5 Training; 4.6 Evaluation; 4.7
  Research Gaps and Limitations; 4.8 Summary.
- Ch 5: 5.1 Overview; 5.2 Transformer; 5.3 EVM; 5.4 SimAM; 5.5 Stem; 5.6 Per-class; 5.7 Published
  results; 5.8 Discussion; 5.9 Conclusion.
- Ch 6: answers to RQs, contributions, limitations, future work, final remarks; no cross-references.

**Figures:** `report_figures_thesis/`. Only Fig 2.1 (Yan et al. 2014, `yan2014`) and the Ch 5
literature-comparison figure (`see2019`) have external sources. Figs 5.1, 5.6 and 5.9 still have informal
in-image labels ("SLSTT", "3D-CNN", "green = helped", "C8 — proposed unified"); generated by
`tools/thesis_fig_regen.py` and `tools/loso_report_figures.py`. Not regenerated (optional, unasked).

---

## 5. Style rules for any thesis text (author's rules)

1. Super academic, short, clear; plain title-case headings; no colloquial phrasing.
2. Never write "corpus"/"corpora" in printed text; use "dataset" (labels like `sec:corpus` may stay).
3. Em dashes to a minimum; report every one used (none in printed text of Ch 1–6).
4. No "see Chapter X / Section Y does Z" pointer sentences; chapter intros list contents, chapter
   conclusions summarise and preview the next chapter; Ch 6 has no cross-references.
5. Every displayed formula: numbered `equation` with `\label{eq:...}`, followed by "where ..." defining
   every symbol. No separate formula file.
6. No code files, scripts, class names or paths in the text (configs are C1–C16).
7. No computational cost anywhere (VRAM, GPU-hours, training time, cheap/expensive).
8. Cite only entries already in `bib/thesis.bib` unless asked; verify every claim against its source;
   direct quotations verbatim.
9. Sentence-initial references: `Section~\ref`, `Chapter~\ref`, `Figure~\ref`, `Table~\ref`
   (the template prints `\autoref` lowercase). `python tools/check_sentence_autoref.py` finds them.
10. **Figure captions (supervisor, 25 Sep):** source only as a citation at the end of the caption, e.g.
    "...right inner brow (AU~4)~\cite{yan2014}." Never "Reproduced from ... under the CC BY licence".
11. No figures in the Literature Review chapter.
12. Chapter `.tex` files use **CRLF** line endings; preserve them when editing.

---

## 6. Defence materials (current, root folder)

### 6.1 `MER_Thesis_Defence.pptx` (40 slides, 16:9, editable)
- Built with pptxgenjs (structured deck: theme, layouts `TITLE_DARK`/`CONTENT`, sections). Charts and
  tables are **native PowerPoint objects** built from thesis numbers (no thesis PNGs except Fig 2.1 on
  slide 2 and the fold-composition figure on slide 28). Fonts Cambria (headings) / Calibri (body).
  Palette: dark teal 0B3C49, teal 0E6B7A, amber E08E2B, slate 7C8D99, brick B4433A, green 3F8F6B.
- Speaker notes on all 40 slides (about 3,250 words for slides 1–24).
- Slides 1–24 (main, 20–25 min): 1 Title; 2 Micro-expressions; 3 Pipelines; 4 Research gap; 5 RQs;
  6 Dataset; 7 Stage 1; 8 Four components; 9 Stage 2 + params; 10 Factorial design; 11 Protocol;
  12 Ranking chart; 13 Best config (stat tiles); 14 Mean effects; 15 All 20 pairs; 16 Per-class F1;
  17 Confusion matrices; 18 Why components behaved so; 19 Comparison with MEGC; 20 Answers to RQs;
  21 Limitations; 22 Future work; 23 Contributions and conclusion; 24 Thank you.
- Slides 25–40 (backup, after "Thank you"): 25 divider; 26 datasets table; 27 CASME II labels + sample
  counts; 28 fold composition + 0.627 ceiling; 29 EVM detail; 30 flow and strain equations; 31 temporal
  sampling; 32 stem layers + SimAM equations; 33 transformer vs SLSTT; 34 training settings + losses;
  35 metrics; 36 published evidence (Li 2018, Liong 2016 tables); 37 full results table; 38 predicted
  class counts chart; 39 twelve divergences; 40 research gap per component.
- Title slide still has placeholders `[Your name] | Supervisor: [name] | [University, date]`.
- The generator scripts lived in a session scratchpad and are gone; **edit the .pptx directly** (e.g.
  unzip/XML, python-pptx, or PowerPoint). If regenerating, the skill's rules apply (pptxgenjs, validate
  with the pptx skill's `validate.py`, render with LibreOffice at
  `C:\Program Files\LibreOffice\program\soffice.exe` and inspect every slide).

### 6.2 `Presentation_Speech_Script.docx`
- Read-aloud text for all 40 slides in deck order; 3,009 words: about 20 min at 150 wpm, 21.5 at 140,
  24 at 125. Each slide heading shows estimated seconds and the running clock.
- Slide 24 ("Thank you") is used as a **transition** into the backup slides; the talk closes on slide 40
  with "Thank you ... happy to take your questions". Suggested shortcut: skip slides 29–36 if short of time.
- Replace `[your name]` on slide 1.

### 6.3 `Defence_Questions_and_Answers.docx` (404 questions, deduplicated 14 Oct)
- Intro + "Numbers to know by heart".
- **Part 1 (236):** per main slide 1–24, about 10 each, easy → medium → hard, full-sentence answers.
- **Part 2 (94):** questions on the cited papers by topic (psychology/FACS, datasets, MEGC/LBP-TOP/TIM,
  magnification, flow/strain, deep MER, transformers/attention, NN foundations, imbalance/optimisation,
  evaluation statistics, surveys) + common topic questions. Terse "caveman-style" answers.
- **Part 3 (74):** backup slides 26–40, results and discussion, how the study could have been better,
  why each design decision was made. Terse answers.
- Rule when adding questions: check they are not already in an earlier part; keep easy → hard order;
  keep numbering sequential (Q1...); answers only from the thesis / cited papers.
- Some answers (e.g. "most surprising result") are personal judgement; the author was told to adapt them.

---

## 7. Full task log (what the author asked, and what was done)

All dates 2026.

**Before 24 Sep (earlier sessions):** chapter-by-chapter revision of Ch 1–6, bibliography expanded to
68 audited entries, figures added, cost material removed, Ch 4 restructured, Ch 5 Discussion, Ch 6
rewritten. Details in `legacy presentations/HANDOVER_CURRENT.md`.

**24 Sep, whole-thesis audit** ("check references, academic language, plagiarism, repetition,
columns of no use, coherent story"):
- `verify_bib.py`: all 68 references verified online; quotes 27/28 verbatim (28th known PDF page-break
  false negative); cited numbers checked; Ch 5 confusion-matrix statements recomputed from raw results.
- Plagiarism: 6/7-word n-gram scan vs 30 cached source texts; only stock phrases; two paraphrases reworded.
- Edits: literary Ch 1 hook made academic; code names removed (`config_N` → C*n*, NumPy/OpenCV/Conv3d etc.);
  overclaims softened; duplicated passages trimmed; Ch 3 no longer claims class-collapse history as a
  contribution; Grad-CAM++ paragraph removed; Ch 6 contribution order matched to Ch 1.
- Table columns removed: Ch 4 "Description", Ch 5 "Correct" and "Pairs".

**24 Sep, task description:** made `task_description.tex` a verbatim copy of `docs/topic.pdf` with its
own reference list, independent of the bib.

**24 Sep, declaration:** created `contents/declaration.tex` (AI-use declaration after the bibliography).

**24 Sep, Ch 2 §2.2.6 MEGC + Fig 2.2:** dissolved the MEGC subsection into "Benchmark Protocols";
moved the fold-composition figure to Ch 4 §4.2.3.

**24 Sep, Fig 4.4 breaking text:** pinned the §4.4.1 table and flowchart with `[H]` at the end of 4.4.1;
author told to add `\usepackage{float}`.

**24 Sep, main.tex review:** listed required changes (section 4 above).

**25 Sep, supervisor feedback:** (3) figure source as end-of-caption citation only, no licence text;
(4) all figures removed from the Literature Review (3 figures + their intro sentences).

**25–26 Sep, reference links:** produced a verified link (DOI or official/arXiv page) for all 68 bib
entries, in bib-file order; all resolved; FACS via APA PsycTests DOI 10.1037/t27734-000.

**2 Oct, thesis submitted.**

**2 Oct, defence deck:** 24-slide editable .pptx with speaker notes (20–25 min, no template), plus a
244-question Q&A .docx (10+ per slide, easy → hard).

**2 Oct, backup slides:** 16 backup slides appended after "Thank you" (thesis content only).

**14 Oct, tidy-up:** all `.md` files and the old deck moved to `legacy presentations/`; final deck to root.

**14 Oct, Q&A Part 2:** +127 questions on referenced papers and common topics (114 on papers), terse
style; paper facts checked against `tools/txt/` with `tools/claim_check.py`.

**14 Oct, Q&A Part 3:** +141 questions on backup slides, results/discussion, improvements, design
decisions.

**14 Oct, speech script:** `Presentation_Speech_Script.docx` for all 40 slides, 20–24 min.

**14 Oct, Q&A deduplication:** 512 → 404 questions; duplicates removed (kept the earlier, fuller
answer), renumbered, intro counts updated.

**14 Oct, root cleanup:** PDFs and .txt notes to `legacy presentations/`; GPU/setup files to
`legacy presentations/setup and GPU scripts/`; root holds only the defence files.

**14 Oct, this handover.**

---

## 8. Tools and environment

- Windows 11; PowerShell 5.1 is the default shell (no `&&`; use `;`), Git Bash also available.
- **Editing trap:** Bash heredocs mangle backslashes (`\cite`, `\alpha`). Write Python edit scripts to a
  file with raw strings, then run them. PowerShell here-strings piped to `python -` work, but avoid
  `/\s+/`-style regex in the same command (a safety filter blocked one such command).
- Preserve CRLF when rewriting `.tex` files (`open(..., newline="\r\n")`).
- Python has `python-docx`, `pymupdf`, `PIL`, `defusedxml`, `lxml`. `python-pptx` is not installed.
  Node 22 is installed; `pptxgenjs` is **not** global (install it in a scratch folder if needed).
  LibreOffice: `C:\Program Files\LibreOffice\program\soffice.exe`; `pdftoppm` absent (use pymupdf).
- Thesis check scripts (run from repo root):

| Script | Purpose |
|---|---|
| `tools/check_manuscript.py` | labels, refs, cites, floats, images, duplicated prose |
| `tools/check_quotes.py` | quotations vs source text (27/28 expected) |
| `tools/verify_bib.py` | re-verify bib entries online (no personal data) |
| `tools/verify_cite_numbers.py <file>` | numbers in cited sentences vs source |
| `tools/cite_sentences.py`, `tools/list_chapter_cites.py` | citation review helpers |
| `tools/claim_check.py <checks.txt>` | regex evidence search in `tools/txt/` (`tag | key | pattern`) |
| `tools/check_sentence_autoref.py [fix]` | sentence-initial `\autoref` |
| `tools/thesis_ch4_figures.py`, `tools/thesis_fig_regen.py`, `tools/loso_report_figures.py` | figures |

- **Never run `tools/check_references.py`**: it sends the author's email address in the HTTP User-Agent.
- Never put the author's email or other personal data in requests, URLs or files.

---

## 9. Open items and known issues

1. Title slide placeholders (deck and speech script slide 1) need the author's name, supervisor,
   university and date.
2. Optional: move the "Thank you" slide (24) to the end of the deck so it follows the backup slides; the
   script currently treats slide 24 as a transition.
3. Optional: regenerate thesis Figs 5.1, 5.6, 5.9 without informal in-image labels (only matters if the
   thesis is revised after examination).
4. The Q&A answers in Parts 2–3 are deliberately terse; Part 1 answers are full sentences.
5. Overleaf `main.tex` changes (section 4) were recommended; whether the author applied them is unknown.
6. The declaration says "essay" rather than "thesis"; this was pointed out and no change was requested.
7. If the examiners request corrections, edit only the active chapter files listed in section 4, follow
   the style rules in section 5, rerun `tools/check_manuscript.py` and `tools/check_quotes.py`, and do not
   change any result number.
