# Handover — full chapter-by-chapter revision (Chapters 1–6) complete

**Written 24 September 2026.** Supersedes the previous `HANDOVER_CURRENT.md`
(23 September; still in git history) and every older `HANDOVER_*` file. Those
older files are now wrong about section numbers, figure/table inventories,
the bibliography policy and several removed sections — do not follow them.

Read §1–§3 before touching anything. §4 is the author's style rules — they are
non-negotiable and apply to every edit. §9 lists what is still open.

---

## 1. What this project is

A master's thesis (LaTeX, compiled in **Overleaf**) on **micro-expression
recognition (MER)**. The study is a **factorial ablation** of a standard MER
pipeline on **CASME II**: four components switched on/off independently —

| Group | Component | Code flag |
|---|---|---|
| Motion modelling | Eulerian video magnification (EVM) | `use_evm` |
| Motion modelling | Convolutional spatial stem (3 streams, spatial-only kernels) | `use_cnn` |
| Motion modelling | SimAM parameter-free attention (only with the stem) | `use_simam` |
| Temporal modelling | Temporal transformer encoder | `use_transformer` |

16 combinations, 4 invalid (SimAM without CNN) → **12 configurations**, each
trained from scratch, **25-fold leave-one-subject-out**, **156 clips**
(Negative 99 / Positive 32 / Surprise 25), single seed 42. Primary metric
**pooled macro F1** (= MEGC 2019 UF1).

**Key results (Chapter 5 — never change these numbers):**

| | |
|---|---|
| Best config | `config_2` (transformer only) **0.7122** pooled macro F1, acc 0.7436 |
| All four components | `config_8` **0.6659**, acc 0.7500 |
| Transformer mean effect | **+0.2173** (6/6 pairs positive, +0.1578 … +0.2786) |
| EVM mean effect | **+0.0152** (4/6; −0.0541 … +0.0795) |
| SimAM mean effect | **+0.0034** (3/4; −0.0288 … +0.0341) |
| CNN stem mean effect | **−0.0310** (2/4; −0.1292 … +0.0094) |
| All-Negative reference | acc 0.6346, macro F1 0.2588 |

The task description (`contents/task_description.tex`, from `docs/topic.pdf`)
frames the thesis: *"how can modeling of temporal phases and motion cues
improve recognition accuracy for subtle and short-lived expressions?"* —
Chapter 1 §1.2 and the Chapter 6 opening are built on its third paragraph.

---

## 2. The manuscript — active files

This is the whole thesis. Nothing else is included.

| # | Chapter | File | Words | Figs | Tables | Numbered eqs | Citations |
|---|---|---|--:|--:|--:|--:|--:|
| 1 | Introduction | `contents/introduction.tex` | 1,617 | 0 | 0 | 0 | 3 |
| 2 | Background | `contents/chapter2_shortened.tex` | 7,407 | 2 | 3 | 22 | 118 |
| 3 | Literature Review | `contents/chapter3_cut.tex` | 6,940 | 3 | 4 | 0 | 164 |
| 4 | Methodology | `contents/chapter4_shortened.tex` | 5,105 | 5 | 6 | 2 | 57 |
| 5 | Results | `contents/chapter5_shortened.tex` | 4,831 | 9 | 9 | 0 | 23 |
| 6 | Conclusion | `contents/chapter6.tex` | 1,707 | 0 | 0 | 0 | 19 |

Plus: `bib/thesis.bib` (**68 entries, all cited**), `contents/list_of_figures.tex`,
`contents/list_of_tables.tex` (front matter; their inventories are *comments*
only and are partly stale — the lists build themselves from short captions),
`contents/task_description.tex`.

**Superseded — never include** (they share labels → 200+ duplicate-label
errors): `chapter2.tex`, `chapter3.tex`, `chapter3_shortened.tex`,
`chapter4.tex`, `chapter5.tex`, all `*_short_half*`/`*_part*` files,
`appendix_formulas.tex`, `formula_guide_*.tex`, `chapter2_pipeline_overview.tex`.

**`contents/formula_guide_complete.tex` is out of date** (equations were
renumbered/added). The author explicitly does NOT want a separate formula
file — formulas are explained in place (§4). Leave it out of the build.

### 2.1 Current structure

**Ch 1** — (hook paragraphs) · 1.1 Research Motivation · 1.2 Research Aim and
Objectives (aim + central RQ from task description, 3 sub-RQs, 4 objectives) ·
1.3 Research Contributions (3 items) · 1.4 Structure of the Thesis.

**Ch 2** — 2.1 Micro-Expressions (Fig 2.1 CASME II frame sequence) ·
**2.2 Micro-Expression Datasets** (moved here from old Ch 3 §3.1: spontaneous
vs posed, widely used datasets incl. CASME/SMIC/SAMM/CAS(ME)²/MMEW/CAS(ME)³/DFME
+ table, CASME II, label grouping, original baseline, MEGC 2019 protocol with
fold-composition Fig 2.2, dataset limitations) · 2.3 Data Acquisition and
Preprocessing · 2.4 Eulerian Video Magnification · 2.5 Motion Representation:
Optical Flow and Optical Strain · 2.6 Neural Network Fundamentals · 2.7 Network
Building Blocks · 2.8 Training with Limited and Imbalanced Data · 2.9
Evaluation Methodology · 2.10 Summary. **22 numbered equations (2.1–2.22), each
followed by a "where …" symbol definition.**

**Ch 3** — intro (+Fig 3.1 MER pipeline) · 3.1 EVM · 3.2 Dense Optical Flow
(+Fig 3.2 flow/strain) · 3.3 Optical Strain · 3.4 Temporal Normalisation ·
3.5 Shallow CNNs (+Fig 3.3 TSNN) · 3.6 SimAM · 3.7 Transformers · 3.8 Class
Imbalance · 3.9 Synthesis and Research Gap (four limitations, research gap,
how the thesis addresses it, **12-row divergence ledger**, `\clearpage` before
Summary so the ledger prints first) · 3.10 Summary. All "Implications for
this Research" subsections were **removed**.

**Ch 4** — 4.1 Research Approach and Experimental Design (Factorial Ablation
Design / Ablated Components / Design Limitations) · 4.2 Dataset and
Preparation (Fig 4.1 class distribution, preprocessing-settings table) · 4.3
Ablation Configurations (Fig 4.2 design grid, 12-config table, matched pairs) ·
4.4 Model Architecture (**4.4.1 Pipeline Overview = the TikZ flowchart + data
shape table**, conv-stem layer table, SimAM, transformer with numbered
positional-encoding equation, head/replacements, Fig 4.3 parameter counts) ·
4.5 Training Procedure (settings table, loss/class weighting incl. the
class-collapse history, Fig 4.4 LR schedule, reproducibility) · 4.6 Evaluation
Protocol (numbered fold-ceiling equation) · **4.7 Research Gaps and
Limitations** (Motion Modelling: EVM / optical flow / optical strain / conv
stem / SimAM; Temporal Modelling: temporal normalisation / transformer;
Training and Evaluation) · 4.8 Summary.

**Ch 5** — 5.1 Overview (ranking table + Fig 5.1, best configuration,
majority-class reference + Fig 5.2, transformer split) · 5.2 Temporal
Transformer · 5.3 EVM · 5.4 SimAM · 5.5 Convolutional Stem · 5.6 Per-Class
Performance (5.6.2 "Recognition of All Three Classes") · 5.7 Comparison with
Published Results (MEGC table, Fig 5.8) · **5.8 Discussion** (component
effects table + Fig 5.9, **confusion-matrix interpretation** with Fig 5.6 and
new **Fig 5.10 prediction distribution**, why each component behaved as it
did, limitations) · 5.9 **Conclusion** (the author renamed "Summary" →
"Conclusion" themselves; keep it).

**Ch 6** — opening built on the task-description aim/RQ · 6.1 Answers to the
Research Questions · 6.2 Contributions (3, academic headings) · 6.3 Limitations
· 6.4 Future Work · 6.5 Final Remarks. **No cross-references to other
sections** (author's request).

---

## 3. The data flow (the pipeline the thesis describes)

**Stage 1 — preprocessing (once per clip):** CASME II original 640×480 frames
(unregistered, uncropped) → select 255 micro-expression clips → drop `Others`
(99) → **156 clips, 25 subjects** (subject 18 only had Others) → greyscale,
resize 224×224 → uniformly sample **33 frames** onset→offset → optional EVM
(α=10, 5–25 Hz at nominal 200 fps, 4-level Laplacian pyramid, FFT mask) →
Farnebäck flow on 32 adjacent pairs → strain magnitude (3-term, shear once) →
stack [u, v, ε] → min–max [0,1] per channel → tensor **(3, 32, 224, 224)**.

**Stage 2 — training/eval (per configuration × fold):** standardise per
channel, random h-flip (negate u) → **stem** (3 unshared streams, Conv3d
(1,3,3) ×2, BN, ReLU, Dropout3d 0.3, MaxPool (1,2,2) → 96×32×112×112) or
**replacement** (avg-pool to 4×4 per frame → Linear 48→96) → optional **SimAM**
(per stream, after pooling; weights ∈ [0.62, 1)) → spatial global average →
32×96 sequence → **transformer** (sinusoidal PE, 4 pre-norm layers, 8 heads,
FF 256, GELU, dropout 0.1, final LayerNorm, mean over time) or **temporal
mean** → head LayerNorm → Dropout 0.3 → Linear 96→3. Focal loss γ=2 + label
smoothing 0.05, **no class weights** (class-balanced sampler active), AdamW
1e-4/1e-4, 5-epoch linear warmup + cosine to 1e-7, 50 epochs, batch 8, clip
1.0, mixed precision, best epoch chosen on the held-out subject (no inner
split — a declared limitation). Confusion matrices summed over 25 folds →
pooled per-class F1 → pooled macro F1.

Parameter counts: stem 14,544 · transformer 348,736 · replacement 4,704 ·
head 483 → totals 5,187 / 15,027 / 353,923 / 363,763.

**Raw results** live in `Ablation_Study/results/config_*/final_results.json`
(`metrics.confusion_matrix`, `per_class_f1`, …). Pooled macro F1 = mean of
`per_class_f1`; the stored `macro_f1` key is the rejected mean-of-folds value.
Model code: `Ablation_Study/models.py` (`AblationMERModel`).

---

## 4. The author's style rules — apply to every edit

1. **Super academic, short, clear, to the point.** No colloquial phrasing, no
   jargon for its own sake. Section/subsection titles must be plain academic
   titles in title case (no "What the corpus supplies…", no "abandon").
2. **Never write "corpus"/"corpora"** in printed text → "dataset" (label names
   such as `sec:corpus-supplies` are invisible and may stay).
3. **Em dashes (`---`) to a minimum; report every one used.** Currently there
   are **none** in the printed text of Chapters 1–6 (only in comments).
4. **No "see Chapter X / Section Y does Z" pointer sentences.** Chapter
   introductions list what the chapter contains; chapter conclusions summarise
   the chapter and preview the next chapter. Ch 6 has no cross-references.
5. **Every displayed formula** is a numbered `equation` with `\label{eq:…}`,
   followed by a "where …" sentence defining every symbol. No separate formula
   file.
6. **No mention of code files, scripts, class names or paths** in the text.
7. **No computational cost anywhere** (no VRAM, GPU-hours, training time,
   "expensive/cheap", memory). Removed thesis-wide on 23–24 September.
8. **References:** cite wherever a claim comes from the literature, but **use
   only entries already in `bib/thesis.bib`** unless the author explicitly asks
   for new ones; verify every claim against the source (see §6).
9. **Sentence-initial references:** the template prints `\autoref` as
   lowercase ("section 2.1"). At a sentence start write
   `Section~\ref{…}` / `Chapter~\ref{…}` / `Figure~\ref{…}` / `Table~\ref{…}`.
   `python tools/check_sentence_autoref.py` finds them (`fix` argument fixes).
10. **Standing rule:** no factual change without checking the source; direct
    quotations are verbatim — never reword one.

---

## 5. What changed in this revision (23–24 September), by chapter

**Ch 1:** "these movements" clarified; Research Aim and Objectives section
built on the task description (aim, central RQ, 3 sub-RQs, 4 objectives);
Research Contributions section added (3 items — the "excluded evaluations"
contribution was later removed); roadmap pointers removed; em dashes removed.

**Ch 2:** intro rewritten as a roadmap; section titles made academic; 22
numbered equations with symbol definitions; Fig 2.1 (CASME II sequence, CC BY)
added; §2.2 Micro-Expression Datasets moved in from Ch 3 and extended with
other datasets; §2.8 (training) de-duplicated against Ch 4; old 2.8.5
evaluation-limitations removed; pipeline overview moved to Ch 4; effective
frame-rate paragraph added to the input-tensor subsection; Summary with Ch 3
preview; computational-cost asides removed.

**Ch 3:** rewritten concisely; dataset section moved to Ch 2; all
"Implications for this Research" subsections removed (unique facts relocated
to Ch 2 §2.5.4 and Ch 4 class-weighting rule); ledger band rows merged → 12
rows; citations extended from 113 to 164 using existing keys; 3 CC-BY figures
from Xie et al. 2022 added; `\clearpage` so the ledger precedes the Summary;
Summary with Ch 4 preview.

**Ch 4:** restructured 46 → 25 subsections (merged subsections keep all old
labels stacked); VRAM, file names, "What is stored" (4.6.6), "Earlier
evaluations" (4.6.7) and the Computational Cost section removed; pipeline
flowchart placed in 4.4.1; tables replace code blocks; 4 charts added;
Research Gaps and Limitations section (motion vs temporal modelling) added;
short Summary.

**Ch 5:** academic titles; Discussion section with confusion-matrix
interpretation and mechanistic explanations (all hedged as plausible, not
tested); Fig 5.10 added; cost subsections, cost table and Fig 5.5 removed;
references audited.

**Ch 6:** fully rewritten (see §2.1); "excluded evaluations" contribution
removed; 19 verified citations added.

**Cross-chapter consequences of removals:** the "excluded evaluations"
contribution and the data-routing-defect story no longer appear anywhere
(Ch 1, 3, 4, 6). Label aliases kept so nothing breaks:
`sec:what-is-stored` → Design Limitations; `sec:simam-cost`, `sec:cnn-cost` →
the SimAM/stem result subsections; `sec:corpus` etc. now live in Ch 2.
Removed labels: `sec:excluded-evaluations`, `tab:five-evaluations`,
`sec:evm-implications`, all `*-implications` except
`sec:transformer-implications`, `sec:computational-environment` family,
`fig:cost-vs-performance`, `tab:compute-per-config`.

---

## 6. Bibliography — 68 entries, fully audited

- Original 30 (papers in `docs/`, text cache in `tools/txt/`) + 35
  foundational sources added 23 Sept + 3 dataset papers (CASME 2013,
  CAS(ME)³, DFME). All 68 cited.
- **Audit (24 Sept):** every entry re-verified against Crossref / Semantic
  Scholar / DBLP / Open Library (title, first author, year, volume, pages);
  24 missing DOIs added. Every citation in Ch 1–5 checked against the paper's
  text (cached sources) or the paper's core contribution (foundational ones);
  7 over-stated wordings corrected. Ch 6 citations repeat verified claims.
- **Do not run `tools/check_references.py`** — it sends the author's email in
  the HTTP User-Agent. Use **`tools/verify_bib.py`** (no personal data).
- `tools/check_quotes.py`: 27/28 verified; the 28th ("might not be enough to
  reveal the ME motion progress", liX2018) is a known false negative (page
  break in the PDF), verified by hand.

---

## 7. Figures

All images in `report_figures_thesis/` (bare filenames; `\graphicspath` must
include it). Licensing noted in captions.

| Fig | File | Source |
|---|---|---|
| 2.1 | `fig2_1_casme2_sequence.png` | Yan et al. 2014 Fig 2, **CC BY** |
| 2.2 | `fig3_2_fold_composition.png` | own (older script) |
| 3.1 | `fig3_1_mer_pipeline.png` | Xie et al. 2022 Fig 2, **CC BY 4.0** |
| 3.2 | `fig3_2_flow_strain.png` | Xie et al. 2022 Fig 5 cropped (panels orig. Liong et al. 2018), **CC BY 4.0** |
| 3.3 | `fig3_3_tsnn_streams.png` | Xie et al. 2022 Fig 6, **CC BY 4.0** |
| 4.1–4.4 | `fig4_1_…` … `fig4_4_…` | `tools/thesis_ch4_figures.py` |
| 4.x | pipeline flowchart | TikZ inside Ch 4 (needs `tikz` + libraries `arrows.meta, positioning, fit, backgrounds`) |
| 5.1–5.4, 5.6–5.9 | `fig5_*.png` | `tools/thesis_fig_regen.py` / older scripts |
| 5.10 | `fig5_10_prediction_distribution.png` | `tools/thesis_ch4_figures.py` (`fig5_10()`) |

`fig5_5_cost_vs_performance.png` and `fig4_5_*` are **no longer used**.
Only three `docs/` PDFs are openly licensed (Xie 2022, Yan 2014, Adegun 2020);
do not reproduce figures from the IEEE/Elsevier papers.

---

## 8. Tools (run from the repo root)

| Script | Purpose |
|---|---|
| `tools/check_manuscript.py` | labels, refs, cites, floats, images, duplicated prose. Benign noise: "hard-coded section numbers" flags decimals; column-spec warnings |
| `tools/check_quotes.py` | every quotation vs source PDF text (27/28 expected) |
| `tools/verify_bib.py` | re-verify all bib entries online (no personal data) |
| `tools/verify_cite_numbers.py <chapter.tex>` | every number in a cited sentence vs the cited source. Expected flags: thesis's own numbers (156, 224), derived differences (6.48, 10.12, 0.057), SimAM 0.62, LaTeX column widths |
| `tools/cite_sentences.py <files…>` | print every citing sentence (for claim review) |
| `tools/list_chapter_cites.py <chapter.tex>` | keys used in a chapter + bib existence |
| `tools/claim_check.py <checks.txt>` | regex evidence search in `tools/txt/` (lines: `tag | key | pattern`) |
| `tools/check_sentence_autoref.py [fix]` | sentence-initial `\autoref` (lowercase in this template) |
| `tools/thesis_ch4_figures.py` | regenerates Figs 4.1–4.4, 5.10 |
| `tools/extract_source_pdfs.py` | rebuilds `tools/txt/` if missing |

Current state: 260 labels, 0 duplicates, 0 unresolved, 0 invalid keys, 0
orphan entries, 0 sentence-initial autorefs, no cross-chapter near-duplicates.

**Editing trap:** Bash heredocs in this environment mangle backslashes
(`\cite`, `\alpha`, `\n`). Write Python edit scripts with the Write tool using
raw strings (`r"…"`), then run them.

---

## 8a. Final audit (24 September, later session)

References re-verified (`verify_bib.py`: 62 verified, 6 metadata-source quirks
all confirmed correct), quotes 27/28 (known false negative), cited numbers OK,
Ch 5 Discussion confusion-matrix figures re-derived from the raw results (all
correct). Plagiarism: 6/7-word n-gram scan of printed text against the 30
cached sources found only stock phrases and names; two cited paraphrases
reworded further. Edits: Ch 1 literary hook made academic, Ch 1/§1.1
duplicate removed, Ch 2 description in §1.4 corrected, robustness aim added to
§1.2 (Ch 6 answers it); code names removed (`config_N` → C*n*, NumPy/OpenCV/
Conv3d/BatchNorm3d/Dropout3d, "code path", "run history"); Ch 3 no longer
claims the class-collapse history as a contribution, Grad-CAM++/disentanglement
paragraph removed, two overclaims softened; duplicated passages trimmed (EVM
pass-band ×3, double-correction rule, encoder specs in Ch 3, fold-composition
and Subject 18 in Ch 2, repeated seed/per-clip caveats in Ch 5, Ch 6 Final
Remarks); Ch 6 contributions reordered to match Ch 1. Table columns removed:
Ch 4 twelve-config "Description" (duplicated the checkmarks), Ch 5 ranking
"Correct" (= accuracy × 156), Ch 5 component-effects "Pairs" (contained in
"Positive pairs").

## 9. Still open

0. **Figure text:** Figs 5.1, 5.6, 5.9 carry informal in-image labels
   ("SLSTT", "3D-CNN", "green band = Transformer ON", "Green = the component
   helped, red = it hurt", "C8 — proposed unified"). Regenerate with neutral
   labels if wanted (`tools/thesis_fig_regen.py`, `tools/loso_report_figures.py`).

1. **Never compiled in this checkout.** Upload every changed file to Overleaf
   and compile; check float placement (ledger before Ch 3 Summary; wide
   tables), and measure the page count (goal ~100 pages).
2. **`main.tex` (Overleaf only):** bibliography must be `bib/thesis` (not
   `bib/example`); remove `\nocite{*}` if present (all 68 are cited anyway);
   `\graphicspath{{figures/}{report_figures_thesis/}}`; packages `amsmath`,
   `amssymb` (`\checkmark`), `graphicx`, `tikz` + the four libraries; decide
   equation numbering style ((2.1) per chapter is the default; continuous
   numbering needs `\counterwithout{equation}{chapter}`).
3. **Overleaf upload list** (everything changed since 23 Sept): all six chapter
   files, `bib/thesis.bib`, `contents/list_of_figures.tex`,
   `contents/list_of_tables.tex`, and images `fig2_1_casme2_sequence.png`,
   `fig3_1_mer_pipeline.png`, `fig3_2_flow_strain.png`,
   `fig3_3_tsnn_streams.png`, `fig4_1…fig4_4`, `fig5_10_prediction_distribution.png`.
4. **Inventory comments** in `list_of_figures.tex` / `list_of_tables.tex` are
   partly stale (Ch 4/5 changed after the last update) — cosmetic only.
5. **Figure text:** `fig5_6_confusion_matrices.png` titles contain em dashes
   ("C1 — pure baseline"); regenerate if the author wants none anywhere.
6. **Possible final pass:** abstract / front matter (not reviewed in this
   checkout), and a whole-thesis read for consistency of tone after compile.
