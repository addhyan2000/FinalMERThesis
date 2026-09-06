# Chapter 3 — Literature Review

Draft chapter for the MSc thesis *Micro-Expression Recognition on CASME-II*.

## Files

| File | Section | Ablation variable |
|---|---|:--:|
| `00_Chapter3_Complete.md` | **assembled chapter — read or submit this** | — |
| `01_3.1_CASME-II_Corpus.md` | 3.1 The evaluation corpus | — |
| `02_3.2_Eulerian_Video_Magnification.md` | 3.2 Motion magnification | A |
| `03_3.3_Optical_Flow.md` | 3.3 Motion representation | input |
| `04_3.4_Optical_Strain.md` | 3.4 Deformation representation | input |
| `05_3.5_Temporal_Normalisation.md` | 3.5 Sequence-length normalisation | input |
| `06_3.6_3D-CNN_Backbone.md` | 3.6 Learned spatial backbone | C |
| `07_3.7_SimAM_Attention.md` | 3.7 Parameter-free attention | B |
| `08_3.8_Transformer.md` | 3.8 Temporal encoder | D |
| `09_3.9_Class_Imbalance.md` | 3.9 Training under class imbalance | constant |
| `10_3.10_Synthesis_and_Gap.md` | 3.10 Synthesis and research gap | — |
| `11_References.md` | **the chapter's single reference list** (27 entries) | — |

The section files carry no reference lists of their own — every citation resolves against `11_References.md`, which the rebuild script appends to the assembled chapter.

## Editing

Edit the numbered section files, then regenerate the assembled chapter:

```bash
./rebuild_complete.sh
```

The script strips each section's per-file heading and concatenates from `## 3.` onwards, so section files stay individually readable.

## Conventions used throughout

- **Written as one continuous chapter, not ten standalone sections.** A fact is established once, in the section that owns it, and cross-referenced thereafter. The canonical homes are: the corpus and its statistics §3.1.3 and §3.1.7; the always-Negative floor and the LOSO protocol §3.1.4; the metric definition (pooled macro F1 ≡ MEGC's UF1) §3.1.6; flow as an analytical feature extractor §3.3.7; strain as a hand-specified differential operator §3.4.7; the 3D-CNN's −0.031 and its 97 % compute share §3.6; the transformer's +0.217 §3.8; the imbalance treatment §3.9.
- **One reference list.** `11_References.md` holds all 27 sources. Works cited only *inside* those papers are attributed in the text to the reviewed source that reports them and are listed in that file's preamble rather than as entries.
- **Structure.** Each section runs: what the technique addresses → how it works → the published evidence → its limitations → implications for this thesis → a closing gap statement. §3.10 consolidates the ten gap statements.
- **Declared divergences.** Wherever the implemented system departs from the literature it draws on, the section says so. All ten are consolidated in Table 3.21 of §3.10.
- **Excluded topics.** Grad-CAM++ auditing and adversarial identity disentanglement are in the project's reading list but not in the implemented system, so they are not reviewed (§3.9 scope note).

## Verification status

**All ten sections verified — 6 September 2026**, then edited for cross-section repetition. The chapter was cut from ~33,000 to ~29,700 words by removing restatement only: ten per-section reference lists were merged into one (some works had been listed four times), ten copies of the sources policy and ten verification stamps were replaced by single statements in the chapter preamble, and facts established in §3.1 are now cross-referenced rather than re-derived. No citation, quotation, table or verified figure was removed from its canonical home.

**Audit outcome by section.** Each was audited independently against (a) the source PDFs in `../docs/`, (b) the project's code, and (c) `Ablation_Study/results/config_*/final_results.json`. Every section file carries a verification stamp in its footer.

| Section | Outcome |
|---|---|
| 3.1 CASME-II corpus | 4 corrections applied |
| 3.2 EVM | clean |
| 3.3 Optical flow | 2 corrections applied |
| 3.4 Optical strain | 2 corrections applied |
| 3.5 Temporal normalisation | 1 addition |
| 3.6 3D-CNN backbone | 1 correction applied |
| 3.7 SimAM | clean |
| 3.8 Transformer | clean |
| 3.9 Class imbalance | clean |
| 3.10 Synthesis | 1 correction applied |

Corrections made during this pass:

- **§3.1** — CASME II's facial area is more than *triple* CASME's, not double (that ratio is CASME II vs SMIC); the MEGC Negative-class footnote is on Table III, not Table II; the difference between this thesis's N = 156 and MEGC's N = 145 is **eleven** clips, not nine (nine are sadness + fear; the residual two are a discrepancy in the challenge's own count and are now flagged as such); Quang et al. marked as out-of-corpus in Table 3.6.
- **§3.3** — Xu et al. cited as 2017 consistently (Table 3.10 said 2016); added the caveat that three clips shorter than the sampling window yield duplicate-frame pairs and hence zero flow.
- **§3.4** — STSTNet's own printed strain magnitude is a **three-term** form matching this project's implementation, so the three-term choice follows its most direct precedent rather than being an unexamined divergence; OSF's temporal pooling is **mean** in Liong et al. (2016), not sum as in the 2014 papers.
- **§3.5** — noted that Ben et al.'s 30–60 frame optimum was measured almost entirely while *upsampling*, whereas this thesis is almost entirely *downsampling*.
- **§3.6** — scope note completed with ResNet-101 and DenseNet-169.
- **§3.10** — the five evaluation protocols produced **four** distinct winning configurations, not five (config_16 wins or ties in three of the five runs). The same overclaim was corrected in `../THESIS_PRESENTATION*.md`, `../SPEAKER_NOTES.md`, `../tools/build_presentation_pptx.js`, and the rebuilt `../MER_Thesis_Final_Presentation.pptx`.

One reported error was **rejected on inspection**: an auditor reported Qu et al. (2016), CAS(ME)², as absent from `docs/`. The PDF is present — its filename (`CASME2_ADatabaseofSpontaneous...`) resembles CASME II. No change made.

## Outstanding — needs a manual check before submission

- Publication venues and page numbers for references whose copies in `docs/` are arXiv preprints or author manuscripts, and therefore cannot be confirmed from those copies: STSTNet (arXiv:1902.03634), STRCN (arXiv:1901.04656), Xia et al. 2020b (arXiv:2006.09674), OFF-ApexNet, Bi-WOOF/"Less is more", Li et al. (2018), Li/Huang/Zhao (2018, 2021), Liong et al. (2014a ISPACS, 2014b ACCV Workshops), Shreve et al. (2011), Li et al. (2013 SMIC), Xu et al. (2017).
- **Xu et al.** — the PDF masthead reads 2016 but the cited volume, IEEE TAC 8(2), corresponds to 2017. The chapter uses 2017 throughout for internal consistency; confirm against the IEEE record.
- **A Delaunay-Based Temporal Coding Model** — the PDF's own front matter gives pages 698–711; the filename's "703-716" is wrong. The chapter cites 698–711.
- **The residual two-clip gap** between this thesis's disgust + repression count (90) and MEGC's reported CASME II Negative count (88), flagged in §3.1.3.
