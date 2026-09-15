# Shortened Chapter 3: parts 9 and 10 — the final batch

Sections **3.9 (training under severe class imbalance)** and **3.10 (synthesis and research gap)**. With these, all ten parts of the shortened Chapter 3 exist. Parts 1–8 were read for continuity; one defect in part 10's source material was traced back to part 2 and is recorded below.

## Inclusion

```tex
\input{contents/chapter3_short_parts9_10}
```

Insert after section 3.8; these close the chapter. Do not include the original sections 3.9–3.10 alongside them — the labels are shared. Compile from the thesis root.

## Length

Prose words on the parts 3–4 convention (comments, floats, equations and LaTeX command names excluded):

- Part 9: 1,666 → 1,124 words, 32.5 % shorter.
- Part 10: 1,294 → 1,159 words, 10.4 % shorter.

Part 10 is the least-compressed section in the chapter, deliberately. It is the keystone: it carries the consolidated research gap, the four recurring limitations that the conclusion cites four separate times, and the twelve-row divergence ledger that four chapters depend on. Its function *is* to restate, and an examiner reads it to get the whole chapter's argument in one place. Cutting it to parity would have gutted it.

### Chapter-wide total, all ten parts

| Section | Before | After | Reduction |
|---|--:|--:|--:|
| 3.1 + 3.2 | 6,858 | 4,952 | 27.8 % |
| 3.3 | 2,154 | 992 | 53.9 % |
| 3.4 | 2,334 | 1,012 | 56.6 % |
| 3.5 | 2,159 | 1,417 | 34.4 % |
| 3.6 | 1,857 | 1,094 | 41.1 % |
| 3.7 | 1,863 | 799 | 57.1 % |
| 3.8 | 1,671 | 793 | 52.5 % |
| 3.9 | 1,666 | 1,124 | 32.5 % |
| 3.10 | 1,294 | 1,159 | 10.4 % |
| **Chapter 3** | **21,856** | **13,350** | **38.9 %** |

**Sections 3.1 and 3.2 are the weakest at 27.8 %**, and are now the largest remaining opportunity in the chapter: they were the first batch, written before the counting convention and the cut method had settled. Bringing them to the ~55 % that sections 3.3, 3.4, 3.7 and 3.8 reached would remove roughly a further 1,900 words and take the chapter past 48 %. That is the single highest-value piece of remaining work.

## A structural defect found in part 10, originating in part 2

The divergence ledger cites its sources by **subsection number**, not by label. Row 3 read **3.2.7** for the 5–25 Hz temporal band. In the shortened chapter that divergence lives in **3.2.6**, because part 2 merged the old "Why magnification may behave differently in front of a motion representation" subsection into "Implications for this research", shifting every later subsection of section 3.2 up by one. The row now reads 3.2.6.

A repeatable checker is included at `<scratchpad>/ledgercheck.py`. It rebuilds the subsection map from the assembled shortened parts and verifies (a) every ledger row number, and (b) every hard-coded `3.x.y` reference in Chapters 2, 4, 5, 6 and the introduction. **All twelve rows and all ten external references now resolve correctly** — 3.1.2, 3.1.6, 3.1.8, 3.3.1, 3.3.3, 3.4.3, 3.5.7, 3.9.5, 3.9.7 and 3.10.3. Run it again after any further renumbering of sections 3.1–3.2.

## Tables, figures, acronyms

- `tab:imbalance-decisions` (five rows) and `tab:divergence-ledger` (**twelve rows**) retain their labels and short captions.
- Neither section contained figures; none was removed or invented.
- `chapter3_short_parts910_acronyms.tex` adds only Grad-CAM++ and XBM; SLSTT was already declared in the parts 7–8 list.
- All eight subsection slots in 3.9 and all six in 3.10 are preserved in order. `chapter2.tex` cites **Section 3.9.5** (metric-level correction) and **Section 3.9.7** (implications) by number, so those positions are fixed.

## Source audit and corrections

Two report-only subagents audited sources and compression; the controller verified each finding against the source before applying it. All six keys in section 3.9 resolve to a bib entry and a real PDF: `zhaoS2021`, `xia2020a`, `liong2016`, `dosovitskiy2021`, `yan2014`, `see2019`. Section 3.10 carries no citations. **All 27 bibliography entries remain cited somewhere in the shortened chapter — no orphans.**

Corrections applied:

- **A wrong locator.** Section 3.9 cited Xia et al.'s ablation as "§V-F-1" twice. In the PDF actually supplied in `docs/` — which is the arXiv preprint, not the IEEE TMM version the bibliography describes — that subsection is **IV-F-1**; the paper has only five top-level sections and V is the conclusion. Since the published numbering may differ again, the text now cites the subsection **by its title** rather than by number, which is robust to either version.
- **An overstated effect size.** Section 3.9 described Xia et al.'s balanced-loss effects as "differences of under a point", repeated in the implications subsection as "the largest effect they report is under a point". Their Table IV, image-verified, gives twelve balanced-loss cells ranging from +0.1 to **+2.1** points, with one cell at −1.0; four of the twelve exceed a point. The largest effect in the table overall is 24.8 points, for removing *augmentation*, not the balanced loss. The text now says the balanced loss helps "only 'slightly' — at most about two points across their twelve cells, and in one cell costing accuracy", which is the authors' own word plus the verified range.
- **The 224 × 224 reframing from parts 5–6 has reached the ledger**, as that batch's handover required. The row now reads "against Xia et al.'s guidance of ≤ 100 × 100 — though Bai et al. and Li, Huang and Zhao both use 224 × 224", instead of implying a departure from the whole corpus.

**Four absence claims in section 3.9 were re-tested and all four stand**, each with a named disproving search across all 38 cached paper texts: that no reviewed paper reports the two corrections failing outright; that label smoothing is named in the corpus (only by Dosovitskiy et al., not a micro-expression paper) but never examined on this task; that γ is nowhere swept on micro-expression data; and that no keyed paper uses class-proportional resampling. The last also confirms that the weighted sampler is this thesis's own choice rather than an inherited one.

The two-failure narrative in 3.9.7 is sourced to this project's run history rather than to the literature and so could not be checked here. The run-time guard that resolves it **was** verified in code, as were γ = 2.0, label smoothing = 0.05, the optional α vector being switched off at run time, and the verbatim log string. Whoever audits Chapters 4 and 5 should confirm the two collapses and the "no configuration abandons any class" claim against the actual logs.

## Verification

Passed, across all ten parts assembled: no citation key lost from any section; all labels preserved and unique; every `\autoref` in the chapter resolves; all 27 bibliography keys still used; twelve ledger rows present and all resolving; every hard-coded cross-chapter subsection number resolving; no chapter or numbered-section signposts in parts 9–10; balanced braces and environments in every part; no control characters anywhere; short captions on both tables.

No master `.tex`, class file or TeX engine exists in this checkout, so compilation, float placement, list generation and pagination remain to be verified in the real project. Rebuild the bibliography and compile to convergence there.

## What remains

1. **Re-cut sections 3.1 and 3.2** to the standard the later batches reached — the largest single win available, roughly 1,900 words.
2. Re-run `ledgercheck.py` afterwards, since renumbering 3.1–3.2 would move ledger rows and Chapter 2's numbered references.
3. Chapter 6 still uses some older cost descriptions that parts 7–8 superseded, and still describes the 224 × 224 divergence in the pre-correction framing. Neither was silently rewritten.

## Two further corrections in part 10, found after the first draft

Both were surfaced by the source audit and independently re-verified by the controller against the PDFs before being applied.

**The chapter's central gap claim was overstated, and contradicted section 3.6 of the same chapter.** Subsection 3.10.2(a) read "No architectural component is ever isolated in a matched pair" and "Every paper proposes a complete system and evaluates it against other complete systems". That is false. Xia et al. compare a basic recurrent convolutional network against three variants — adding a wide expansion, a shortcut connection and an attention unit — and state that "all parameters ... are set consistent in all models" and that "each parameter will be changed individually in every experiment while others would use the [same basic setup]" of a 60 x 60 flow-map, 5 x 5 average pooling and 16 feature maps. That is a matched-pair ablation of learned modules. Section 3.6.3 of this chapter already calls the same experiment the corpus's most useful, so the old wording contradicted the chapter as well as the source.

The claim is now the one that survives every search: components *are* isolated, but always one at a time, from inside a single proposed design; no reviewed study varies several jointly so that their interactions can be read. The bold gap statement in 3.10.3 and the framing in 3.10.4 were adjusted to match.

**This defect propagates outside Chapter 3 and was not silently repaired.** `chapter4.tex:29` still says "no reviewed study isolates a *learned* component --- a convolutional backbone, an attention module, a temporal encoder --- against an otherwise identical pipeline", which Xia et al.'s attention-unit variant contradicts directly. `chapter6.tex` still says "The reviewed corpus evaluates components only inside complete systems". Chapter 6's wording is hedged with "seldom" and is closer to defensible; Chapter 4's is not. **Both need the same correction, and both are outside this batch.**

**A representation claim in 3.10.1 was wrong.** It credited the strain field with "an independently supported interior optimum". Section 3.4.5 reports strain only on/off, with a weak effect, under leave-one-*video*-out; and 3.10.2(d) of the same section names the three parameters that do have interior optima, strain not among them. The sentence now states what the review supports: strain adds a cue, weakly and under a protocol that is not subject-disjoint, while the magnification factor and the interpolated sequence length are the two parameters with swept interior optima.

## One question the audit could not settle, left for you

Ledger row 2 asserts that the pipeline reads "the original recorded frames, unregistered and uncropped", with "no landmarking or alignment of any kind". The audit could not confirm this from the code and flagged it as the one row it could not verify.

The controller's own check supports the row: `Processed_Data/master_thesis_labels.csv` gives frame directories under **`CASME2-RAW`**, not under `Cropped`, and `chapter2.tex` states the same position explicitly. Against that, `Stage1_DataPipeline/config.py` comments describe its images mode as reading "CASME-II Cropped" folders, and `frames_root` points at `Raw_Videos_Magnified/CASME2`. **The label table and Chapter 2 agree, so the row is very probably right, but the config comments should be reconciled with it.** Nothing was changed on this point.

## Four more ledger corrections, from the compression audit

The second audit checked every ledger row against the *shortened* section it summarises, rather than the original. Four rows had drifted:

- **Row 5 (3.4.7)** asserted "Strain channel is noisier than in the reviewed work". The shortened 3.4.6 only hedges: "spatial differentiation *can* amplify flow-estimation noise". The row now matches that hedge.
- **Row 10 (3.7.7)** named the pooling axes as "(D, H, W), not the paper's (H, W)". Part 7's own prose does not use those labels, and the axis names risk asserting more than either section says. The row now states the distinction part 7 actually draws: whole-clip rather than per-frame statistics.
- **Row 12 (3.9.7)** said label smoothing "has no source in the review corpus". `HANDOVER_CH3_SHORTENING.md` §4 records that exact form as one of the nine false gap claims already repaired once — Dosovitskiy et al. *are* in the corpus and do name it. The row now carries the precise version 3.9.6 uses.
- **A cross-reference in 3.10.2(a) was wrong.** It cited `sec:cnn-shrink` for Xia et al.'s three-module comparison; that material is in **3.6.4** (`sec:cnn-limited-data`), while `sec:cnn-shrink` holds the depth-against-resolution sweep. Both are now cited, each for what it actually carries.

The audit also challenged the 3.10.2(a) rewrite itself as "a factual change dressed as a correction". It was adjudicated against the source before being applied and again afterwards: `revealing-invisible-shrinking` states that the basic model and the three module variants "are abbreviated as RCN, RCN-W, RCN-S, and RCN-A", that "all parameters ... are set consistent in all models", and that "each parameter will be changed individually in every experiment while others would use the" basic setup of a 60 x 60 flow-map, 5 x 5 average pooling and 16 feature maps. Two independent agents and the controller reached the same reading, one of them from a rendered page image. The correction stands.

**Section 3.6.5 was harmonised to match.** It previously read "No study isolates the backbone in a matched pair", which the corrected 3.10.2(a) contradicts. It now reads "No study isolates the backbone itself", acknowledges the matched comparisons the corpus does contain, and makes the surviving point: those comparisons all *add* to a backbone that stays in place, and none removes the learned spatial stem entirely.
