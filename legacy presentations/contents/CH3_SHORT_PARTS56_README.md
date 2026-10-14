# Shortened Chapter 3: parts 5 and 6

These are sections **3.5 (temporal normalisation)** and **3.6 (the shallow 3D-CNN spatial backbone)** of the ten-part Chapter 3 shortening task. They are not replacements for thesis Chapters 5 and 6. The original `chapter3.tex` and the existing parts 1–4 were not edited.

## Use the new text

```tex
\input{contents/chapter3_short_parts5_6}
```

The wrapper includes `chapter3_short_part5.tex` and `chapter3_short_part6.tex`. It belongs after section 3.4 and before section 3.7, inside Chapter 3. It contains no chapter declaration, preamble or counter overrides. Compile from the thesis root. Do not include the original sections 3.5–3.6 alongside these replacements: they use the same labels.

## Tables, figures, acronyms and numbering

- Both existing tables remain, with short captions for the List of Tables and their original labels: `tab:tim-decisions` and `tab:cnn-decisions`.
- The original sections 3.5–3.6 contained no figures. None was removed and none invented. Keep the master document's `\listoftables` and `\listoffigures`; entries generate from captions on compilation.
- `chapter3_short_parts56_acronyms.tex` supplies portable description-list entries for the List of Acronyms. Merge once with the existing entries, removing the six already declared in `chapter3_short_parts34_acronyms.tex` (CNN, LBP-TOP, LOSO, MEGC, SMIC, STSTNet). If the master uses `glossaries` or `acronym`, translate rather than inserting a second list. JCFDA and MMEW were expanded from Ben et al.'s own text, not guessed.
- **All 19 labels survive**, including the subsection positions other chapters address **by number**: `chapter2.tex:139` cites "Section 3.5.7", and four rows of the divergence ledger cite 3.5.7 and 3.6.6. Section 3.5 keeps eight subsections with Implications seventh; section 3.6 keeps seven with Implications sixth. Do not reorder them.
- One heading changed: §3.5.4 "The second systematic comparison" → "The remaining systematic comparisons", because a third comparison was found (see below). §3.5.8 was renamed to "Research gap and scope" to match parts 3/4. Both labels are unchanged.
- Citations use stable BibTeX keys. First-appearance numbering may change after shortening: rebuild the bibliography and compile to convergence, including both lists.

## Reduction and scope

Prose words on the parts-3/4 convention (excluding comments, floats, equations and LaTeX command names):

- Part 5: 2,159 → 1,419 words, a 34.3 % reduction.
- Part 6: 1,857 → 1,094 words, a 41.1 % reduction.
- Together: 4,016 → 2,513 words, approximately 37 % shorter.

**This is below parts 3/4's ~56 %, and deliberately so.** Both audits independently concluded that 56 % parity is arithmetically unreachable here: §3.5.7 and §3.6.6 are floored near 810 words between them by four divergence-ledger rows and six cross-references from Chapters 2, 5 and 6. Roughly 180 words were also *added back* to §3.5 to repair false absence claims (below); accuracy was preferred to the target. Running total for §3.3–§3.6: 8,504 → 4,486 words, a 47 % reduction.

## Source audit and corrections

Two report-only subagents audited sources and duplication; the controller verified every finding against the source before applying it. All 12 unique citation keys in this batch resolve to a bib entry and to a real PDF in `docs/`: `lu2015`, `ben2021`, `liX2018`, `liX2013`, `zhaoS2021`, `yan2014`, `qu2016`, `liong2019b`, `liong2019a`, `bai2021`, `xia2020b`, `xia2020a`. **No fabricated reference.** Bib metadata is correct throughout; the only gap is `ben2021` lacking volume and pages, because the supplied PDF is the IEEE early-access version.

`pdftotext` corrupts three tables this audit depends on. Read from the text layer alone, STSTNet's Table V appears to attribute 2.77 M parameters to AlexNet. Every table- and formula-derived figure was therefore re-checked by rendering the PDF page as an image. On that basis the thesis's numbers are exact: STSTNet's 0.00167 M / 2 layers / 28 × 28 × 3, OFF-ApexNet's 2.77 M, GoogLeNet's 7 M, UF1 0.8382, SMIC's 48.78 %, the 5.6-point TIM–Newton margin, the 30-or-60 optimum. The internal figures were recomputed from the project's own data: 6600/N effective frame rate, 100 fps at the median clip, 52 fps at the longest, 81 of 156 clips downsampled by ≥ 2×, 3 clips of 31 frames, 14,544 backbone parameters and 15,027 with the head.

**Six false absence claims were corrected.** Five trace to one paper the original §3.5 never engaged, `xu2017`:

- "No paper in the corpus examines the interaction between frame synthesis and motion extraction" — false. Xu et al. extract flow from an interpolated sequence *and* choose linear interpolation over TIM because it "suffices to characterize the motion pattern with less error" at a high frame rate. Their reasoning **supports** this thesis's no-synthesis decision, so the section now cites it rather than claiming the ground is empty. They assert rather than measure the point, and the text says so.
- "The trade-off between normalisation and acquisition frame rate goes unremarked" — contradicted by the same paper.
- "Two studies interrogate interpolation length" — three do. Xu et al. sweep 10/15/20 frames over four datasets, which also refutes "the one systematic comparison is single-corpus": theirs is the only multi-corpus one.
- §3.5.5's length enumeration omitted the 120 frames §3.5.3 attributes to `qu2016` two subsections earlier. Now reads 2, 10, 11, 15, 20, 30, 60, 110, 120.
- §3.6.5's "None reports what happens if the stem is removed entirely" is literally false; the corpus compares descriptor-only against learned methods on identical folds. Reworded as the matched-pair claim, which is true and is what the thesis actually needs.

**One correction runs in the thesis's favour and touches the divergence ledger.** §3.6.6 called 224 × 224 "the clearest unforced departure from the literature". Two papers in the review corpus use exactly that resolution — Bai et al. resize frames to 224 × 224, and without alignment or cropping, as this pipeline does; Li, Huang and Zhao resize faces to the same — and Xia et al.'s own earlier work runs a stream at 300 × 245. The departure is from **Xia et al.'s ≤ 100 × 100 guidance**, which is a composite-database result that bites hardest on models deeper than this one, not from the corpus. The ledger row stands; its framing is corrected in both the prose and `tab:cnn-decisions` row 5. **This changes how the divergence should be described in §3.10's ledger, §5.5 and §6.2 — review those when those parts are rewritten.**

Other repairs: a quotation ellipsis in §3.6.3 had dropped "(Model 2, Model 3 and Model 4)", the parenthetical stating that resolution degradation reaches the *shallow* model Xia et al. adopt — restored, because §3.6.6 leans on the shallow-robustness reading. §3.6.4's "three strategies, each with a counterpart in this thesis" was false, as the few-shot scheme has no counterpart. §3.6.5's "sensitivity strong enough to reverse a comparison" asserted more than §3.6.3 shows. §3.5's `\cite{liong2019a}` was attributed to "Liong et al." although the corrected bib records Gan as first author; parts 5–6 refer to OFF-ApexNet by method name, following the parts-3/4 convention. A scope caveat deleted in an intermediate draft — that onset/offset-bounded sampling holds on the image-folder path and not the loader's `.avi` branch — was restored after a grep confirmed it existed nowhere else in the thesis.

An unattributed compute claim ("convolving a 224 × 224 × 32 volume is the pipeline's most expensive operation"), the residue of a Chapter 5 result removed earlier, was replaced by the arithmetic that supports it.

## Validation

Passed: citation-key preservation (no key lost from either section), bibliography membership, all 19 labels preserved and unique across parts 1–6, every `\autoref` target resolving, brace and environment balance, no chapter or numbered-section signposts, no Chapter 5 result figures, and short captions on both tables. `ben2021` retains 5 cite instances and `lu2015` 6 — both are §3.5-only keys and both would orphan if the section were cut further. `xia2020a`'s first occurrence in the whole thesis is §3.6.2's STRCN sentence; it was preserved in place, because moving it renumbers the reference list.

No master `.tex`, class file or TeX engine is present in this checkout, so compilation, float placement, list generation and pagination remain to be verified in the real project.

Audit reports are in the session scratchpad under `reports/`. Continue with parts 7 and 8 (§3.7 SimAM, §3.8 transformer) using the same report-only source and duplication review, then controller editing. Two items carry forward: `bai2021` reports the 1/5–1/25 s duration rule **and then rejects it** as noisy, so any 5–25 Hz derivation attributed to it needs that caveat; and the 224 × 224 reframing above must reach the divergence ledger in part 10.
