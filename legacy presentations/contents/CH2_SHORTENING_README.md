# Shortened Chapter 2 — two halves

Both halves are complete. The original `chapter2.tex` is preserved.

- `chapter2_short_half1.tex`: sections 2.1–2.5, approximately **2,097 words**.
- `chapter2_short_half2.tex`: sections 2.6–2.9, approximately **2,005 words**.
- `chapter2_shortened.tex`: the complete, assembled replacement, including the chapter heading and brief introduction: **4,156 words**, down from **9,358** (**55.6% shorter**).

Counts use the same estimator on both versions. They include headings, table prose and captions, but exclude comments, mathematical expressions, numeric-only tokens, citation/label keys and LaTeX commands. They are approximate prose counts, not TeXcount or rendered PDF counts. Differences from earlier estimates reflect this explicit counting convention.

## Include in the thesis

Replace the existing Chapter 2 inclusion with:

```tex
\include{contents/chapter2_shortened}
```

The assembled file contains both halves physically, so it can be copied as one chapter file. Do not include the original chapter or the separate halves alongside it: they intentionally retain the same labels. Continue using the existing bibliography at `bib/thesis.bib` and the thesis's bibliography commands/style.

For front matter, include these once, where the corresponding lists belong:

```tex
\input{contents/list_of_tables}
\input{contents/list_of_figures}
\input{contents/list_of_acronyms}
```

If the master already calls `\listoftables` or `\listoffigures`, keep those existing calls instead of adding duplicates. The two list files use LaTeX's automatic lists: captions in **all included chapters** supply the entries and page numbers. Rebuild the bibliography and compile to convergence in the full thesis project.

`list_of_acronyms.tex` supplies an unnumbered heading and 50 deduplicated entries combining the new Chapter 2 terminology with the four existing Chapter 3 acronym companion files. It replaces inclusion of those separate companions. If a master acronym list already exists, merge the entries into it instead. `chapter2_short_acronyms.tex` is the smaller, Chapter-2-only merge alternative; do not include both alternatives. A full audit of acronym coverage in every later chapter was not part of this edit.

## Section word counts: original → shortened

- 2.1 Phenomenon: **770 → 328**.
- 2.2 Recording and input: **770 → 313**.
- 2.3 Motion magnification: **965 → 422**.
- 2.4 Flow and strain: **1,036 → 475**.
- 2.5 Neural-network fundamentals: **1,648 → 559**.
- 2.6 Network blocks: **1,213 → 642**.
- 2.7 Training under scarcity and skew: **1,285 → 687**.
- 2.8 Evaluation: **1,117 → 588**.
- 2.9 Conclusion: **361 → 88**.

Chapter opening text accounts for the remainder of the totals.

## What was retained and checked

All nine sections, 40 subsections and 50 original labels remain in their original order. Existing references from the introduction and Chapters 3–6 still resolve; this was checked with both versions of Chapter 3. Chapter previews and numbered cross-chapter signposting have been removed from the replacement.

The chapter retains the phenomenon and annotation definitions; input provenance and selection; EVM rationale and departures; flow and strain equations; sampling, tensor shape and normalisation; neural-network fundamentals; component controls; the training objective; and the evaluation distinctions needed to interpret the thesis results. Repeated motivation, extended textbook explanations and duplicated summaries were compressed.

Two new tables have captions, labels, short list captions and explicit prose references:

- `tab:ch2-evm-departures`: filtering, amplification, colour treatment and preprocessing order.
- `tab:ch2-component-controls`: enabled operations and their disabled-component replacements.

The original Chapter 2 contained no table or figure floats. No existing Chapter 2 figure was removed, and no decorative graphs were added. Results plots remain with the results material.

Two report-only subagents checked sources, implementation facts, downstream dependencies and the completed drafts. All **11 original citation keys** remain in their original first-use order and resolve to real papers supplied in `docs/`: `yan2014`, `qu2016`, `liY2018`, `bai2021`, `liong2019a`, `shreve2011`, `zhaoS2021`, `yang2021`, `xia2020a`, `zhang2022`, `see2019`. No new bibliography entries or manually assigned reference numbers were introduced. Numeric citation numbering remains the responsibility of the master bibliography style.

Accuracy corrections include the aperture-problem direction, small-strain terminology, qualified flow invariance, the learned no-CNN projection, SimAM's reference-code variance and temporal context, permutation-equivariant attention, and the distinction between pooled and mean-fold metrics. The limitations from checkpoint selection on the reporting subject, a single seed and absent per-clip predictions remain explicit.

## Existing cross-chapter discrepancies to reconcile

The Chapter 2 audit also found two existing statements in `chapter3_shortened.tex` that conflict with the implementation. They were not copied into the new chapter or edited as part of this Chapter 2 replacement:

1. Around line 291, the amplitude-magnification paragraph calls the Butterworth-based formulation the implemented variant. The implementation uses a hard Fourier mask. Chapter 2 explicitly distinguishes the published Butterworth filter from that implementation.
2. Around line 677, the physical interval formula uses `N/(200*32)`. For N inclusive original frames, the ideal average interval between 33 endpoint-preserving samples is `(N-1)/(200*32)`; rounded sample indices also make individual intervals nonuniform. Chapter 2 states the timing mismatch without reproducing the erroneous formula.

These are remaining manuscript consistency issues, not unresolved citation keys in the new chapter.

## Rebuild and validation

After editing a half, run from the repository root:

```text
python tools/assemble_short_chapter2.py
python tools/check_short_chapter2.py
```

The checker verifies exact assembly, section/subsection and label preservation, citation keys and order, cross-chapter label integration, both table captions/references, acronym deduplication, balanced braces/environments and approximate section counts.

Static validation passed. This checkout has no thesis master or LaTeX engine, so actual float placement, generated list contents, bibliography rendering and page count still require compilation in the full thesis project. A 55.6% prose reduction does not imply an identical page reduction or establish that the complete thesis now fits within 100 pages.
