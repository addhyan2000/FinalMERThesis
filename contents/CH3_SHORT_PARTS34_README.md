# Shortened Chapter 3: parts 3 and 4

These are sections **3.3 (optical flow)** and **3.4 (optical strain)** of the ten-part Chapter 3 shortening task. They are not replacements for thesis Chapters 3 and 4. The original `chapter3.tex` and existing parts 1–2 in `chapter3_short.tex` were not edited.

## Use the new text

Replace the original sections 3.3 and 3.4 with:

```tex
\input{contents/chapter3_short_parts3_4}
```

The wrapper includes `chapter3_short_part3.tex` and `chapter3_short_part4.tex`. It belongs after section 3.2 and before section 3.5, inside Chapter 3. It intentionally contains no chapter declaration, document preamble or counter overrides. Compile from the thesis root. Do not include the original versions of sections 3.3–3.4 alongside these replacements: they use the same labels.

For the staged shortened version, the wrapper follows the existing `chapter3_short.tex`, which currently ends with part 2. Parts 5–10 still need shortening; retain their original text in a complete thesis build until their replacements are ready. The existing parts 1–2 were read for continuity but were not re-audited in this pass.

## Tables, figures, acronyms and numbering

- All three existing tables remain, with short captions for the List of Tables and their original labels: `tab:flow-decisions`, `tab:strain-performance`, and `tab:strain-decisions`. The performance table now also includes the source paper's OFF + OFW control.
- No figures occurred in the original sections 3.3–3.4, and none were removed or invented. The dataset figures in parts 1–2 remain in their existing file. Keep the master document's `\listoftables` and `\listoffigures`; entries are generated from captions on compilation.
- `chapter3_short_parts34_acronyms.tex` supplies portable description-list entries for the List of Acronyms. Merge them once with the thesis's existing entries, removing duplicates. If the master uses `glossaries` or `acronym`, translate these entries to that package's declarations rather than inserting a second list.
- All eight subsection slots in each part are preserved, including the positions referenced numerically elsewhere (3.3.1, 3.3.3 and 3.4.3). All 21 section/subsection/table labels survive.
- Citations use stable BibTeX keys. First-appearance numbering may change after shortening: rebuild the bibliography and compile the complete thesis to convergence, including both lists. Do not manually preserve old printed table or reference numbers.

## Reduction and scope

A consistent approximate prose-word count, excluding comments, equations, citation/label commands and LaTeX command names, gives:

- Part 3: 2,582 to 1,141 words, a 55.8% reduction.
- Part 4: 2,803 to 1,224 words, a 56.3% reduction.
- Together: 5,385 to 2,365 words, approximately 56% shorter.

These are source-based estimates, not TeXcount or rendered page counts. Float placement, captions, equations and the master template affect the actual page reduction. No claim is made that this batch alone reaches the overall 100-page target.

The shortened text retains the analytical-feature interpretation, the strain operator and shear-weighting divergence, the measured literature evidence and protocol caveats, and the untested estimator/alignment/strain/filtering experiments. It removes repeated mechanisms, chapter signposts, broad absence claims and the thesis's own results from these literature-review sections.

## Source audit and corrections

Two report-only subagents checked sources and duplication/cross-chapter dependencies. The controller reviewed their findings and applied the edits. All nine unique citation keys in this batch correspond to PDFs supplied in `docs/`: `xu2017`, `see2019`, `liong2019a`, `liong2019b`, `zhaoS2021`, `shreve2011`, `liong2014a`, `liong2014b`, and `liong2016`. This is a substantive audit of these two sections, not a new claim-by-claim audit of all 27 thesis references.

Important source findings incorporated:

- STSTNet's printed equation was inspected as an image. Its repeated mixed derivative is defective; retaining the printed factor outside the square while repairing the derivatives yields the four-term norm. The text no longer attributes the implemented three-term form unambiguously to STSTNet.
- The early optical-strain paper's equation already averages over frame pairs. The purported historical change from sum pooling to mean pooling was removed.
- OSW weights the XY-plane LBP-TOP histograms, not all three planes.
- Liong 2016's Table 8 includes a matched flow-descriptor control: 55.87% versus 63.16% CASME II accuracy, a 7.29-percentage-point difference. The 0.41–1.22-point improvement describes comparison with the texture baseline, not this control. Its prose reverses the block-size ordering implied by Tables 1 and 2; the new text uses the range without reproducing that contradiction.
- Flow is not asserted to be identity-invariant or a closed-form substitute for learned spatial processing. The sequence representation is a design choice, not an ablation against onset–apex input. The study does not isolate strain's marginal benefit.

Two shared bibliography records were corrected without changing their keys:

- `liong2014b`: publication year 2015, with pages, volume and DOI, following the [Springer publication record](https://link.springer.com/chapter/10.1007/978-3-319-16631-5_47). ACCV 2014 is the conference name/date.
- `liong2019a`: the final journal author list begins with Gan, whereas the supplied preprint begins with Liong. The entry now follows the final journal metadata documented by the [author institution](https://scholar.nycu.edu.tw/en/publications/off-apexnet-on-micro-expression-recognition-system/), with DOI added. The new parts refer to OFF-ApexNet by method name, avoiding version-dependent narrative attribution.

`BIBLIOGRAPHY.md`, `bib/thesis.bib` and generated `BIBLIOGRAPHY_NUMBERED.md` are synchronised. The bibliography renderer was run for all six assembled chapters. Older narrative references elsewhere to OFF-ApexNet as “Liong et al.” still reflect the supplied preprint; harmonising those legacy attributions to the final Gan-first publication is an outstanding whole-thesis editorial task, not a change silently applied to the original chapters. Likewise, source contradictions identified here remain in the original long Chapter 3, which is retained for comparison.

## Validation and continuation

Passed: citation-key preservation, bibliography membership, all subsection positions and labels, environment/bracket balance, no new unresolved thesis references or duplicate labels in an integrated replacement, no new orphan bibliography entries, no chapter signposts, and no thesis-result leakage in the two new parts. The existing `bibliography.py check` and `ch3_check.py` gates also pass; those gates primarily cover the Markdown thesis and do not replace the separate new-LaTeX checks.

No master `.tex`, class/style files or TeX engine are present in this checkout/environment. Full compilation, float layout, list generation and final pagination therefore remain to be verified in the actual thesis project.

Working audit reports, inspected equation images and the repeatable validation script are in `tmp/ch3_shortening/` (scratch, normally ignored by Git). Continue with parts 5 and 6 using the same report-only source and duplication review, then controller editing. Carry the substantive future experiments into the shortened synthesis; preserve all twelve declared divergences there.
