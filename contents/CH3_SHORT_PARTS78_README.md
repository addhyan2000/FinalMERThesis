# Shortened Chapter 3: parts 7 and 8

Completed sections **3.7 (SimAM)** and **3.8 (the temporal transformer)**. Parts 5 and 6 were read for continuity and remain in place. Part 5 received one byte-level syntax repair: an ASCII BEL character followed by `utoref` was restored to `\autoref{sec:evm-departures}`. No prose in that part was rewritten.

## Inclusion

Insert this after section 3.6 and before section 3.9, replacing the original sections 3.7–3.8:

```tex
\input{contents/chapter3_short_parts7_8}
```

The wrapper includes the two new section files. Compile from the thesis root. It supplies no chapter declaration or counter overrides; do not include the original sections alongside the replacements because their labels are shared. Parts 9 and 10 still require shortening. Retain their original text when assembling a complete draft in the meantime.

## Length and retained material

Using the same approximate prose counter as the parts 3–4 validation (comments, equations, citation/label commands and LaTeX command names excluded; table prose and captions included):

- Part 7: **2,152 → 964 words**, 55.2% shorter.
- Part 8: **1,949 → 968 words**, 50.3% shorter.
- Combined: **4,101 → 1,932 words**, 52.9% shorter.

The method differs from the parts 5–6 README's float-excluding count; do not sum their published totals without recounting all parts consistently. These are source estimates, not rendered page counts.

Retained: all **21 labels**, eight SimAM subsections, seven transformer subsections, four tables, the reciprocal-energy/gating expression, the six SLSTT divergences, conditional matched-pair scope and all substantive follow-up experiments. Implications remain at 3.7.7 and 3.8.6, preserving the numbered references in the divergence ledger.

Repeated mechanisms, operator inventories, sweeping claims of absence and thesis-result commentary were removed. Neither new section contains chapter signposting or reports this thesis's measured component effects.

## Tables, figures and acronyms

The four existing tables retain their labels and short captions for automatic List of Tables entries:

- `tab:attention-modules`: all six module rows, compressed to parameter costs.
- `tab:simam-decisions`: implementation and scope.
- `tab:slstt-results`: all eight selected source rows and numeric columns retained.
- `tab:slstt-divergences`: all six differences retained.

The original sections contain no figures. None was removed or invented. Keep the thesis master's `\listoftables` and `\listoffigures`; the earlier parts' figures remain unaffected.

Merge `chapter3_short_parts78_acronyms.tex` into the existing List of Acronyms once, removing any duplicate entries. It uses a portable description list because the master glossary setup is not available here. Adapt entries to `glossaries` or `acronym` declarations if the real master uses those packages. EMR is **Expression Magnification and Reduction**, not the title of Liu et al.'s paper.

## Source and scope audit

Two report-only subagents checked source evidence and compression/cross-chapter consistency; the controller inspected the reported findings and edited the text. Both reviewed the final drafts. All seven unique citation keys resolve to supplied PDFs and the existing bibliography: `yang2021`, `xia2020b`, `zhang2022`, `dosovitskiy2021`, `liong2019a`, `liong2019b`, and `liu2019`.

Important clarifications incorporated:

- SimAM's printed shared variance divides by **M**, but its Figure 3 code divides by **M−1**. The project follows the code. Both PDF pages were visually inspected rather than relying on extracted equation text.
- Yang et al. select **10⁻⁴ on CIFAR and 0.1 on ImageNet**. The inherited setting here was not re-tuned. Zero learned parameters does not imply zero computation, immunity to overfitting or guaranteed suppression of neutral frames.
- Xia et al.'s supporting parameter-free argument concerns **composite-database** recognition and a different module; it is not a SimAM experiment.
- SimAM's no-CNN exclusion belongs to this implementation, not to the mathematical definition of attention.
- The no-CNN path includes a learned linear projection after spatial pooling. The temporal-encoder comparison keeps that path fixed within each matched pair.
- SLSTT's **0.844 → 0.901** CASME II UF1 difference is a published aggregation comparison. It cannot be transferred as an expected 5.7-point gain here: this thesis averages tokens **after temporal self-attention**, whereas SLSTT-Mean averages spatially encoded frames.
- SLSTT reports both separate-database and joint three-class evaluations under its CDE heading, plus a distinct five-class CASME II SDE. The heading alone does not prove every per-database result was trained jointly. These protocol differences are retained without treating the source results as directly comparable to the present 156 clips.
- Long-term flow is argued for and used throughout SLSTT's experiments, rather than isolated in a matched short-term-flow ablation. Testing alternative flow requires re-training and evaluation.

The existing `zhang2022` journal year, volume and pages were confirmed by the [author institution's publication record](https://research-portal.st-andrews.ac.uk/en/publications/short-and-long-range-relation-based-spatio-temporal-transformer-f/). Bibliography entries were not changed. OFF-ApexNet's known preprint/final author-order difference remains handled by using its method name and the existing stable key.

This is a substantive source audit of parts 7–8, not a new claim-by-claim audit of every chapter or all 27 bibliography entries.

## Verification and remaining integration work

Passed: citation-key preservation and first-appearance order for this replacement; 21 labels and subsection positions; balanced braces/environments; all four short captions; no new unresolved references or duplicate labels in both the original-with-replacements and shortened-parts-1–8 assemblies; all 27 bibliography keys still used; no chapter signposting or thesis-result leakage. Existing Markdown bibliography and cross-reference gates also pass. The repeatable new-section validator and incremental reports are under `tmp/ch3_shortening/` (scratch, normally ignored by Git).

Full LaTeX compilation, float layout, list generation and pagination remain unverified: this checkout has no master document/class files and no available TeX engine. Rebuild the bibliography and compile the complete thesis to convergence in the actual thesis project.

For part 10 and later whole-thesis editing, carry the revised scope forward. Preserve all twelve divergence-ledger rows, but avoid reviving claims that SimAM cannot act without a CNN, that its gate guarantees neutral-frame suppression, that the transformer is the only component with whole-clip access, that SLSTT's LSTM margin transfers here, or that follow-up flow experiments cost no training time. Existing Chapter 6 prose still uses some of those older cost descriptions. Retain the prior part-6 clarification that 224×224 departs from Xia et al.'s specific guidance rather than the entire literature. None of those later sections was silently rewritten in this batch.
