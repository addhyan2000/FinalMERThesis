# Chapter 5 shortening — completed in two halves

The complete replacement is `chapter5_shortened.tex`. It physically contains both editing halves and a brief chapter introduction. The original `chapter5.tex`, result files and figure images are unchanged.

## Length and scope

- Half 1, sections 5.1–5.4: **1,388 approximate words**.
- Half 2, sections 5.5–5.8: **1,778 approximate words**.
- Combined, including the introduction: **3,259 words**, compared with **6,938** in the original: **53.0% shorter**.

Both versions use the same estimator as the Chapter 2 check: headings, captions and table prose count; comments, mathematics, numeric-only tokens, citation/label keys and LaTeX command names do not. These are approximate source-prose counts, not rendered PDF or TeXcount measurements.

Section counts, original → shortened:

- 5.1 Overview: 868 → 443.
- 5.2 Temporal transformer: 672 → 348.
- 5.3 Motion magnification: 749 → 291.
- 5.4 Parameter-free attention: 728 → 306.
- 5.5 Convolutional stem: 941 → 426.
- 5.6 Per-class performance: 986 → 441.
- 5.7 Published comparison: 892 → 469.
- 5.8 Summary and limitations: 959 → 442.

Opening text accounts for the remainder. All **eight sections, 39 subsections and 67 labels** remain in their original order. Two subsection titles were tightened to avoid overstating the importance of a pair or claiming configuration 2 is the most even configuration overall.

## Integrate the combined chapter

Replace the original Chapter 5 inclusion with:

```tex
\include{contents/chapter5_shortened}
```

Do not also include the original chapter or either half. Their labels are intentionally shared. Include from the repository root, and retain the existing bibliography at `bib/thesis.bib`. The chapter uses `graphicx` and the existing support for `\checkmark` (normally `amssymb`); it does not introduce a custom document class or bibliography system.

All nine figures use explicit `figures/fig5_*.png` paths, matching the existing `figures/` folder in your Overleaf project. The local source PNGs remain under `report_figures_thesis/`; no image re-upload or folder renaming is needed in Overleaf. Replace the contents of your Overleaf `contents/chapter5.tex` with the combined shortened file if that is the filename your master already includes.

## Lists of tables, figures and acronyms

The current shared `list_of_tables.tex` generates **both** the List of Tables and List of Figures, preserving the combined arrangement already present after the Chapter 4 work. It now documents all Chapter 5 float labels as well as the Chapter 2 entries. Include it once in the front matter:

```tex
\input{contents/list_of_tables}
```

Do not additionally include `list_of_figures.tex` or duplicate the master's existing list commands. The standalone figure-list file is an alternative only. LaTeX gathers every included chapter's short captions and generates numbers/page references automatically after compilation to convergence; the commented inventory is not a manually numbered list.

The current consolidated acronym file contains entries only. If the master does not already supply its heading:

```tex
\chapter*{List of Acronyms}
\addcontentsline{toc}{chapter}{List of Acronyms}
\input{contents/list_of_acronyms}
```

Existing entries were retained; missing Chapter 2 entries and the Chapter 5 terms CK+, SAMM and GPU were added without duplicate labels. `chapter5_short_acronyms.tex` is an optional Chapter-5-only merge list for an external master. Do not include that companion alongside the consolidated list. These instructions describe the **current** shared files and supersede older inclusion snippets that treated the table and figure lists separately or placed the acronym heading inside its file.

## Evidence preserved and verified

- **Ten tables and nine figures retained**, with original labels and short captions. Every float has a prose reference. Table data, flags, pair identities and numerical order are unchanged. Long configuration names were shortened to unambiguous `config_N` identifiers in the two wide tables, and column headings were compacted.
- All twelve configurations' pooled macro F1, pooled accuracy, correct counts and class scores were recomputed from their stored confusion matrices. Every matrix has class support 99/32/25 and 156 clips; all records cover 25 LOSO folds.
- All twenty pair differences and the four component means were recomputed from unrounded values. Their displayed four-decimal values match the original tables.
- The all-Negative reference and the 0.6267 mean-fold ceiling were checked. Costs were recomputed from retained `configuration_summary.txt` timings: 50.61 estimated GPU-hours overall, 48.87 for stem-bearing configurations, 96.6% of the total. Their extrapolated nature remains explicit.
- All nine existing figures were visually checked against the results and source values. The caption for the component-effects plot clarifies that its SLSTT/3D-CNN panel names refer to the implemented switches, not faithful reproductions of the original architectures.

Two report-only subagents audited the source facts and reviewed the completed halves. Their reports are retained under `tmp/ch5_shortening/` for local review; the delivered chapter does not depend on those temporary files.

## Reference verification

All **six original citation keys remain in their original first-use order** and refer to papers supplied in `docs/`:

- `see2019`: MEGC 2019 challenge summary, the actual source for all seven published comparison scores.
- `liong2019a`: OFF-ApexNet; the method name avoids differing author order between the supplied preprint and final bibliography.
- `liong2019b`: STSTNet.
- `liu2019`: A Neural Micro-Expression Recognizer, EMR.
- `liong2018`: Less Is More, Bi-WOOF.
- `zhaoG2007`: Dynamic Texture Recognition Using Local Binary Patterns, LBP-TOP.

The CASME II UF1 column of See et al.'s Table IV was visually verified: 0.8764, 0.8621, 0.8382, 0.8293, 0.7805, 0.7068 and 0.7026 in the chapter's order. The two daggered methods, Zhou and Quang, remain attributed through See et al.; no unsupported standalone bibliography entries were added. Quang's method is correctly described as CapsuleNet. Numeric reference numbering remains controlled by the master's bibliography style.

## Wording corrections made while compressing

The new text distinguishes descriptive results from unsupported causal or statistical conclusions:

- Six positive transformer comparisons are consistent observations, not independent statistical replications or a confidence interval.
- A small SimAM mean does not establish statistical equivalence to zero.
- EVM's largest absolute change is the +0.0795 gain; −0.0541 is its largest loss.
- Configuration 9 has the second-lowest Surprise F1, and configuration 2 is most even only within the transformer group.
- A one-clip change has no fixed F1 increment; aggregate counts do not reveal which clips changed predictions.
- Nonzero class scores do not prove the sampling guard caused them.
- The no-CNN control retains a learned projection. Spatial convolutional kernels do not make the complete stem temporally independent because its statistics can span time.
- Published comparisons use different training/evaluation populations; EMR additionally uses CK+ data. No guaranteed cross-corpus benefit or fair-protocol ranking is inferred.

Single-seed evaluation, checkpoint selection on the reporting subject, missing per-clip predictions, demographic limits, fixed hyperparameters, restricted component variants and incomplete timing records remain explicit.

## Repeatable checks and remaining limits

After editing either half:

```text
python tools/assemble_short_chapter5.py
python tools/check_short_chapter5.py
```

The checker verifies exact assembly, original labels and citation order, every table data row, source-result arithmetic, the nine Overleaf image paths and their local source files, automatic-list commands, acronym coverage and static LaTeX syntax. Integration passed with all eight combinations of original/shortened Chapters 2, 3 and 4, alongside the introduction and Chapter 6, without introducing unresolved references or duplicate labels. The repository bibliography and cross-reference gates also passed.

No master document or LaTeX engine is available in this checkout. Final float placement, bibliography rendering, generated list pages and total pagination must be checked when compiling the full thesis. Keeping all nineteen floats protects the results evidence but means a 53% prose reduction will not necessarily halve the chapter's page count.
