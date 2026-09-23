# Handover — manuscript assembled and audited; next task is minor text edits

**Written 23 September 2026.** Supersedes `HANDOVER_FORMULA_APPENDIX.md` and the
older `HANDOVER_CH3_*` files for anything concerning the current manuscript.
Those remain accurate about how the shortening was originally done, but their
file lists and their warnings about section numbering are now out of date — see
§5.1.

Read §1 for the manuscript as it stands, §2 for what changed, §3 for the file
inventory, §4 for the next task, §5 for the traps.

---

## 1. The manuscript

### 1.1 The six files, in order

This is the whole thesis. Nothing else is included.

| # | Chapter | File | Words | Sections | Tables | Figures |
|---|---|---|--:|--:|--:|--:|
| 1 | Introduction | `contents/introduction.tex` | 1,056 | 2 | 0 | 0 |
| 2 | Background | `contents/chapter2_shortened.tex` | ~5,000 | 10 | 3 | 1 |
| 3 | Literature Review | `contents/chapter3_cut.tex` | 10,200 | 10 | 5 | 1 |
| 4 | Methodology | `contents/chapter4_shortened.tex` | 6,569 | 8 | 5 | 0 |
| 5 | Results | `contents/chapter5_shortened.tex` | 2,750 | 8 | 10 | 9 |
| 6 | Conclusion | `contents/chapter6.tex` | 2,081 | 4 | 0 | 0 |
| | **Total** | | **~27,300** | **42** | **23** | **11** |

Chapter 3 is 38 % of the manuscript and remains the outlier. Chapter 2 now
includes the end-to-end pipeline section (§2.10), which is why it carries a
figure and a third table.

### 1.2 Verified state, as of this handover

Run `python tools/check_manuscript.py` from the repo root to reproduce all of
this:

- **236 labels, zero duplicates.**
- **126 referenced targets, zero unresolved.** Every `\autoref` resolves, and
  every one points at the right kind of object (no `tab:` label on a figure).
- **30 bibliography entries, 27 cited, zero invalid keys.**
- **Every float carries a `\label` and a short `\caption[...]`**, so the List of
  Tables and List of Figures generate themselves.
- **All 11 images resolve** in `report_figures_thesis/`.
- **No duplicated prose.** No sentence of nine words or more appears in two
  chapters; no near-duplicate pair above 85 % token overlap.

### 1.3 Integrity, verified against sources

- **No ghost references.** All 30 entries are real. Each was matched to its PDF
  in `docs/` (title and full author list), and then to the published record —
  28 via the Crossref DOI registry, and the two ML conference papers (SimAM at
  ICML, ViT at ICLR, neither of which registers DOIs) via their proceedings.
- **All 28 direct quotations are verbatim** in the paper they are attributed to.
- **Every number attributed to a source is correct**, along with the arithmetic
  derived from those numbers.
- **No plagiarism found.** Only 0.02 % of the manuscript's 8-word sequences
  appear in any of the 30 cited papers, and those matches are a paper title and
  a list of channel names. A sentence-level comparison (~1,500 thesis sentences
  against 11,068 source sentences) found no close paraphrase. Twelve
  exact-phrase web searches on the highest-risk sentences returned no match.

**The limit on that last point:** the web check sampled 12 sentences through a
search engine, not a plagiarism database. It cannot see paywalled journals or
student-paper repositories. **Turnitin or iThenticate is still the definitive
check** — ask your supervisor for a pre-submission report. Given the source
comparison came back at 0.02 %, expect it to be clean.

---

## 2. What this session did

1. **Built the complete formula guide.** `contents/formula_guide_complete.tex`
   is one standalone document covering every formula in all six chapters — 40
   numbered equations, none repeated, with symbols, mechanism, source and
   implementation status for each. Chapters 1, 5 and 6 contain no displayed
   mathematics, which it states rather than padding.
2. **Wrote the pipeline overview**, now §2.10 of Chapter 2: a two-stage
   flowchart, a shape-trace table, and a short walkthrough.
3. **Cut Chapter 3** from 1,155 to 514 lines (14,781 → 10,200 words), per
   `CH3_REDUNDANCY_REVIEW.md`. 17 tables → 5, 2 figures → 1, 7 display
   equations → 0, 76 subsections → 27. All 24 externally-referenced labels
   preserved.
4. **Audited the whole manuscript** and fixed: two Chapter 6 references that
   pointed at content the cut had deleted, and two stale front-matter
   inventories (`list_of_tables.tex` named 12 tables that no longer existed).
5. **Acted on `uncited-paragraph-audit.md`.** Of 59 flagged paragraphs, 12 were
   genuinely missing a citation and were fixed using existing keys; 2 were false
   positives; 37 were the author's own work, misfiled by the keyword heuristic.
6. **Ran the plagiarism and ghost-reference audit** described in §1.3, and
   corrected two bibliography entries.

---

## 3. File inventory

### 3.1 Active — part of the thesis

```
contents/introduction.tex
contents/chapter2_shortened.tex      (includes the pipeline section)
contents/chapter3_cut.tex
contents/chapter4_shortened.tex
contents/chapter5_shortened.tex
contents/chapter6.tex
contents/list_of_tables.tex          front matter, inventory regenerated
contents/list_of_figures.tex         front matter, inventory regenerated
bib/thesis.bib                       30 entries
```

### 3.2 Active — separate documents, not part of the thesis

```
contents/formula_guide_complete.tex  compile standalone; bibliography embedded
```

### 3.3 Superseded — do not include

| File | Superseded by |
|---|---|
| `contents/chapter3_shortened.tex` | `chapter3_cut.tex` |
| `contents/chapter2.tex`, `chapter3.tex`, `chapter4.tex`, `chapter5.tex` | the `_shortened` / `_cut` files |
| `contents/chapter*_short_part*.tex`, `*_half*.tex` | the assembled chapters |
| `contents/appendix_formulas.tex` | `formula_guide_complete.tex` |
| `contents/formula_guide_chapters1_2.tex`, `chapters1_4.tex`, `chapter5.tex`, `chapter6.tex` | `formula_guide_complete.tex` |
| `contents/chapter2_pipeline_overview.tex` | already pasted into `chapter2_shortened.tex` |

These are kept for history. **They share labels with the active files — including
one of them alongside its replacement produces 200+ duplicate-label errors.**

### 3.4 Verification scripts (now permanent, in `tools/`)

Earlier handovers lost these to a temporary scratchpad. They are committed now.
Run from the repo root:

| Script | What it checks |
|---|---|
| `tools/check_manuscript.py` | labels, refs, cites, floats, images, cross-chapter duplication |
| `tools/check_quotes.py` | every quotation against the source PDF text |
| `tools/check_overlap.py` | 8-word overlap against all 30 source papers |
| `tools/check_references.py` | every bib entry against Crossref (needs network) |
| `tools/extract_source_pdfs.py` | rebuilds the PDF text cache the above two need |

`check_quotes.py` and `check_overlap.py` read a cache of the source PDFs' text
in `tools/txt/`. It is already built, and gitignored because it is 1.8 MB of
other people's papers. If it goes missing, `python tools/extract_source_pdfs.py`
rebuilds it in about a minute.

One known false negative in `check_quotes.py`: it reports 27 of 28 quotations
verified. The 28th ("might not be enough to reveal the ME motion progress") **is**
correct — Li et al.'s PDF splits that sentence across a page break with a figure
caption in the middle, so the text layer never contains it contiguously.
Verified by hand.

---

## 4. The next task: minor text changes

Straightforward editing. What matters is knowing which file owns each chapter
and what not to disturb.

### 4.1 Where to edit

Edit the file named in §3.1 directly. **Do not edit `chapter3_shortened.tex` or
the `_part` / `_half` files** — they are no longer the source of truth, and
editing them changes nothing that compiles.

### 4.2 Paste corruption — the one mechanical trap

An earlier browser paste replaced a whole table with a literal `<br>`, which
rendered in the PDF as the glyph pair `¡br¿`. If you paste text through a
browser, check for it afterwards:

```bash
grep -n '<br>\|&nbsp;\|&amp;' contents/*.tex
```

### 4.3 After editing

```bash
python tools/check_manuscript.py
```

Anything under `UNRESOLVED`, `duplicates`, `invalid keys`, or `*** MISSING` is a
real problem. Some known-benign noise: the "hard-coded section numbers" section
flags ordinary decimals such as `22.03` and `1.60`, and the tabular column
checker misreads `@{}lcc@{}`-style specs. Both are false positives.

### 4.4 Three things to leave alone unless you mean it

- **Display equations in Chapters 2 and 4.** `formula_guide_complete.tex`
  reproduces them verbatim. If you change one, change it there too.
- **The first column of Chapter 3's divergence-ledger table.** It is `\ref{}`
  now, not hard-coded numbers, so it maintains itself. Leave it as `\ref{}`.
- **Direct quotations.** All 28 are verified word-for-word against the sources.
  Re-wording one silently makes it a misquote.

---

## 5. Traps

### 5.1 Two stale warnings from earlier handovers — both now false

- `HANDOVER_FORMULA_APPENDIX.md` §4.1 and the old Chapter 3 header warn that
  **Chapter 2 cites Chapter 3 subsections by number** and that section positions
  are therefore frozen. **This is no longer true.** The shortened Chapter 2 has
  no by-number references. Verified by scan.
- The same files warn that the **divergence ledger cites twelve subsections by
  number**. It no longer does — that column is `\ref{}`.

Section numbering in Chapter 3 is therefore free. You can add, remove or reorder
subsections without breaking anything, as long as the labels in §5.2 survive.

### 5.2 Labels other chapters depend on

**24 Chapter 3 labels** are referenced from the introduction, Chapter 4 and
Chapter 6 — Chapter 6 alone uses 16. `check_manuscript.py` catches a break
immediately. Chapter 6 is the most reference-dense file in the thesis and has no
shortened variant, so it is the usual casualty of a Chapter 3 edit.

### 5.3 `main.tex` problems, still unresolved

These are in the Overleaf project, which this checkout does not contain:

1. **`\bibliography{bib/example}`** — `bib/example.bib` does not exist here. It
   should almost certainly be `bib/thesis`.
2. **`\nocite{*}`** — forces every bib entry into the bibliography whether cited
   or not. With it in place, the three uncited entries below will print.
3. **`\graphicspath{{figures/}}`** — needs the second entry to work in this
   checkout: `\graphicspath{{figures/}{report_figures_thesis/}}`.
4. **TikZ** must be loaded for the pipeline figure, and it is:
   `\usepackage{tikz}` plus
   `\usetikzlibrary{arrows.meta, positioning, fit, backgrounds}`.

### 5.4 Three uncited bibliography entries

`adegun2020`, `li2022survey` and `xie2022` are cited only in
`contents/task_description.tex`, which is not part of the six-chapter flow.
Either cite them in the thesis or remove `\nocite{*}`, or they appear in the
bibliography unused.

### 5.5 A pending decision: citing the originators of standard techniques

Eight paragraphs describe standard techniques whose originators have **no entry**
in `thesis.bib`: batch normalisation, sinusoidal positional encoding, label
smoothing, ReLU/GELU, backpropagation, convolution, dropout/LayerNorm, pre-LN
transformers. The bibliography's stated policy is to attribute through the
reviewed paper and add no outside entries, and that policy was held.

**The weak spot is the printed equations.** Batch normalisation and the
positional encoding appear as displayed formulas with no citation at all, and
no existing key can fix them honestly — citing Dosovitskiy for the sinusoidal
encoding would be wrong, because ViT uses *learned* positional embeddings.

Recommendation: add three entries — **Vaswani et al. (2017)**, **Ioffe &
Szegedy (2015)**, **Szegedy et al. (2016)** — covering every printed equation
that currently lacks an originator, and leave the textbook concepts uncited,
which is normal in a thesis. Note that adding entries does **not** break
numbering: BibTeX with `unsrt` renumbers automatically. Only hand-maintained
files such as `BIBLIOGRAPHY_NUMBERED.md` go stale.

### 5.6 This checkout may lag Overleaf

Editing has happened in both places. The Overleaf project contains
`contents/global/` and `figures/`, neither of which exists here. If the two
disagree, Overleaf is the one that compiles. Everything in §1.2 and §1.3 was
verified against **this** checkout.

### 5.7 The standing rule

**Any factual change is a defect, including one that looks like a correction.**
If a formula, number or quotation looks wrong, check the source PDF in `docs/`
before touching it — and prefer reporting it to fixing it. `pdftotext` mangles
equations routinely; render the page as an image before concluding anything.
Two examples from this session: a quotation that appeared to be missing was
split across a page break by a figure caption, and Yan et al.'s "640×480"
extracts as "6406480" because the PDF encodes `×` as a `6`.

---

## 6. Also outstanding

1. **Chapter 3 is 1,700 words over** the 8,500-word target of
   `CH3_REDUNDANCY_REVIEW.md`. Every structural redundancy the review named is
   gone; what remains is unique evidence and the qualifications the review
   itself insists on keeping. Going further means cutting one or the other.
   Sections 3.1 (+464) and 3.10 (+223) hold the most room.
2. **Chapter 6 needs a consistency pass.** It repeats some older broad claims
   that the shortened chapters have since narrowed. Line 24 still carries the
   absolute claim that "No reviewed study isolates a *learned* component …
   against an otherwise identical pipeline", which Xia et al.'s RCN variants
   contradict; Chapter 3 now states the narrower, defensible version.
3. **A compiled page count** has never been measured. The 100-page goal cannot
   be assessed from source word counts.
