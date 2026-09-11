# Chapter 3 Reformatting — Design

**Date:** 11 September 2026
**Status:** awaiting review. Sections S0–S5 were approved in discussion; S6–S8 are new and unreviewed.
**Scope:** `Thesis_Chapter3_LiteratureReview/Tech-wise/` only. Chapter 2 and all other chapters are out of scope for this task.

**Numbering convention in this document.** This spec’s own sections are prefixed `S` (S0–S8). Bare `§3.1`, `§3.1.6` and similar always refer to sections of the *thesis*, never to this document.

---

## S0. Purpose and provenance

Chapter 3 is complete, fact-audited and reference-audited (see the folder `README.md`). This task changes **how it is presented**, not what it says, so that it matches the presentation conventions of two exemplar MSc theses supplied for reference:

- `ImprovingMusicTranscription_Thesis_PrashantVaradarajan.pdf` (BTU Cottbus, 2025) — deep, scaffolded literature review; four numbered sections, two heading levels, repeated unnumbered `Models and Methods` / `Challenges` blocks.
- `main.pdf` — *Comparative Analysis of Molecular Representations for Machine Learning-Based Fuel Property Prediction* (BTU Cottbus, 2026) — flat literature review; six numbered sections, no subsections, closing `Research Gap and Thesis Contribution`.

The exemplars are a reference for **format only**. No content, argument, citation or phrasing is to be drawn from either of them.

### What the two exemplars actually share

They disagree on depth and length, so only their common conventions are binding:

- Numeric bracketed citations, with the author name kept in the running prose.
- The chapter opens with unnumbered framing paragraphs before the first numbered section.
- No blockquotes anywhere.
- Captioned figures, referenced in the prose before they appear.
- Footnotes used for term definitions.
- A closing section that states the gap or conclusion, and a bridging sentence into the next chapter.

### Measured gap against our chapter

| Trait | Varadarajan | Hakari | Chapter 3 as it stands |
|---|---|---|---|
| Citations | numeric `[39]`, `[6, 38]` | numeric `[2]`, `[11, 20]` | author–year, APA |
| Chapter opening | 3 unnumbered paragraphs | 1 unnumbered paragraph | `## Scope and conventions` + `## Contents` table |
| Depth | 4 sections, 2 levels | 6 flat sections | 10 sections, 80+ numbered subsections |
| Sub-subheads | bold, unnumbered | none | numbered `###` |
| Section close | `Challenges` paragraph | prose | `3.x.8 Limitations of the existing work…` |
| Chapter close | `3.5 Conclusion` + bridge | `3.6 Research Gap and Thesis Contribution` | `3.10 Synthesis…`, no bridge |
| Visuals | figures | figures, TikZ flowcharts | 21 tables, no figures |
| Footnotes | term definitions | none | none |
| Length | ~6,000 words | ~2,500 words | 29,445 words |
| Medium | LaTeX | LaTeX | Markdown |

---

## S1. Decisions taken

Four decisions were settled before design, and they bound everything below.

| Decision | Choice | Consequence |
|---|---|---|
| What "same format" covers | **Surface conventions only** | All 29,445 words and every verified fact survive. Subsection numbering is untouched, so no cross-reference breaks. This is a reformatting job, not a rewrite. |
| Citation style | **Numeric `[n]`, name kept in prose, numbered by first appearance** | Matches both exemplars. Creates a deliberate inconsistency with Chapter 2, which remains author–year — see Risk R4. |
| Medium | **Stay Markdown, LaTeX-ready** | Section files and `rebuild_complete.sh` are preserved. Conventions are adopted in forms that convert cleanly to LaTeX later. |
| Figures | **Reuse existing PNGs where they genuinely fit** | No new diagrams authored. Figures will cluster in the thesis’s §3.1; see S5. |

---

## S2. The convention sheet

Ten rules. Each states exactly what changes and what does not. This sheet is the contract every agent in the pipeline is held to.

### C1 — Numeric citations, name kept in prose

- Narrative form: `Yan et al. (2014)` becomes `Yan et al. [1]`.
- Parenthetical form: `(Yan et al., 2014)` becomes `[1]`; multiple sources become `[1, 4]`.
- Year-suffixed pairs (`2019a`/`2019b`, `2014a`/`2014b`, `2020a`/`2020b`) receive **distinct numbers**. The letter suffix disappears from the body text and survives only in the reference list.
- Left untouched because they are not citations: `MEGC 2019`, `FG 2019`, `CASME II`, and every `§3.x.y` or `§2.x` cross-reference.

### C1e — Out-of-corpus attributions receive no number

Works attributed in the text but deliberately absent from the reference list — `Zhao & Pietikäinen, 2007` for LBP-TOP, and the roughly twenty listed in the preamble to `11_References.md` — keep their author–year form and receive **no** bracket. They have no entry to resolve against, so a bracket would create a ghost reference. The Stage 0 map lists them explicitly as a do-not-touch set.

### C2 — Chapter opens with unnumbered framing prose

The `## Scope and conventions` heading, the `## Contents` table and the three bolded policy paragraphs are replaced by three to four unnumbered paragraphs stating what the chapter reviews, how it is organised, and what it argues. The substance of the sources, structure and verification policies moves to the folder `README.md`, which already carries all three; nothing is lost, it simply stops being body text.

### C3 — Per-section blockquotes are removed

The ten `> **Scope of this section.**` blocks become ordinary opening paragraphs beneath their section heading. Neither exemplar uses a blockquote anywhere.

### C4 — Numbered subsections stay; bold sub-subheads are added inside them

`### 3.7.7` keeps its number and title, so every cross-reference survives. Where a subsection runs long and already carries bolded lead-ins such as `**A declaration about the 5-D extension.**`, those are promoted to standalone bold lines on their own — the exemplars' `**Models and Methods**` / `**Challenges**` pattern.

### C5 — Figures

Three existing PNGs, regenerated without their baked-in titles, captioned in the form `*Figure 3.1 — …*` and referenced in the prose before they appear. Numbered sequentially across the chapter. Details in S5.

### C6 — Footnotes for term definitions

Markdown footnotes replace parenthetical definitions where the definition currently interrupts the argument. Both exemplars do this; our chapter does none.

### C7 — Bridging sentences

Each section closes by pointing at the next. The chapter closes by pointing at Chapter 4. Currently absent throughout.

### C8 — §3.10 becomes the chapter's conclusion in form as well as substance

Both exemplars close on such a section. Ours already is one in substance. It takes the exemplars' shorter title form and gains a closing bridge. **Its number does not change**, so references to §3.10.x stay valid.

### C9 — Table captions are already correct

The existing `*Table 3.n — …*` form matches. No change.

### C10 — Explicitly unchanged

Every fact, number, quotation, table, declared divergence, gap statement and implications table. All 80-odd subsection numbers. The verification stamps in each section footer.

---

## S3. Stage 0 — the global citation pass

Numbering by first appearance is a whole-chapter property: §3.7 cannot know what number a reference received in §3.2 unless a map exists first. Stage 0 therefore runs once, before any per-section work.

**Inputs:** the ten section files in reading order; `11_References.md` (27 entries).

**Measured scale:** 27 distinct references, roughly 250–300 in-text occurrences across the section files.

### Step 0.1 — Enumerate

A script extracts every citation-shaped token from `01_…` through `10_…`, in file order then line order. Output is `citation_map.md`: one row per distinct work giving the surface forms it appears in, its first-appearance location, its assigned `[n]`, and the matching reference-list entry.

### Step 0.2 — Classify (human-checked before any edit)

Every extracted token is sorted into one of three buckets, and the map is reviewed before substitution runs. This is the one place where a mistake propagates everywhere.

| Bucket | Action |
|---|---|
| In-corpus — resolves to one of the 27 entries | assign `[n]` |
| Out-of-corpus attribution | do not touch (C1e) |
| Not a citation (`MEGC 2019`, `FG 2019`, `CASME II`, `§3.x.y`) | do not touch |

### Step 0.3 — Known collisions

Two author collisions exist in the reference list and are **already disambiguated in the prose**. The script must preserve, not collapse, the distinction:

- `Li, X.` appears twice — 2013 (SMIC) and 2018 (IEEE TAC survey). The prose writes the latter as `Li et al. (2018)`.
- `Li, Y.` appears twice — 2018 (ICIP) and 2021 (IEEE TIP). The prose writes both as `Li, Huang and Zhao (2018)` / `(2021)`, never as `Li et al.`

### Step 0.4 — Surface forms handled

All seven occur in the text.

| Form | Before | After |
|---|---|---|
| Narrative | `Yan et al. (2014)` | `Yan et al. [1]` |
| Parenthetical | `(Yan et al., 2014)` | `[1]` |
| With locator | `Li et al. (2018, Table 6)` | `Li et al. [6] (Table 6)` |
| Combined years | `Li, Huang and Zhao (2018, 2021)` | `Li, Huang and Zhao [7, 8]` |
| Multi-source | `(Yan et al., 2014; Li et al., 2018)` | `[1, 6]` |
| Ampersand | `(Li, Huang & Zhao, 2018, 2021)` | `[7, 8]` |
| Out-of-corpus, incl. inside table cells | `LBP-TOP (Zhao & Pietikäinen, 2007)` | unchanged |

### Step 0.5 — Substitute

By script, **longest-match-first**, so that `Liong et al. (2019a)` is consumed before any shorter `Liong et al.` pattern can fire. Run on a git branch; every file diff is reviewable.

### Step 0.6 — Renumber the reference list

`11_References.md` is reordered into first-appearance order and each entry prefixed `[n]`. The out-of-corpus preamble paragraph is preserved exactly.

### Step 0.7 — Reconciliation gate

A script asserts, before any per-section work begins:

- count of distinct `[n]` in the body equals the count of reference-list entries equals 27;
- no `[n]` is unused;
- no author–year string remains in the body that is not on the do-not-touch list.

Fails loud. Nothing proceeds until it passes.

---

## S4. The per-section cycle

The five roles in `AGENT_PROMPTS.md` were built for **writing new content from verified facts**. This task **edits already-verified text**, which inverts the risk: the danger is no longer an invented claim but a silently altered one. The roles are repurposed accordingly.

### Shared preamble — two changes

Paste the existing shared preamble from `AGENT_PROMPTS.md`, retaining the **CRITICAL METRIC WARNING** unchanged. An auditor that misreads the `macro_f1` key will "correct" a correct number, which is precisely the failure recorded in §1 of `HANDOVER.md`.

Add this clause:

> **This section is already fact-audited and reference-audited. You are not re-verifying its claims.** Any factual change is a defect, including one that looks like a correction. If you believe a fact is wrong, report it — do not fix it.

Add this clause:

> **Never edit `00_Chapter3_Complete.md`.** It is generated by `rebuild_complete.sh` from the numbered section files.

### Stage 1 — Format audit (report only)

Replaces the scope check. Reads the section plus this convention sheet and returns:

- every deviation from C1–C10, quoted with a line number;
- which long subsections warrant bold sub-subheads, and the exact heading text proposed;
- whether a figure belongs here, which PNG, and the sentence that should reference it;
- which parenthetical definitions should become footnotes;
- where the closing bridge sentence goes and what it points at;
- `VERDICT: n changes required`, category by category, stating explicitly where a category is empty.

Forbidden: editing; and proposing any wording change that is not one of C1–C10.

### Stage 2 — Format writer (edits one file)

Applies the Stage 1 report. Hard constraints in the brief:

- Change no fact, number, quotation, table cell, citation target, section number or gap statement.
- Framing paragraphs and bridge sentences are the **only** new prose permitted, and both must be assemblable from material already present in the section.
- Do not touch `00_Chapter3_Complete.md`.

Reports back: word count before and after, and an itemised list of what was added.

### Stage 3 — three audits, dispatched in one message, in parallel (all report only)

**3a — No-drift audit.** The load-bearing one, and the reason this pipeline differs from the original. Receives the `git diff` for the section plus both versions. Its sole job is to prove that nothing but formatting changed. It enumerates every diff hunk and classifies each as C1–C10, new framing prose, new bridge, or **DRIFT**. Every number, quotation, table cell and technical claim present in the before-text must be present unaltered in the after-text. `VERDICT: NO DRIFT / DRIFT (n)`.

**3b — Citation audit.** The ghost-reference role, repurposed. Every `[n]` resolves to an entry in the renumbered reference list; every name-plus-number pairing is correct against that entry's byline; no out-of-corpus attribution has acquired a bracket; no in-corpus citation remains in author–year form. `VERDICT: CLEAN / PROBLEMS (n)`.

**3c — Cross-reference audit.** Replaces the duplication checker, which would burn budget re-solving a settled problem — duplication was resolved in the earlier consolidation pass and no content moves in this one. Instead: every `§3.x.y` and `§2.x` reference still resolves to a heading that exists; every `Table 3.n` and `Figure 3.n` reference resolves; figure numbering is sequential chapter-wide. `VERDICT: CLEAN / BROKEN (n)`.

### Stage 4 — Orchestrator

Applies fixes; checks every reported error before accepting it, since two of roughly twenty-five findings were wrong in the previous pass; rebuilds; commits the section.

### Batching

Three audits per message. Stages 1 and 2 are sequential and single. Never five heavy agents in one message.

---

## S5. Figures and execution order

### Figures

Two findings constrain this.

**The existing PNGs carry report-register titles baked into the image** — for example "Figure L3 — The 25 LOSO folds are wildly unequal". A thesis figure takes a neutral caption below the image, not an editorial title inside it. All figures used are therefore regenerated title-free from `tools/loso_report_figures.py`.

**Literature-review-appropriate figures exist only for §3.1.** Sections 3.2–3.9 review techniques, which the exemplars illustrate with conceptual diagrams that this project does not have and which are out of scope. Figures will therefore cluster in §3.1 rather than spread evenly. This is accepted: forced figures are worse than none.

| № | Source PNG | Site | Caption |
|---|---|---|---|
| Figure 3.1 | `report_figures_loso/figL14_dataset.png` | §3.1.3 | Composition of CASME II: all 255 released clips by raw label, and the 156-clip three-class pool after "others" is dropped |
| Figure 3.2 | `report_figures_loso/figL3_fold_composition.png` | §3.1.6 | Class composition of the 25 leave-one-subject-out folds, and the resulting per-fold macro-F1 ceiling |
| Figure 3.3 | `report_figures_loso/figL2_metric_definitions.png` | §3.1.6 | Pooled versus mean-of-folds macro F1 — **conditional**, see below |

Figure 3.2 earns its place on argument, not decoration: §3.1.6 argues that the mean-of-folds metric is structurally capped at 0.6267, and that figure derives the 0.627 ceiling from the fold composition directly. It is the chapter's central methodological argument and is currently carried by prose alone.

Figure 3.3 ships only if the Stage 1 agent for §3.1 confirms it is Chapter 3 material and not Chapter 5's. If it does not, the chapter ships with two figures.

### Execution order

**Pilot: §3.9 Class imbalance.** 1,867 words, the smallest section, clean at last audit, no figures. The full cycle runs on it, the diff is reviewed jointly, and the four briefs are tuned before being spent on the remaining nine sections.

**Then §3.1.** 6,064 words, all three figures, the most cross-reference targets in the chapter, and the section every other one defers to. Running it second means the briefs are calibrated before they meet the hardest section.

**Then the remaining eight in file order:** 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.10.

**Then the whole-chapter pass:** consistency editor, `rebuild_complete.sh`, final reconciliation gate.

**Git:** one branch; one commit per section, so any single section can be reverted alone.

---

## S6. Verification gate and failure handling

### Per-section gate

A section is not committed until all three of its Stage 3 audits return clean, or until every finding has been individually checked by the orchestrator and either applied or rejected with a stated reason.

| Audit result | Action |
|---|---|
| 3a reports DRIFT | Revert the hunk to the before-text. Drift is never argued with — the before-text is the audited text. |
| 3b reports a ghost or mispairing | Verify against `11_References.md` and the source PDF before changing anything. |
| 3c reports a broken cross-reference | Verify the target heading actually exists before editing. A broken reference may mean the writer renamed a heading it was forbidden to rename — in which case the fix is to restore the heading, not to repoint the reference. |
| An audit reports a factual error in the original | Do not fix it in this task. Log it in the folder `README.md` under outstanding items. Fact corrections are a separate task with a separate audit trail. |

### Whole-chapter gate, before final commit

1. The Stage 0 reconciliation script passes again against the finished chapter.
2. `rebuild_complete.sh` runs and `00_Chapter3_Complete.md` regenerates without error.
3. Word count of the assembled chapter is within +5% / −0% of 29,445. A drop means content was lost; the framing paragraphs and bridges should push it slightly up.
4. Every `Figure 3.n` is referenced in prose exactly once before it appears, and numbering is gapless.
5. Every one of the 27 reference entries is cited at least once, and every `[n]` in the body resolves.
6. Diff the full before-and-after with the numeric-citation substitution reversed; the remaining diff should contain only C2–C8 changes.

### Rollback

One commit per section means a bad section is reverted alone. If Stage 0's substitution proves unsound, the whole branch is discarded and the map is rebuilt — nothing downstream is trusted, because every section's citation audit was run against that map.

---

## S7. Risks

| № | Risk | Mitigation |
|---|---|---|
| R1 | A writer agent silently paraphrases a verified number or quotation | Stage 3a exists solely to catch this; its verdict is binding and drift is reverted, not debated |
| R2 | Blind regex numbers an out-of-corpus attribution, creating a ghost reference | C1e; the do-not-touch bucket is human-checked in Stage 0 before substitution |
| R3 | `Li et al. (2018)` collapses two distinct works | Documented in Step 0.3; longest-match-first substitution; Stage 3b checks name-plus-number pairing per section |
| R4 | Chapter 2 stays author–year while Chapter 3 becomes numeric | Accepted for now — Chapter 3 only was requested. Logged as an outstanding item so Chapter 2 can be converted in a later task with the same pipeline |
| R5 | Removing the `## Contents` table loses navigability | Its content is preserved in the folder `README.md`, which already carries the same table |
| R6 | An audit reports an error that is not one | Every finding is checked before it is applied; two of roughly twenty-five findings were wrong in the previous pass |

---

## S8. Out of scope

- Chapters 1, 2, 4, 5 and 6.
- Any factual correction to Chapter 3's content.
- The outstanding bibliographic items listed in the folder `README.md` — arXiv preprints without printed volume and page data, the Xu et al. 2016/2017 discrepancy, the Delaunay pagination.
- Authoring new conceptual diagrams.
- Converting the thesis to LaTeX.
