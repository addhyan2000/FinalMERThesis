# Thesis writing — handover

**Supersedes `HANDOVER_CH3_REFORMAT.md`**, which covered only the Chapter 3 reformatting and is kept for its detailed record of that task. `HANDOVER.md` remains the project-level document about the experiment itself; this one covers the *writing*.

**Last updated:** 13 September 2026. The thesis body is complete at six chapters. The most recent session verified **all 246 citations against the source PDFs**, thinned Chapter 3's results reporting, and made ~60 corrections; see §8.

---

## 1. State

| Chapter | Words | State |
|---|--:|---|
| 1 — Introduction | 826 | ✅ complete, written last against Ch2–6; states no result |
| 2 — Background | 9,593 | ✅ complete, pipeline order; citations verified |
| 3 — Literature Review | 28,471 | ✅ complete; results reporting removed (§8), citations verified |
| 4 — Methodology | 9,739 | ✅ complete; §4.6.7 added; preprocessing values moved here from Ch2 |
| 5 — Results | 7,911 | ✅ complete, 9 figures; audited and corrected |
| 6 — Conclusion | 2,018 | ✅ complete, 4 sections, zero citations |
| **Body total** | **58,558** | |

Single thesis-wide bibliography: 27 entries, all cited, no orphans.

## 1a. How this project uses subagents

`SUBAGENT_WORKFLOW.md` is the method and it works. Three stages: **scope check** (report-only, reads the code or the sources and reports facts) → **writer** (1–2 files, one at a time) → **verify** (scripts first; an agent only for the judgment half). Two files per writer, at most three agents dispatched at once, and never two writers on the same file.

Three things that repeatedly earned their cost, and should stay in every brief:

> Before writing "no concerns", run a check that would disprove it. Name that check in your report.

> If any number in this brief is contradicted by the source data, do not write it — report the discrepancy instead.

> This chapter is already fact-audited. Any factual change is a defect, **including one that looks like a correction**. If you believe a fact is wrong, report it; do not fix it.

**Briefs are not a trusted source.** Across this project's sessions, agents have corrected their own briefs roughly a dozen times — including finding a promise the controller's inventory had missed, a distinct-value count that was off by one, and a premise about what `docs/` supports that was simply wrong. That is the instruction working, not the agents being difficult.

**The controller does not delegate**: deciding what to build, ruling on a conflict, applying fixes after a review, and reading the finished prose.

**On rate limits.** Opus has a weekly cap. When it is exhausted, Sonnet has separate capacity and does this work well — the §3.8/§3.9 citation audit and the whole-thesis coherence read were both done on Sonnet and both found real defects. A dead agent is not necessarily lost work: check the filesystem before re-dispatching.

## 2. Run these before doing anything

```bash
cd /Users/addhyanpant/Desktop/Thesis5/FinalMERThesis
python3 tools/bibliography.py check     # 27/27, no ghosts, no orphans
python3 tools/ch3_check.py              # structural: cross-refs, tables, figures
```

| Tool | Does |
|---|---|
| `tools/bibliography.py` | `check` and `render`. Resolves `[@key]` to numbers by first appearance across the thesis. `THESIS_ORDER` is the one place a new chapter is registered, and its order **is** the numbering. |
| `tools/ch3_check.py` | Structural gate: every `§x.y`, `Table n.m`, `Figure n.m` resolves. Its citation counters are obsolete — that job moved to `bibliography.py`. |
| `tools/ch3_verify.py` | Drift check against a git ref: every number, quotation, heading, table row preserved. Built for the reformatting task; useful whenever editing verified text. |
| `tools/loso_report_figures.py` | Set `CH3_THESIS_FIGS=1` for title-free thesis figures into `report_figures_thesis/`. **Needs `tools/data.json`, which is not in the checkout** and must be regenerated first. |
| `tools/thesis_fig_regen.py` | Rebuilds Figures 5.4 and 5.8 from `Ablation_Study/results/` directly, with no `data.json`. Both were defective: 5.8 plotted the placeholder CSV, and 5.4's left panel was an unreported N=39 holdout run contradicting Table 5.1. Needs matplotlib, which is **not installed on this machine**. |

Each chapter has `rebuild_complete.sh`. It concatenates the numbered section files and then calls `bibliography.py render`. **Never hand-edit a `00_Chapter*_Complete.md`** — it is generated.

## 3. Conventions

**Citations.** Write `[@key]` or `[@key1; @key2]` in section files. Never a number, never author–year. Keys live in `BIBLIOGRAPHY.md`; add an entry there before citing it. Keys are lowercase surname + year, with an initial where surnames collide (`liX2018` vs `liY2018`, `zhaoG2007` vs `zhaoS2021`) and the entry's own suffix (`liong2019a`).

**Cite only what is in `docs/`.** Works cited only *inside* those papers are attributed in prose to the reviewing paper and get no entry. Standard ML background with no `docs/` source is stated **uncited** — never with an invented citation.

**Division of labour, enforced.** Chapter 2 explains *what a mechanism is*. Chapter 3 surveys *who did what and what gap remains*. Chapter 4 states *what this study did, with which values*. Chapter 5 reports *what it found*. A specific parameter value in Chapter 2 is a defect; re-explaining a mechanism in Chapter 4 is a defect. This was violated and repaired once — see §6.

**Section files must NOT end with a `---` divider.** The rebuild scripts append one. Sixteen files once carried their own, producing doubled dividers.

**Bold sub-subheads, sparingly.** Promote an existing bold lead-in to its own line only when (a) its subsection exceeds ~400 words **and** (b) the lead-in describes what follows rather than asserting a claim. Exception: a set of *parallel named-entity labels* qualifies regardless of length, if all members are promoted together. Rejected in practice: claim-assertions, vague labels, ambiguous subjects, single items from a First/Second/Third enumeration.

**Bridges.** Each section closes with ~20–25 words pointing at the next, assembled from its own material. They must not share a construction — a writer once produced an identical template for two sections and it had to be rewritten. Never paste a real bridge from another section into a brief as an "example"; describe the shape instead.

**British spelling** throughout: normalisation, generalisation, greyscale.

## 4. Facts that are easy to get wrong

- **Pooled macro F1 = mean of the `per_class_f1` array.** The code never stores it. The stored `macro_f1` key is mean-of-folds and is **rejected** — structurally capped at 0.6267 by fold composition. `micro_f1` is pooled accuracy; the `accuracy` key is mean-of-folds and inflates `config_8` by 6.3 points.
- **Batch size is 4, not 2.** The config defaults to 2; `tools/run_ablation_gpu.py` overrides to 4, and 4 produced every result.
- **The code builds 16 configurations, not 12.** `is_valid()` rejects SimAM-without-CNN at run time. The "8-cell" and "12-cell" docstrings are both stale.
- **The 48.9/50.6 GPU-hour figure is an extrapolation**, not a measurement: stored single-fold time × 25, and only the *last* fold's timing survives the loop.
- **Two model implementations exist.** `Ablation_Study/models.py::AblationMERModel` produced every result. `Stage2_Architecture/models/hybrid_model.py` is an earlier fixed prototype with no toggles.
- **26 vs 25 subjects.** CASME II has 26 valid participants; 25 have usable three-class clips. LOSO runs 25 folds.
- **`config_4` is the baseline; `config_1` is the EVM-off control.** Measuring against `config_1` instead gives a different count of configurations "below baseline" — this is what produced the old five-vs-four error. Against `config_4`, exactly four fall below: `config_1`, `config_3`, `config_5`, `config_16`. **`config_13` (0.4480) clears it.**
- **`ALL_RESULTS_LOSO.md`, `TECHNOLOGY_LADDER_RESULTS.md` and `THESIS_PRESENTATION*.md` still contain the uncorrected "five" language** in two places. Take no prose from them.
- **`Ablation_Study/literature_baselines.csv` is placeholder data.** Its rows are an invented "Example Transformer MER", a non-existent "Vivian et al. 2019", a misattributed STSTNet ("Li et al. 2018", 0.63 against its true 0.8382 UF1), and a client target. Its `accuracy` column is not UF1. `compare_with_literature.py:56` also stamps `validation: "holdout"` on this project's LOSO rows. **Cite nothing from it, and nothing from `results/literature_comparison.csv`'s literature rows.** See `Ablation_Study/literature_baselines.README.md`. The verified figures are MEGC 2019 Table IV, reproduced as Table 3.6 and Table 5.8.
- **`LOSO_Validation_Report.md` builds on that placeholder data** — its "Every configuration beats the published literature" (line 32) rests on the invented 0.65. Take no literature comparison from it.
- **A complete 16-cell matrix gives every flag eight matched pairs.** EVM and the transformer realise six; SimAM and the CNN stem realise four. "Four rather than six" is wrong.
- **SimAM is free in parameters only.** It costs +11% training time and +5,292 MB peak VRAM (+35.9%) per matched pair — about 2.64 of the sweep's 50.61 extrapolated GPU-hours, 5.2%. Any claim that it "costs nothing" is false.
- **Lowest Surprise-class F1 is `config_4`'s 0.2462**, not `config_9`'s 0.3000. `config_2` holds the *highest* Positive and Surprise F1 in the study, and the widest spread belongs to `config_9` (0.5491).
- **Chapter 6 cites, it never restates.** Only `0.7122` and `0.6659` appear as figures, once each, in §6.1. Every other number is referenced by section. Chapter 6 carries **zero citations**, matching both exemplar theses.
- **Chapters 3 and 5 make 23 forward commitments to Chapter 6.** §6.3 discharges them. Adding a commitment elsewhere without adding it to §6.3 leaves a promise the thesis does not keep — grep Chapters 2–5 for `Chapter 6`, `further work`, `next ablation` before assuming the list is closed.
- **The five-protocol / four-winner claim is withdrawn.** It appeared three times in Chapter 3 and was never supported: the five runs differ in N (24–156), epochs (5/50/60) and class coverage, and the two earliest carry the EVM routing defect. §4.6.7 now records the history honestly. Do not reinstate it.
- **`tools/ch3_check.py` now gates every chapter's cross-references** (`broken refs, all chapters`). It validates that a `§x.y` target *exists*, not that it says what the citing sentence claims — that still needs a reader.
- **Chapter 1 states no result.** No figure, ranking or configuration name appears in it, deliberately: the two headline numbers appear exactly once each, in §6.1. §1.1 closes on a qualitative clause only.
- **Chapter 1 is registered FIRST in `THESIS_ORDER`** (`tools/bibliography.py`), because citation numbers are assigned by first appearance. It cites `yan2014` and `xia2020a`, both of which already appeared early, so no renumbering occurred — but a *new* key cited in Chapter 1 would renumber the whole thesis and every chapter would need rebuilding.
- **Applications claims are the source papers' motivations, not demonstrated use.** `yan2014` and `xia2020a` support national-safety / interrogation / clinical / lie-detection framings. Write "is motivated by" or "reported as promising for" — never "is used for". The four most quotable application papers in `docs/` are deliberately NOT in the bibliography; citing one is a defect.
- **Four binary flags give sixteen combinations, twelve of them valid.** Any phrasing that presents twelve as the direct product of independent toggling is wrong (§4.1.3 is titled "Twelve configurations, not sixteen").

## 5. The headline result

`config_2_temporal_only` — a transformer alone, no CNN, no SimAM, no magnification — scores **0.7122** pooled macro F1. The proposed `config_8_proposed_unified` scores **0.6659**. `config_8` leads on *accuracy* (0.7500 vs 0.7436) by one clip. **The proposed system is not the best configuration**, and Chapter 5 says so in §5.1, §5.5 and §5.8.

Component effects: transformer **+0.2173** (6/6 pairs positive), magnification **+0.0152** (4/6), attention **+0.0034** (3/4), convolutional stem **−0.0310** (2/4) while consuming **96.6%** of the sweep's compute.

## 6. Structural repairs already made — do not undo

- **Chapter 2 was reordered into pipeline order.** Magnification (§2.3) now precedes motion representation (§2.4); it used to follow it, walking the reader through the pipeline backwards. Two sections were added: §2.2 (what the pipeline receives) and §2.5 (neural-network fundamentals, which the chapter entirely lacked before §2.6 opened on `Conv3d`).
- **Chapter 2 had drifted into stating Chapter 4's decisions.** §2.7.2, §2.7.4, §2.7.7 and §2.8.5 gave specific hyper-parameter values and this study's own outcomes, because they were written before Chapter 4 existed. They now give mechanism only and defer. §2.8.5 was rewritten wholesale.
- **Chapter 3 was reformatted** to the exemplars: numeric citations, unnumbered framing prose, bold sub-subheads, two figures, bridges, and §3.10 retitled as the chapter conclusion (number unchanged).
- **Chapter 3 no longer reports Chapter 5's results, and Table 3.20 is gone** (§8). Do not reinstate either: the thesis now tells the reader in §1.2 that Chapter 5 reports the findings, and Chapter 3 keeps that promise.
- **The preprocessing parameters live in §4.2.5, not Chapter 2** (§8). Chapter 2 explains the mechanism and defers the values, as §2.7 and §2.8 already did.
- **Chapter 6 cites nothing and states no result beyond 0.7122 / 0.6659, once each in §6.1.** Both exemplar theses have citation-free conclusions. Chapter 1 states no result at all.
- **Chapter 1 is registered FIRST in `THESIS_ORDER`** (`tools/bibliography.py`), because citation numbers are assigned by first appearance. Its two citations sit in the chapter frame inside `rebuild_complete.sh`, not in a section file — so a key cited *only* in a frame would be invisible to `bibliography.py check`'s orphan test. Worth knowing before writing another frame.
- **§2.2's most valuable content is a negative** — this project performs no face detection, landmarking or registration. That is the corpus authors' work, described in §3.1.2. A reader who has just read §3.1.2 will otherwise assume otherwise.

## 7. Remaining work

**The thesis body is finished.** What is left is not writing:

- **Front matter has never been discussed**: abstract, acknowledgements, declaration, list of figures, list of tables, table of contents. Both exemplar theses carry all of these.
- **`liong2019a` author order — needs your decision.** The bibliography lists Liong first, matching the arXiv preprint in `docs/`, but carries the *journal* venue (*Signal Processing: Image Communication*, 74, 129–139). MEGC 2019 (ref [26]) and Zhao et al. (ref [12]) both cite the published version with **Gan first**. If the journal record is authoritative, nine in-text "Liong et al." references in §3.3 become "Gan et al.". Check the actual SPIC record before changing anything.
- **Other bibliographic details still unconfirmed**: venues and pages for entries whose `docs/` copies are preprints or author manuscripts; the Xu 2016/2017 discrepancy; the Delaunay paper's pagination (the `docs/` filename says 703–716, the running heads say 700–711, the bibliography says 698–711 — the bibliography is right, the filename is wrong).
- **Deferred minors from the citation audit**, none of which changes a claim: unmarked elisions inside a few quotations in §3.3, §3.5 and §3.7; Table 3.11's caption does not record that Liong et al.'s Table 8 mixes 5×5 and 8×8 block settings across columns; §3.3.7 and §3.3.8 say "full factorial" where twelve of sixteen cells ran; `Table 3.5`'s CAS(ME)² ethnicity cell is an inference, not a figure qu2016 states.
- **`docs/` housekeeping**: two copies of the Xia et al. STRCN paper (same paper, different file sizes); `liong2018` also appears twice (preprint and typeset).

---

## 8. The citation audit of 13 September 2026 — what it found, and why it matters

Every one of the thesis's **246 citation instances** was opened against its source PDF. Nothing was fabricated and no key pointed at the wrong paper; every reproduced table matched cell for cell. But four Critical defects surfaced, and **three were the same shape**:

> **A claim that "no study does X", where the cited paper does X in a named subsection.**

- §3.5 said CAS(ME)²'s baseline "combines TIM with LBP-TOP". Qu et al. normalise to **120 frames by linear interpolation** under LOVO; TIM appears in that paper only in related work.
- §3.5 said Ben et al. ran "the only controlled experiment" on interpolation length. Li et al. §5.2.1 is titled *"Effect of the interpolation length"*, sweeps 10–80 frames on three datasets under LOSO, and concludes the opposite (TIM10 best).
- §3.9 said, in bold, "No ablation of focal loss against cross-entropy exists in the corpus." Zhao et al. §5.2 is titled *"The effect of focal loss"* and is exactly that ablation — cited two sentences before and after the claim.

**If you add any gap claim to this thesis, search the cited paper for the thing you say is missing before writing it.** A gap is the one assertion that cannot be checked by reading the sentence.

The fourth Critical: **the √2 strain divergence had no precedent.** §3.4.3 claimed STSTNet uses the three-term form. STSTNet's Eq. (6) is typographically broken — its shear term repeats ∂u, and its ½ sits *outside* the square. Repaired as the surrounding text implies, it is the **four**-term form. The divergence is real and correctly derived; only its claimed precedent was wrong.

### Errors that recurred in more than one place

Each of these was fixed once, then found again elsewhere. **When you fix one of these, sweep the whole thesis for the class, not the instance.**

| Error | Instances found |
|---|--:|
| "five successive protocols → four winning configurations" asserted as a result | **5** (§3.10 ×2, §3.1 ×2, one more in §3.10's bullet list) |
| EVM switch described as inert in "every earlier run" | **4** (it was inert only in the N=39 holdouts and the 5-fold pilot; runs C and D were clean) |
| Stem compute share written as "97 %" | **4** (the figure is 96.6 %) |
| Chapter 3 reporting Chapter 5's results | **9 passages + Table 3.20** |

### Structural changes made

- **Chapter 3 no longer reports Chapter 5's results.** Numeric findings were removed from §3.2, §3.3, §3.4, §3.6, §3.7 and §3.8, and **Table 3.20 was deleted** (it duplicated Table 5.9). Each passage now states what the ablation will measure and defers the number. Tables renumbered contiguously to 3.1–3.20. Only the literature's own figures remain.
- **Preprocessing parameters moved to §4.2.5.** α = 10, the 5–25 Hz band, four pyramid levels and the Farnebäck settings had lived only in Chapter 2, while Table 3.7 assigned them to Chapter 4 and §4.2.5 deferred to Chapter 2 — so the Methodology did not contain the values needed to reproduce the study.
- **Declared divergences added**: the 5–25 Hz band is the one Bai et al. *rejected* as noisy (they narrowed to 15–25); MEGC's three-class scheme is two per-corpus footnotes, not one rule, and this thesis follows MEGC's **SAMM** Negative mapping rather than its CASME II one.

### Verification discipline that paid off

- **Check a finding before acting on it.** One reported Critical — wrong epochs and wrong winner in §4.6.7's Table 4.4 — was a **false positive**: two different N=52 holdout runs exist, one in the working tree at 5 epochs and one on branch `new_gui_loso_holdout` at 50. The table describes the branch run and is correct. Roughly two findings in twenty-five are wrong in either direction.
- **A grep is not a check.** Two false negatives this session came from malformed greps: one required a sentence-ending period and so missed a table row, causing a correct cross-reference to be "fixed" into a worse one. Confirm absence by reading.
- **Render the page when the text layer looks odd.** The STSTNet equation defect was only settled by rendering the PDF page as an image.
- **`tools/ch3_check.py` gates section-reference *existence*, not content.** A reference can resolve to a real heading and still misdescribe it — that class needs a reader. The tool now covers all six chapters (it originally scanned only Ch2–Ch3; three separate agents found three separate gaps in one fix).
