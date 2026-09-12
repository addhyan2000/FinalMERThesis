# Thesis writing — handover

**Supersedes `HANDOVER_CH3_REFORMAT.md`**, which covered only the Chapter 3 reformatting and is kept for its detailed record of that task. `HANDOVER.md` remains the project-level document about the experiment itself; this one covers the *writing*.

**Last updated:** 12 September 2026, after Chapter 5's three-way audit (numbers, citations, duplication), its corrections, and the Figure 5.4 / 5.8 regeneration.

---

## 1. State

| Chapter | Words | State |
|---|--:|---|
| 1 — Introduction | — | ❌ not started |
| 2 — Background | 9,783 | ✅ complete, restructured into pipeline order |
| 3 — Literature Review | 29,367 | ✅ complete, reformatted to the exemplar conventions |
| 4 — Methodology | 8,949 | ✅ complete, written from a code-verified scope check |
| 5 — Results | 7,911 | ✅ complete, 9 figures; audited and corrected |
| 6 — Conclusion | — | ❌ not started |
| **Body total** | **56,010** | |

Single thesis-wide bibliography: 27 entries, all cited, no orphans.

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

## 5. The headline result

`config_2_temporal_only` — a transformer alone, no CNN, no SimAM, no magnification — scores **0.7122** pooled macro F1. The proposed `config_8_proposed_unified` scores **0.6659**. `config_8` leads on *accuracy* (0.7500 vs 0.7436) by one clip. **The proposed system is not the best configuration**, and Chapter 5 says so in §5.1, §5.5 and §5.8.

Component effects: transformer **+0.2173** (6/6 pairs positive), magnification **+0.0152** (4/6), attention **+0.0034** (3/4), convolutional stem **−0.0310** (2/4) while consuming **96.6%** of the sweep's compute.

## 6. Structural repairs already made — do not undo

- **Chapter 2 was reordered into pipeline order.** Magnification (§2.3) now precedes motion representation (§2.4); it used to follow it, walking the reader through the pipeline backwards. Two sections were added: §2.2 (what the pipeline receives) and §2.5 (neural-network fundamentals, which the chapter entirely lacked before §2.6 opened on `Conv3d`).
- **Chapter 2 had drifted into stating Chapter 4's decisions.** §2.7.2, §2.7.4, §2.7.7 and §2.8.5 gave specific hyper-parameter values and this study's own outcomes, because they were written before Chapter 4 existed. They now give mechanism only and defer. §2.8.5 was rewritten wholesale.
- **Chapter 3 was reformatted** to the exemplars: numeric citations, unnumbered framing prose, bold sub-subheads, two figures, bridges, and §3.10 retitled as the chapter conclusion (number unchanged).
- **§2.2's most valuable content is a negative** — this project performs no face detection, landmarking or registration. That is the corpus authors' work, described in §3.1.2. A reader who has just read §3.1.2 will otherwise assume otherwise.

## 7. Remaining work

**Chapter 6 (Conclusion)** is well prepared. §5.8 already gathers the five limitations; `HANDOVER.md` §8 lists the follow-up experiments in priority order — save per-clip predictions to enable McNemar, rerun the CNN arm at 28×28 or 56×56, magnify before subsampling, multi-seed runs.

**Chapter 1 (Introduction)** is conventionally written last, and is now cheap: Chapters 2–5 fix the scope, the gap and the findings.

**Outstanding, needing manual confirmation** (from the Chapter 3 README): publication venues and page numbers for references whose `docs/` copies are preprints or author manuscripts; the Xu et al. 2016/2017 discrepancy; the Delaunay paper's pagination.

**Not recorded anywhere, and deliberately not invented:** GPU model, CPU, RAM, library versions. §4.7.1 states this as a gap in the record.
