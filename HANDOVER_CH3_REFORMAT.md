# Handover — Chapter 3 reformatting (read this first)

**Purpose.** Lets a fresh session resume the Chapter 3 literature-review reformatting without re-deriving anything. Read end to end before touching a file. This is a *task* handover; the project-level one is `HANDOVER.md` and still applies.

**The job.** Reformat the completed Chapter 3 to the presentation conventions of two exemplar BTU Cottbus MSc theses supplied by the user (`ImprovingMusicTranscription_Thesis_PrashantVaradarajan.pdf`, `main.pdf`). **Format only — no content, argument or citation is drawn from those PDFs.** Every fact, number, quotation, table and subsection number in Chapter 3 is preserved.

**Last updated:** 11 September 2026, after the Chapter 2 rebuild (see §9).

**Earlier:** after Task 17. **THE REFORMATTING IS COMPLETE — all 17 tasks done, all gates pass.** What remains is the user's own review, and committing.

---

## 1. Where things stand

| Task | State |
|---|---|
| 1 — verification gate `tools/ch3_check.py` | ✅ complete (3 fix rounds) |
| 2 — citation map, 27 works / 106 surface forms | ✅ complete (1 fix round) |
| 3 — reference list renumbered `[1]`–`[27]` | ✅ complete |
| 4 — 224 citations substituted across all ten sections | ✅ complete |
| 5 — §3.9 pilot (style calibration) | ✅ complete |
| 6 — three title-free figures | ✅ complete |
| 7 — §3.1 (blockquote, 2 figures, 3 subheads, bridge) | ✅ complete |
| 8A — §3.2 + §3.3 | ✅ complete |
| 8B — §3.4 + §3.5 | ✅ complete |
| 8C — §3.6 + §3.7 | ✅ complete |
| 8D — §3.8 + §3.10 | ✅ complete |
| 16 — chapter preamble | ✅ complete |
| 17 — consistency pass + final gate | ✅ complete |

**All ten sections are reformatted.** Citations numeric, blockquotes gone, bold sub-subheads placed, bridges written. Sub-subhead density: §3.1=3, §3.2=2, §3.3=2, §3.4=3, §3.5=1, §3.6=1, §3.7=2, §3.8=4, §3.9=1, §3.10=2.

**§3.10 was retitled** to `## 3.10 Synthesis and Research Gap` under convention C8 — the number is unchanged and nothing cites it by title. This is the only authorised title change in the whole task, and it is why `ch3_verify.py` reports one expected `HEADING CHANGED OR REMOVED` finding against `HEAD`. **`TOTAL ISSUES: 1` is the correct clean state from here on** — it becomes 0 again once the current work is committed.

**Nothing is committed by the agent pipeline.** All work is in the working tree; the user commits when they choose.

---

## 1b. Final gate result

```
PASS  citations == entries == 27
PASS  residual author-year == 0
PASS  no broken section refs
PASS  no broken table refs
PASS  no broken figure refs
PASS  no uncited entries
PASS  assembled words 29,377 in 29,237-30,917
```

`ch3_verify.py` reports `TOTAL ISSUES: 1` — the authorised §3.10 retitle, measured against `HEAD`. It returns to 0 once the work is committed.

## 2. Run these before doing anything

```bash
cd /Users/addhyanpant/Desktop/Thesis5/FinalMERThesis
python3 tools/ch3_check.py      # gate: citations, cross-refs, word count
python3 tools/ch3_verify.py     # drift vs HEAD + name/number pairing
```

Expected right now: `27` citations = `27` entries, `0` residual, no broken section/table/figure references, and `TOTAL ISSUES: 0`.

If either reports a problem before you have changed anything, stop and investigate — do not proceed.

| Script | Does |
|---|---|
| `tools/ch3_check.py` | Counts distinct `[n]`, checks every `§3.x.y`/`§2.x`/`Table 3.n`/`Figure 3.n` resolves, finds residual author–year strings, reports word count. Read-only. |
| `tools/ch3_verify.py` | Compares each section against `HEAD`: every number, quotation ≥12 chars, heading and table row must survive. Also checks every `Author et al. [n]` pairs with the right reference entry. Takes an optional filename filter and `--ref`. |
| `tools/ch3_citations.py` | `extract` / `apply`. **Already run. Do not run `apply` again.** |

These two scripts replace what were three separate audit agents. They cost nothing and are more reliable — use them instead of asking an agent to check numbers or references.

---

## 3. Documents

| Path | What |
|---|---|
| `docs/superpowers/specs/2026-09-11-chapter3-reformat-design.md` | The design. Section **S2** is the convention sheet C1–C10 — the binding contract. |
| `docs/superpowers/plans/2026-09-11-chapter3-reformat.md` | The plan. **Read the `REVISION` section at the end** — it supersedes the original Tasks 5–17. |
| `docs/superpowers/plans/2026-09-11-chapter3-agent-briefs.md` | Agent brief templates. Briefs 3b and 3c are **retired** (scripts do that work now). |
| `.superpowers/sdd/2026-09-11-chapter3-reformat/progress.md` | The ledger. Every ruling, with its cost-if-wrong. |
| `Thesis_Chapter3_LiteratureReview/Tech-wise/citation_map.tsv` | The citation map. Source of truth for `[n]`. |

---

## 4. The conventions being applied

Per section: **C3** strip the `> **Scope of this section.**` blockquote to ordinary paragraphs (keep every word and bold lead-in) · **C4** promote a bold lead-in to its own line only under the density rule below · **C6** footnotes for parenthetical definitions that interrupt the argument · **C7** one closing bridge sentence pointing at the next section.

Already done centrally, do not repeat: **C1/C1e** citations · **C5** figures (§3.1 only) · **C9** table captions. **C2** (chapter preamble) is Task 16. **C8** (§3.10 as chapter conclusion) is Task 8D.

### The density rule — calibrated on the §3.9 pilot, do not deviate

> Promote a bold lead-in to a standalone line only when **(a)** its subsection exceeds roughly 400 words **and** **(b)** the lead-in describes what follows rather than asserting a claim.

Expect **2–4 promotions in a 6,000-word section**, not ten. **Exception:** a set of *parallel named-entity labels* (method or variant names, e.g. `**Optical Strain Features (OSF).**` / `**Optical Strain Weighted features (OSW).**` / `**OSF + OSW.**`) qualifies regardless of the word threshold, provided every member of the set is promoted together. Those are the closest thing in this chapter to the exemplars' own structural labels. The threshold still governs single lead-ins. Reasoning: the exemplars run ~1 heading per 130–200 words, but their bold subheads exist *because their subsections are unnumbered*. Ours are numbered and already carry that scaffold — §3.x.6 "Limitations" ≡ their "Challenges", §3.x.7 "Implications" ≡ their "Models and Methods". Adding many more overshoots the exemplars rather than matching them.

**Rejected in practice, learn from these:** claim-assertions (`**γ = 2 is inherited rather than tuned.**`), vague labels (`**A second declaration.**`), anything whose subject is ambiguous in context (`**A caution on this figure.**` — "figure" now collides with Figures 3.1/3.2), and any single item pulled out of a First/Second/Third enumeration, which breaks parallelism.

### Bridge sentences

~20–25 words, assembled from material already in the section, no new claim. Example from §3.1: *"The 200 fps capture rate established here is what lets the Eulerian Video Magnification pre-processing stage, reviewed next in §3.2, target a matched temporal band."*

**When briefing a writer, describe the bridge's shape — never paste a real bridge from another section as an "example".** Doing so once nearly caused §3.9's bridge to be duplicated into §3.1.

**Forbid this construction explicitly:** *"X, the …, is reviewed next in §3.n, \<full section title\>."* The batch 8A writer produced it for both its sections — an identical template that also restates a section title the reader can already see. Reference the next section **by number only**, and vary the sentence shape between sections. The four written so far all differ from each other; keep it that way:

| § | Bridge |
|---|---|
| 3.1 | "The 200 fps capture rate established here is what lets the Eulerian Video Magnification pre-processing stage, reviewed next in §3.2, target a matched temporal band." |
| 3.2 | "What magnification is placed in front of therefore matters as much as its own parameters, and §3.3 turns to that motion representation." |
| 3.3 | "The same frame pair that yields this flow field also yields the third channel, optical strain, whose derivation §3.4 takes up." |
| 3.4 | "§3.5 addresses how many frames a normalised sequence should retain, the temporal axis that every strain method reviewed here has already collapsed." |
| 3.5 | "In §3.6, the spatial backbone takes these now-equal-length spatio-temporal volumes as its input, the batched-network requirement this normalisation exists to satisfy." |
| 3.6 | "The same parameter-free principle invoked here to justify shrinking the backbone is the one §3.7's attention module puts into practice." |
| 3.7 | "Having pooled the temporal axis into SimAM's statistics rather than modelling it explicitly, §3.8 considers an architecture that treats time as its own dimension." |
| 3.8 | "Isolating this component across six matched pairs was possible only because §3.9 covers the training regime that stayed fixed beneath every one of them." |
| 3.9 | "These three gaps, and the two this thesis addresses, are drawn together with those from the other ablated components in §3.10." |
| 3.10 | "The class grouping, representation, fold structure and metric this review has fixed are what Chapter 4 now specifies in full as the study's methodology." |

---

## 5. Rulings already made — do not relitigate

1. **No commits, no worktree.** User directive. Revert with `git checkout -- <file>`; diff with `git diff -- <file>`.
2. **Non-citation brackets.** The chapter contains `$[8, 16]$` (α interval), `` `[8]` ``/`` `[12]` `` (Shreve's own markers), `$[0,1]$`, and Python `dim=[2,3]`. `ch3_check.py` masks fenced code, inline code, math and quoted spans. Three were disambiguated in the text during Task 4.
3. **Three possessive citations** — `Yang et al.'s [14]`, `Dosovitskiy et al.'s [27]`, `Shreve et al.'s [2]` — needed explicit map rows. The extraction regex cannot match `'s`, and seven works are dated 2021, so bare `(2021)` resolved to Bai `[15]`. Two citations would have been silently wrong.
4. **Two composites were split** before substitution so their labels survived: `(MEGC 2019) [4]` and `(LBP-TOP) [5]`. Collapsing them to `[4]`/`[5]` would have deleted abbreviations used 19+ times.
5. **Figure 3.3 rejected.** `fig3_3_metric_definitions.png` plots per-configuration results (C2, C8, C12 … C16) — Chapter 5's material. Chapter 3 has **two** figures. No renumbering.
6. **Word-count floor is 29,237, not 29,445.** Numeric citations are shorter than author–year, so Task 4 legitimately removed 208 words. Task 17's band is 29,237–30,917.
7. **`(Xu et al., 2016)`** in the text is a known typo for 2017; mapped to `[18]`. Already recorded in the folder README.
8. **Method-name aliases** (`STSTNet [11]`, `OFF-ApexNet [10]`, `EMR [12]`, `RCN [25]`) map to their underlying papers. Correct and matches the exemplars.

---

## 6. How to run a batch (8A–8D)

Two agents per batch. Batches are **two sections**, deliberately — a writer holding four files drifts.

1. **Writer** (one dispatch, two files). Give it: the file paths, C3/C4/C6/C7, the density rule verbatim, the bridge target for each section, and the hard constraints. It must edit **one file at a time**, finishing and reporting the first before opening the second.
2. **Run the scripts per file**, not in aggregate: `python3 tools/ch3_verify.py 02_3.2` then `03_3.3`. A drifting file must not hide behind a clean one.
3. **Prose-judgment agent** (one dispatch). Give it only the new prose from `git diff` and one question: *does any new sentence introduce a claim, hedge, comparison or evaluation not present in the source text?* It does not check numbers, citations or references — the scripts own those.
4. Any file reporting drift is reverted **alone** (`git checkout -- <file>`) and redone individually. Never re-run a whole batch.

### Batch contents and bridge targets

| Batch | Files | Bridge points to |
|---|---|---|
| ~~8A~~ | ~~§3.2, §3.3~~ — **done** | — |
| ~~8B~~ | ~~§3.4, §3.5~~ — **done** | — |
| ~~8C~~ | ~~§3.6, §3.7~~ — **done** | — |
| ~~8D~~ | ~~§3.8, §3.10~~ — **done** | — |

**§3.10 is special (C8):** it is the chapter's conclusion. Its title shortens toward the exemplars' form, its **number stays `3.10`**, and its bridge points at Chapter 4 (Methodology), not at another section.

### Section-specific cautions

- **§3.2** — the α discrepancy (§3.2.4) and pass-band argument (§3.2.6) turn on specific numbers. Lines 69 and 108 now carry `$[8, 16]$`; line 55's `[8, 16]` is inside a verbatim quotation and must stay bare.
- **§3.4** — §3.4.3 is a declared divergence about three-term vs four-term strain, corrected in an earlier audit. It must not drift back. Line 15 has backticked `` `[8]` ``/`` `[12]` `` inside and outside a quotation — do not normalise them.
- **§3.6** — carries the 3D-CNN's −0.031 and its 97 % compute share, both canonical here.
- **§3.8** — carries the transformer's +0.217, canonical here.
- **§3.10** — carries Table 3.21 and the corrected claim that **four** distinct configurations win across the five protocols, not five.

---

## 6b. Do NOT "fix" the repeated declaration headings

Three promoted headings share a construction — §3.5.7 "A declaration about what this pipeline actually implements.", §3.6.6 "A declaration about what this component actually is.", §3.7.7 "A declaration about the 5-D extension." This is **not** the bridge-template problem. These are the author's own pre-existing bold lead-ins, only promoted to their own line; rewriting them would change the author's words. "Declaration" is also this chapter's term of art for a declared divergence (ten of them, consolidated in Table 3.21 of §3.10.5), so the recurrence signals a real convention. Leave them.

## 7. Tasks 16 and 17

**Task 16 — done.** The preamble heredoc in `rebuild_complete.sh` now holds three unnumbered framing paragraphs. The displaced Sources/Structure/Verification policy text and the `## Contents` table are preserved verbatim in `Tech-wise/README.md` under "Chapter conventions (moved from the chapter preamble, 11 September 2026)". The chapter opens title → framing prose → §3.1.

**Task 17 — consistency pass and final gate.** One editor agent varies the ten bridge sentences (ten identical constructions read mechanically), checks sub-subhead density is comparable across sections, and makes footnote numbering sequential. Then run both scripts, `rebuild_complete.sh`, and confirm the assembled chapter is 29,237–30,917 words.

---

## 8. Known-good state and gotchas

- `rebuild_complete.sh` regenerates `00_Chapter3_Complete.md`. **Never hand-edit that file.** Its reference counter was fixed in Task 3 for the `[n]` prefix.
- Figures live in `report_figures_thesis/`, generated with `CH3_THESIS_FIGS=1 python3 tools/loso_report_figures.py`. That script needs `tools/data.json`, which is **not in the checkout** and must be regenerated from tracked sources first.
- Section files are referenced from `Thesis_Chapter3_LiteratureReview/Tech-wise/`, so figure paths are `../../report_figures_thesis/…`.
- Subagents on this plan have twice returned "Concerns: None" while shipping a real defect. **Verify every claim against the file.** Roughly two in twenty-five reported findings have also been wrong in the other direction — check those too before acting.
- One agent staged files with `git add` despite explicit instruction. Check `git status` before committing.

---

## 9. Chapter 2 rebuild — COMPLETE

A second, separate task run after the Chapter 3 reformatting. Chapter 2 was restructured into **pipeline order** with two new sections and a conclusion, because it previously walked the reader through the pipeline backwards and jumped into `Conv3d` with no foundation.

**Final structure — 10,349 words, 9 sections:**

| § | Section | State |
|---|---|---|
| 2.1 | The phenomenon | subsections numbered 2.1.1–2.1.4, bridge added |
| 2.2 | From recording to pipeline input | **new**, 724 words |
| 2.3 | Motion magnification | moved before motion; bridge added |
| 2.4 | From video to motion | was §2.2; bridge added |
| 2.5 | Neural network fundamentals | **new**, 1,731 words |
| 2.6 | Network building blocks | was §2.4; bridge added |
| 2.7 | Learning under scarcity and skew | was §2.5; bridge added |
| 2.8 | Evaluating on a small corpus | was §2.6; no bridge — §2.9 follows naturally |
| 2.9 | Conclusion | **new**, bridges to Chapter 3 |

**41 `§2.x` cross-references were remapped** across both chapters (2.2→2.4, 2.4→2.6, 2.5→2.7, 2.6→2.8). All resolve.

**Things a successor should know:**

- **§2.2's most important content is a negative.** This project performs no face detection, landmarking, registration or alignment. The 68-point ASM + Local Weighted Mean work Chapter 3 §3.1.2 describes is the *corpus authors'*. A reader who has just read §3.1.2 will otherwise assume the pipeline does it too.
- **§2.5 cites nothing, deliberately.** Chapter 2's convention is that standard ML material with no `docs/` source is stated uncited rather than with an invented citation.
- **Section files must NOT end with a `---` divider.** Both rebuild scripts append one. Sixteen files carried their own, producing doubled dividers in the assembled output; that has been fixed and should not be reintroduced.
- **Risk R4 is RESOLVED.** Chapter 2 was converted to numeric `[n]` citations, reference list renumbered `[1]`–`[11]` by first appearance. `(Lin et al., 2017)` correctly remains author–year: focal loss is out-of-corpus and has no reference entry. Chapter 2's `rebuild_complete.sh` reference counter was fixed at the same time — it had the same author–year regex bug Chapter 3's had.
- **A new question this raises.** Chapter 2 is now `[1]`–`[11]` and Chapter 3 is `[1]`–`[27]`, numbered independently because the thesis keeps one reference list per chapter. So `[1]` means Yan et al. in Chapter 2 and Yan et al. in Chapter 3 — but `[3]` means Li, Y. in one and Li, X. in the other. Both exemplar theses avoid this with a single thesis-wide Bibliography. Decide deliberately.
- **Also open:** both exemplar theses use a single thesis-wide Bibliography; this thesis keeps one reference list per chapter. A deliberate divergence, not an error, but decide it consciously.

---

## 10. Chapter 4 (Methodology) — COMPLETE

Written from a code-verified scope check, not from the handover's summary figures. **9,052 words, 8 sections.**

| § | Section | Words |
|---|---|--:|
| 4.1 | Research approach and experimental design | 1,187 |
| 4.2 | The corpus and its preparation | 986 |
| 4.3 | The ablation matrix | 909 |
| 4.4 | Model architecture | 1,455 |
| 4.5 | Training procedure | 1,337 |
| 4.6 | Evaluation protocol | 1,520 |
| 4.7 | Computational environment | 874 |
| 4.8 | Conclusion | 554 |

**Facts the scope check corrected or established — do not revert these:**

- **Batch size is 4, not 2.** `ablation_config.py` defaults to 2; `tools/run_ablation_gpu.py` overrides to `--batch_size 4`, and 4 produced every stored result.
- **The code builds 16 configurations, not 12.** `itertools.product` makes all sixteen; `is_valid()` rejects SimAM-without-CNN at run time, leaving twelve. The "8-cell" and "12-cell" docstrings are both stale.
- **The 48.9/50.6 GPU-hour figure is an extrapolation, not a measurement** — stored single-fold time × 25, and only the *last* fold's timing survives the fold loop. §4.7 says so explicitly.
- **Two model implementations exist.** `Ablation_Study/models.py::AblationMERModel` produced every result; `Stage2_Architecture/models/hybrid_model.py` is an earlier fixed prototype with no toggles and no dropout in its head. Chapter 4 describes the former.
- Confirmed exactly as the handover claimed: inactive α term, no inner validation split, single seed 42, pooled macro F1 never stored by the code.

**Not recorded anywhere, and deliberately not invented:** GPU model, CPU, RAM, and all library versions. §4.7.1 states this as a gap in the record.

**Still to write:** Chapters 1, 5 and 6.
