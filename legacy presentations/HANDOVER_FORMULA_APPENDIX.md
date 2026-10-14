# Handover — reference audit complete; next task is the formula appendix

**Written 16 September 2026.** Supersedes nothing; sits alongside `HANDOVER_CH3_SHORTENING.md` (the Chapter 3 method) and the four `contents/CH3_SHORT_PARTS*_README.md` batch records.

Read §1 for where the thesis stands, §2 for the task, §3 for the formula inventory you will not have to rediscover, and §4 for the traps.

---

## 1. State at the end of this session

### 1.1 The six chapter files that matter

Use these. The unshortened originals still exist and share every label — **never include both**.

| Chapter | File to use | Tables | Figures | Cite keys |
|---|---|--:|--:|--:|
| 1 Introduction | `contents/introduction.tex` | 0 | 0 | 2 |
| 2 Background | `contents/chapter2_shortened.tex` | 2 | 0 | 11 |
| 3 Literature Review | `contents/chapter3_shortened.tex` | 17 | 2 | 27 |
| 4 Methodology | `contents/chapter4_shortened.tex` | 5 | 0 | 4 |
| 5 Results | `contents/chapter5_shortened.tex` | 10 | 9 | 6 |
| 6 Conclusion | `contents/chapter6.tex` | 0 | 0 | 0 |

Chapters 3 and 4 are single assembled files. Their editing units (`chapter3_short_part3..10.tex`, `chapter4_short_half1/2.tex`) are retained but the assembled file is what compiles. If you edit a half, re-run its assembler; if you edit the assembled file, retire the halves.

### 1.2 What this session did

A full reference audit of all six files, done with scripts rather than subagents. Results:

- **292 labels, zero duplicates.**
- **250 `\autoref`/`\ref` targets, zero unresolved.**
- **273 citation instances, all 27 keys present in `bib/thesis.bib`, zero orphan entries.**
- **45 floats, every one carrying both a `\label` and a short `\caption[...]`.**
- **10 hard-coded `N.M.K` section numbers** (the divergence-ledger first column), all resolving.
- No doubled float words (`Table~\autoref{...}`), no hard-coded `Table 3.4`-style numbers in prose.

**One real defect found and fixed.** Chapter 3 referenced its two figures with bare filenames; Chapter 5 prefixed all nine with `figures/`. Those conventions cannot both resolve under one `\graphicspath` — whichever you set, one chapter's figures go missing silently. Chapter 5's nine were normalised to bare names. All eleven images live in `report_figures_thesis/`.

**You must have this in the master preamble:**

```tex
\usepackage{graphicx}
\graphicspath{{figures/}{report_figures_thesis/}}
```

Both entries, so the same source compiles in Overleaf (images in `figures/`) and in this checkout (`report_figures_thesis/`).

### 1.3 Front matter written this session

- `contents/list_of_tables.tex` — emits `\listoftables`, plus the audited inventory of all 34 tables.
- `contents/list_of_figures.tex` — emits `\listoffigures`, all 11 figures with their image files, and the `\graphicspath` instructions.
- `contents/list_of_acronyms.tex` — 80 entries in seven groups, extracted from the actual chapter text.

Include all three. **Do not** also include the four superseded per-batch files `contents/chapter3_short_parts{34,56,78,910}_acronyms.tex`.

### 1.4 Reusable scripts (session scratchpad — copy them somewhere permanent if you want them)

`refcheck.py` (labels/refs/cites/floats/images), `numcheck.py` (hard-coded section numbers), `inventory.py` (float inventory with assigned numbers), `ledgercheck.py` (divergence-ledger row numbers). They were written to the session scratchpad, which is temporary. **Re-create them from the descriptions above if they are gone** — each is under 120 lines and the logic is simple regex over the six files.

---

## 2. The next task

**Create a new `.tex` file collecting every formula in the thesis, explaining each one, and citing the paper it comes from.** Work chapter by chapter, **Chapter 3 first, then Chapter 4**, using the shortened files. Then continue to Chapter 2, which is where most of the mathematics actually lives.

Suggested file: `contents/appendix_formulas.tex`, as an appendix rather than a chapter, so it does not disturb the chapter numbering that four files depend on (see §4.1).

### 2.1 What each entry should carry

For every formula:

1. **The formula itself**, reproduced exactly as it appears in the chapter — same symbols, same subscripts.
2. **Where it appears**, by `\autoref` to the owning subsection label (the labels are listed in §3), so the appendix tracks renumbering automatically.
3. **What each symbol means**, in prose.
4. **What the formula does** — the mechanism, not a restatement of the algebra.
5. **Its source**, cited with the existing bib key — *subject to the attribution rule in §4.2, which is the single most important constraint in this task.*
6. **Whether this thesis implements it as written**, and if not, what it does instead. Several formulas are declared divergences; §3.3 lists them.

### 2.2 Ordering

Sequential by chapter, then by order of appearance within the chapter. Chapters 5, 6 and the introduction contain **no display mathematics at all** — verified — so the appendix covers Chapters 2, 3 and 4 only.

---

## 3. The formula inventory

**25 display-math blocks total.** Verified by regex over the three files. Inline math (`$...$`) also carries a few quantities worth including — see §3.4.

### 3.1 Chapter 3 — 7 blocks (do this chapter first)

| # | Subsection label | Formula | Source |
|--:|---|---|---|
| 1 | `sec:megc-metrics` | per-class `F1_c` and `UF1 = (1/C) Σ F1_c` | `see2019` — the MEGC 2019 definition |
| 2 | `sec:strain-tensor` | `ε_F = √(ε_xx² + ε_yy² + ε_xy² + ε_yx²)` — the four-term Frobenius norm | `liong2016`, `liong2014a`; see §4.3 |
| 3 | `sec:strain-divergence` | `ε_os = √(ε_xx² + ε_yy² + ε_xy²)` — the three-term form this thesis implements | **declared divergence** — no source |
| 4 | `sec:strain-divergence` | `½(∂u/∂y + ∂v/∂x)² = 2ε_xy²` — the shear-weighting identity | derivation, no source |
| 5 | `sec:simam-energy` | `d_i`, `s²`, `E_i⁻¹` — the reciprocal energy | `yang2021` |
| 6 | `sec:simam-energy` | `x̃_i = x_i · sigmoid(E_i⁻¹)`, `μ = (1/M) Σ x_i` | `yang2021` |
| 7 | `sec:imbalance-focal` | `FL(p_t) = −α_t (1−p_t)^γ log p_t` | `zhaoS2021`; see §4.2 |

### 3.2 Chapter 4 — 2 blocks

| # | Subsection label | Formula | Source |
|--:|---|---|---|
| 8 | `sec:transformer-config` | sinusoidal positional encoding, `PE(p,2i)` and `PE(p,2i+1)` | Vaswani et al., **not in the bib** — see §4.2 |
| 9 | `sec:mean-of-folds-rejected` | the 0.627 mean-of-folds ceiling, `(10·⅓ + 8·⅔ + 7·1)/25` | this thesis's own arithmetic, no source |

### 3.3 Chapter 2 — 16 blocks (the bulk of the mathematics)

| # | Subsection label | Formula | Source |
|--:|---|---|---|
| 10 | `sec:eulerian-idea` | `I(x,t) = f(x+δ(t)) ≈ f(x) + δ(t)·∂f/∂x` | `bai2021` (reporting Wu et al.) |
| 11 | `sec:eulerian-idea` | `Ĩ(x,t) ≈ f(x) + (1+α)δ(t)·∂f/∂x` | `bai2021` |
| 12 | `sec:optical-flow` | brightness constancy, `I(x,y,t) = I(x+uΔt, y+vΔt, t+Δt)` | `liong2019a` |
| 13 | `sec:optical-flow` | the flow constraint, `I_x u + I_y v + I_t = 0` | `liong2019a` |
| 14 | `sec:optical-strain` | strain tensor `ε = ½(∇u + (∇u)ᵀ)`, with `ε_xx`, `ε_yy` | `shreve2011` |
| 15 | `sec:optical-strain` | shear terms `ε_xy = ε_yx = ½(∂u/∂y + ∂v/∂x)` | `shreve2011` |
| 16 | `sec:optical-strain` | `ε_mag = √(ε_xx² + ε_yy² + ε_xy²)` — the implemented three-term scalar | **declared divergence**, cross-ref Ch3 #3 |
| 17 | `sec:convolution` | `N_params = C_out(C_in·k_H·k_W + 1)` | standard, no source |
| 18 | `sec:normalisation` | batch normalisation, `x̂` and `y = γ_c x̂ + β_c` | Ioffe & Szegedy, **not in bib** |
| 19 | `sec:simam` | `μ`, `v = (1/(M−1)) Σ (x_i−μ)²` | `yang2021`; see §4.4 |
| 20 | `sec:simam` | `E_i⁻¹` and `x̃_i = x_i·sigmoid(E_i⁻¹)` | `yang2021` |
| 21 | `sec:self-attention` | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` | Vaswani, **not in bib** — attribute via `dosovitskiy2021` |
| 22 | `sec:focal-loss` | `FL(p_t) = −α_t(1−p_t)^γ log p_t` | `zhaoS2021` (reporting Lin et al.) |
| 23 | `sec:label-smoothing` | `L_smooth = −(1−ε)log p_t − (ε/C) Σ log p_c` | named by `dosovitskiy2021`; see §4.2 |
| 24 | `sec:confusion-matrix` | `P_c`, `R_c`, `F1_c` | standard, no source |
| 25 | `sec:macro-micro` | `F1_macro = (1/C) Σ F1_c` | standard; `see2019` for the pooled variant |

**Note the duplication.** Formulas 5–7 (Ch3) restate 19, 20 and 22 (Ch2); 2–3 (Ch3) restate 14–16 (Ch2). That is by design — Chapter 2 gives the mechanism, Chapter 3 gives the literature's version and the divergence. The appendix should present each **once**, noting both locations, rather than duplicating.

### 3.4 Inline quantities worth including

Not display math, but formula-like and load-bearing:

- `6600/N` — effective frame rate after sampling 33 frames from an `N`-frame clip at 200 fps (Ch3 `sec:tim-implications`).
- `N/(200 × 32)` — the true inter-frame interval, which is why the realised EVM pass-band is clip-dependent.
- `1/n_c` — the `WeightedRandomSampler` weight (Ch4 `sec:sampler-config`).
- `√(u² + v²)` — flow magnitude (Ch2 `sec:optical-flow` region).
- Parameter totals: 14,544 / 348,736 / 4,704 / 483 and the four sums (Ch4 `sec:parameter-counts`).

---

## 4. Traps

### 4.1 Do not disturb the numbering

Four files reference sections **by number**, not by label. `contents/chapter2_shortened.tex` no longer does (the shortened rewrite removed them), but `chapter3_shortened.tex`'s divergence ledger cites ten: 3.1.2, 3.1.3, 3.2.6, 3.4.3, 3.4.7, 3.5.7, 3.6.6, 3.7.7, 3.8.6, 3.9.7.

Adding the formula file **as an appendix** (`\appendix` before `\include`) avoids this entirely. Adding it as a numbered chapter between existing ones would not. Re-run `numcheck.py` afterwards either way.

### 4.2 The bibliography deliberately omits the original sources — do not "fix" this

`bib/thesis.bib`'s header states the policy explicitly:

> Works cited only *inside* those papers (Ekman, FACS, Wu et al. on EVM, Farneback, TV-L1, Pfister TIM, Lin et al. focal loss, SE/CBAM/ECA, Vaswani, VGG/ResNet/DenseNet) are attributed in the text to the reviewed paper that reports them and deliberately carry no entry here.

So for the formula appendix:

- **Focal loss** is Lin et al.'s, but cite `zhaoS2021`, which is the paper in `docs/` that brings it to micro-expression recognition. Say "the formulation developed for one-stage object detection, adopted by Zhao et al." — do not add a Lin et al. bib entry.
- **Self-attention** is Vaswani et al.'s; attribute through `dosovitskiy2021`.
- **Eulerian magnification** is Wu et al.'s; attribute through `bai2021`.
- **Batch normalisation** has no corpus paper at all — present it as standard practice with no citation.
- **Adding new bib entries would break the numbering.** `tools/bibliography.py` assigns reference numbers by first appearance across the whole thesis; a new entry renumbers everything after it.

Every one of the 27 keys is currently cited and none is orphaned. Keep it that way.

### 4.3 The strain formula is the thesis's most consequential divergence

Chapter 3 `sec:strain-divergence` establishes that the published Frobenius norm counts shear **twice** (`ε_xy² + ε_yx²`) while this thesis's implementation counts it **once** — weighting shear lower by √2 relative to normal strain. This is row 4 of the twelve-row divergence ledger (`tab:divergence-ledger`, Table 3.17).

Handle it carefully in the appendix. It is not a typo to be silently corrected, and an earlier audit established that STSTNet's own printed expression is typographically defective in two ways — which is why this thesis's three-term choice is a divergence *from the whole corpus* rather than a following of STSTNet. Read `sec:strain-divergence` in full before writing this entry. **Do not change either formula.**

### 4.4 SimAM's variance denominator

The paper's printed shared-variance definition divides by `M`; its Figure 3 reference code divides by `M−1`. The implementation follows the code. Both Chapter 2 (`sec:simam`) and Chapter 3 (`sec:simam-energy`) say so. Preserve that distinction — it was verified by rendering the PDF page as an image, because the text layer mangles it.

Relatedly, `M = THW` here (whole-clip statistics), against the source's `HW` (per-frame). That is ledger row 10.

### 4.5 General rules that held all through this project

- **Any factual change is a defect, including one that looks like a correction.** If a formula looks wrong, check the source PDF in `docs/` before touching it — and prefer reporting it to fixing it.
- **`pdftotext` mangles equations routinely.** For any claim about a formula, render the PDF page as an image before concluding. The Read tool's `pages` parameter has failed in this environment; earlier sessions installed PyMuPDF and rendered page regions to PNG instead.
- Extracted paper text was cached at `<scratchpad>/txt/*.txt` with a key→file map in `KEYMAP.md`. That scratchpad is session-scoped and is probably gone; regenerate with `pdftotext -layout docs/<file>.pdf <out>.txt`. Watch for Windows path-length limits on the five longest filenames.

---

## 5. Verification to run when the appendix is written

```bash
python refcheck.py        # labels, refs, cites, floats, images across all files + the new one
python numcheck.py        # the ten ledger row numbers still resolve
python ledgercheck.py     # divergence ledger, if you kept it
```

Then check by hand:

- Every formula in the appendix matches its chapter source **character for character**. A silently "tidied" subscript is a defect.
- Every `\cite` in the appendix uses an existing key; no new bib entries.
- The appendix does not introduce duplicate labels — prefix its own labels, e.g. `eq:strain-frobenius`.
- All 27 bib keys still cited somewhere; no orphans.

---

## 6. Also still outstanding, unrelated to this task

1. **Re-cut Chapter 3 sections 3.1 and 3.2.** They sit at 27.8 % reduction against the 53–57 % the later sections reached — roughly 1,900 words available, which would take Chapter 3 past 48 %. Largest remaining win in the thesis. Re-run `ledgercheck.py` afterwards, since renumbering 3.1–3.2 moves ledger rows.
2. **`chapter4.tex:29` carries a claim an audit found false** — "no reviewed study isolates a *learned* component against an otherwise identical pipeline". Xia et al.'s RCN-W/RCN-S/RCN-A variants contradict it directly; they compare three architectural modules against a common baseline with all other parameters held constant. Chapter 3's §3.10.2(a) and §3.6.5 were corrected this way; Chapter 4's original was not, and `chapter6.tex` carries a hedged version ("seldom") that survives better. **The shortened `chapter4_shortened.tex` is already corrected** — this applies only to the unshortened original.
3. **Chapter 6** still uses some cost descriptions that the Chapter 3 parts 7–8 batch superseded.

### Chapter-length status

| Chapter | Before | After | Reduction |
|---|--:|--:|--:|
| 3 | 21,856 | 13,350 | 38.9 % |
| 4 | 8,175 | 5,878 | 28.1 % |

Chapter 4 is deliberately less compressed: it is the reproduction chapter, and every parameter value, configuration name, tensor shape and seed was kept.
