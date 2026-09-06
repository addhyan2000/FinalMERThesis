#!/usr/bin/env bash
# Reassembles 00_Chapter3_Complete.md from the numbered section files
# plus the single consolidated reference list. Run after editing any section.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter3_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 3 — Literature Review

## Scope and conventions

This chapter reviews the literature bearing on each component of the pipeline evaluated in this thesis. It is written to be read continuously: a fact is established once, in the section that owns it, and referred to by cross-reference thereafter. The corpus and its properties are established in §3.1, the metric convention in §3.1.6, and each component's measured contribution in the section that reviews it.

**Sources.** Every work cited is a paper held in the project's `docs/` corpus. Works cited only *inside* those papers are attributed in the text to the reviewed source that reports them, and are not listed as separate references; the reference list at the end of the chapter enumerates which works are handled this way. Two topics that appear in the project's reading list — Grad-CAM++ visual auditing and adversarial identity disentanglement — are excluded because the implemented system does not use them (§3.9).

**Structure.** Each section runs: what the technique addresses → how it works → the published evidence → its limitations → the implications for this thesis → a closing statement of the gap and how this work differs. §3.10 consolidates those ten closing statements into a single research gap, and Table 3.21 collects every point at which the implemented system departs from the literature it draws on.

**Verification.** Every quotation in this chapter was checked verbatim against the source PDF; every reference was confirmed to correspond to a PDF in `docs/`; and every project-specific figure was recomputed from the code and from `Ablation_Study/results/config_*/final_results.json` using pooled macro F1 (mean of `per_class_f1`), not the mean-of-folds column that §3.1.6 rejects. Bibliographic details that could not be confirmed because the copy in `docs/` is a preprint or author manuscript are listed in this folder's `README.md`.

## Contents

| § | Topic | Ablation variable |
|---|---|---|
| 3.1 | The evaluation corpus: CASME II and the benchmark family | — |
| 3.2 | Motion magnification: Eulerian Video Magnification | A |
| 3.3 | Motion representation: dense optical flow | — (input) |
| 3.4 | Deformation representation: the optical strain tensor | — (input) |
| 3.5 | Temporal normalisation | — (input) |
| 3.6 | The learned spatial backbone: shallow 3D CNNs | C |
| 3.7 | Parameter-free attention: SimAM | B |
| 3.8 | The temporal encoder: transformers | D |
| 3.9 | Training under severe class imbalance | — (constant) |
| 3.10 | Synthesis and research gap | — |

---

HDR

for f in $(ls -1 [0-9][0-9]_3.*.md | sort); do
  awk 'f{print} /^## 3\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

# single consolidated reference list
cat 11_References.md >> "$OUT"

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 3\.' "$OUT") sections, $(grep -c '^[A-Z].*([12][0-9]\{3\}[ab]\?)\.' 11_References.md) references."
