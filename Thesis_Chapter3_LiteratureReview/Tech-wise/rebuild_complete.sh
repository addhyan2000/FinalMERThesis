#!/usr/bin/env bash
# Reassembles 00_Chapter3_Complete.md from the numbered section files
# plus the single consolidated reference list. Run after editing any section.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter3_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 3 — Literature Review

This chapter reviews the literature bearing on each component of the recognition pipeline evaluated in this thesis: the corpus on which every experiment is run, the four techniques placed under ablation, the three input transformations they operate on, and the training regime held constant across all of them.

It is organised as a single continuous argument rather than as ten independent surveys. A fact is established once, in the section that owns it, and referred to by cross-reference thereafter. The corpus and its properties are established in §3.1, the metric convention in §3.1.6, and each component's measured contribution in the section that reviews it. Each section runs from what the technique addresses, through how it works and what the published evidence shows, to its limitations and the implications for this thesis, closing with a statement of the gap and how this work differs.

Two claims recur and are worth stating at the outset. The first is that on a corpus of 156 clips, several decisions usually presented as modelling choices are in fact forced by properties of the data. The second is that several results usually attributed to architecture turn out, on inspection, to be attributable to the evaluation protocol. Section 3.10 consolidates the nine closing statements into a single research gap, and collects every point at which the implemented system departs from the literature it draws on.

---

HDR

for f in $(ls -1 [0-9][0-9]_3.*.md | sort); do
  awk 'f{print} /^## 3\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

# single consolidated reference list
cat 11_References.md >> "$OUT"

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 3\.' "$OUT") sections, $(grep -c '^\[[0-9][0-9]*\] ' 11_References.md) references."
