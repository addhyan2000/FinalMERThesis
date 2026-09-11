#!/usr/bin/env bash
# Reassembles 00_Chapter4_Complete.md from the numbered section files.
# Citations are keys ([@key]) in the section files and are resolved to numbers
# against the thesis-wide BIBLIOGRAPHY.md by tools/bibliography.py.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter4_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 4 — Methodology

This chapter specifies the study as it was carried out: the experimental design, the corpus and its preparation, the twelve configurations trained, the architecture and its ablatable components, the training procedure, the evaluation protocol, and the computational conditions under which the sweep ran.

Chapter 2 explains what each mechanism is and how it works, and Chapter 3 reviews the literature for each and identifies the gap. This chapter states what was done and with which values, so that the work can be reproduced; where a mechanism has already been explained it is cross-referenced rather than re-derived. Every parameter reported here is taken from the configuration and code that produced the results in Chapter 5.

---

HDR

for f in $(ls -1 0[1-9]_4.*.md | sort); do
  awk 'f{print} /^## 4\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 4\.' "$OUT") sections."
python3 ../tools/bibliography.py render >/dev/null
echo "  citations resolved against BIBLIOGRAPHY.md"
