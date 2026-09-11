#!/usr/bin/env bash
# Reassembles 00_Chapter2_Complete.md from the numbered section files
# plus the single consolidated reference list.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter2_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 2 — Background

This chapter sets out the mechanisms the rest of the thesis depends on, in the order the pipeline applies them: what a micro-expression is, what the recording supplies, how small motions are magnified and converted into a motion representation, what a neural network does, which blocks this system assembles from that foundation, how it is trained under a corpus that cannot be balanced, and how it is measured when that corpus is small.

It explains **what each mechanism is and how it works**. Chapter 3 surveys who has used each one, with what result, and what gap remains; where a fact is needed in both places it is established here and cross-referenced there. Scope is limited to what the implemented system actually does — techniques a reader might expect in a background chapter but which this project does not use are omitted, and their absence is noted where it would otherwise be assumed.

Two constraints recur. The corpus is small enough that its size, rather than any modelling preference, forces most of the design. And several stages depart from the textbook form of the technique they implement; each departure is named where it occurs.

---

HDR

for f in $(ls -1 0[1-9]_2.*.md | sort); do
  awk 'f{print} /^## 2\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

cat 10_References.md >> "$OUT"

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 2\.' "$OUT") sections, $(grep -c '^[A-Z].*([12][0-9]\{3\}[ab]\?)\.' 10_References.md) references."
