#!/usr/bin/env bash
# Reassembles 00_Chapter2_Complete.md from the numbered section files
# plus the single consolidated reference list.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter2_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 2 — Background

## Scope and conventions

This chapter sets out the concepts and techniques needed to follow the methodology in Chapter 4 and the results in Chapter 5. It explains **what each mechanism is and how it works**; Chapter 3 surveys who has used each one, with what result, and what gap remains. Where a fact is needed in both places it is established here and cross-referenced there.

**Scope is limited to what the system actually does.** Every technique described is present in the implemented pipeline. Techniques a reader might expect in a background chapter but which this project does not use — quantisation, pruning and knowledge distillation among them — are not covered, and their absence is noted at the point where it would otherwise be assumed (§2.5).

**Sources.** Every work cited is a paper held in the project's `docs/` corpus. Standard machine-learning material for which the corpus holds no source is presented without citation rather than with an invented one; the reference list states which topics are handled that way. Works cited only inside the reviewed papers are attributed in the text to the paper that reports them.

## Contents

| § | Topic |
|---|---|
| 2.1 | The phenomenon: micro-expressions as involuntary, brief, low-intensity movement |
| 2.2 | From video to motion: optical flow and optical strain |
| 2.3 | Motion magnification |
| 2.4 | Network building blocks |
| 2.5 | Learning under scarcity and skew |
| 2.6 | Evaluating on a small corpus |

---

HDR

for f in $(ls -1 0[1-6]_2.*.md | sort); do
  awk 'f{print} /^## 2\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

cat 07_References.md >> "$OUT"

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 2\.' "$OUT") sections, $(grep -c '^[A-Z].*([12][0-9]\{3\}[ab]\?)\.' 07_References.md) references."
