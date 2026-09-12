#!/usr/bin/env bash
# Reassembles 00_Chapter5_Complete.md from the numbered section files.
# Citations are keys ([@key]) in the section files and are resolved to numbers
# against the thesis-wide BIBLIOGRAPHY.md by tools/bibliography.py.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter5_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 5 — Results

This chapter reports what the ablation found. Chapter 4 specified the design, the twelve configurations, the architecture and the evaluation protocol; this chapter presents the scores those choices produced and reads them within the bounds Chapter 4 established.

Every figure reported here is computed from the stored per-configuration results under `Ablation_Study/results/`. The primary metric throughout is pooled macro F1 — the mean of the per-class F1 scores taken over a confusion matrix summed across all twenty-five folds — for the reasons given in §4.6.4 and §4.6.5. Where accuracy is reported it is pooled accuracy, not the mean of per-fold accuracies. The distinction is not cosmetic: the two orderings disagree, and §5.1 shows where.

The chapter takes each ablated component in turn, reports its effect over the matched pairs that isolate it, and then draws the four findings together against per-class behaviour, computational cost, and the published literature.
HDR

for f in $(ls -1 0[1-9]_5.*.md | sort); do
  awk 'f{print} /^## 5\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 5\.' "$OUT") sections."
python3 ../tools/bibliography.py render >/dev/null
echo "  citations resolved against BIBLIOGRAPHY.md"
