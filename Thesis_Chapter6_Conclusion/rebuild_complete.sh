#!/usr/bin/env bash
# Reassembles 00_Chapter6_Complete.md from the numbered section files.
# Citations are keys ([@key]) in the section files and are resolved to numbers
# against the thesis-wide BIBLIOGRAPHY.md by tools/bibliography.py.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter6_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 6 — Conclusion

This thesis investigated which stages of the standard micro-expression recognition pipeline are responsible for its performance. It treated that pipeline as a factorial experiment rather than as a proposal: four components — Eulerian magnification, a convolutional spatial stem, parameter-free attention, and a temporal transformer — varied independently across twelve configurations covering every architecturally valid cell of the matrix, each trained from scratch and scored under complete 25-fold leave-one-subject-out on CASME II, with every non-varied factor held identical.

This chapter closes the account. §6.1 states what the study contributes; §6.2 sets out the limits within which those contributions hold; §6.3 sets out the work the design leaves undone, including the commitments made to it earlier in the thesis; and §6.4 says what a reader should take away.

---

HDR

for f in $(ls -1 0[1-9]_6.*.md | sort); do
  awk 'f{print} /^## 6\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 6\.' "$OUT") sections."
python3 ../tools/bibliography.py render >/dev/null
echo "  citations resolved against BIBLIOGRAPHY.md"
