#!/usr/bin/env bash
# Reassembles 00_Chapter1_Complete.md from the numbered section files.
# Citations are keys ([@key]) in the section files and are resolved to numbers
# against the thesis-wide BIBLIOGRAPHY.md by tools/bibliography.py.
set -euo pipefail
cd "$(dirname "$0")"
OUT=00_Chapter1_Complete.md

cat > "$OUT" <<'HDR'
# Chapter 1 — Introduction

In the moment after a difficult question lands, a face can give something away. A tightening at the brow, a pull at one corner of the mouth — there and gone again before the person has registered producing it, and at odds with the composed expression held over the top of it. Movements of this kind are micro-expressions: brief, involuntary facial movements that betray an emotion its owner is working to keep hidden [@yan2014]. They last under half a second, and they are faint enough that an observer watching in real time will ordinarily miss them entirely.

That combination — informative, and effectively invisible — is what has drawn sustained effort towards recognising them by machine. Yan et al. [@yan2014] motivate the CASME II corpus on the grounds that a robust automatic recogniser "would have broad applications in national safety, police interrogation, and clinical diagnosis"; the capability is likewise reported as promising for lie detection, business negotiation and psychoanalysis [@xia2020a]. Those are the field's motivating claims rather than records of anything deployed, but they are why the problem has kept its audience.

The systems built to attempt it are not single models. They are pipelines, assembled stage by stage from techniques that arrived in the field separately and are now adopted together. This thesis takes one such pipeline apart and measures its stages one at a time.

§1.1 sets out the practical problem that makes such a measurement worth making, and the form of study built to produce it. §1.2 describes how the remainder of the thesis is organised.

---

HDR

for f in $(ls -1 0[1-9]_1.*.md | sort); do
  awk 'f{print} /^## 1\./{if(!f){f=1; print}}' "$f" >> "$OUT"
  printf '\n---\n\n' >> "$OUT"
done

echo "Rebuilt $OUT — $(wc -w < "$OUT" | tr -d ' ') words, $(grep -c '^## 1\.' "$OUT") sections."
python3 ../tools/bibliography.py render >/dev/null
echo "  citations resolved against BIBLIOGRAPHY.md"
