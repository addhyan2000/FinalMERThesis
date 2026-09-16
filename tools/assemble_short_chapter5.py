"""Combine the two independently editable Chapter 5 halves into one TeX file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = ROOT / 'contents'
HEADER = r"""% Chapter 5 -- Results, shortened and assembled.
% Source halves: chapter5_short_half1.tex and chapter5_short_half2.tex.
% Rebuild with tools/assemble_short_chapter5.py after editing either half.
% Include IN PLACE OF chapter5.tex; do not also include the separate halves.
% Requires graphicx and the existing thesis support for \checkmark.
% Figure paths use the existing figures/ folder in the Overleaf project root.
\chapter{Results}
\label{ch:results}

The results use the twelve complete LOSO evaluations stored under
\texttt{Ablation\_Study/results/}. The primary metric is pooled macro F1:
sum the confusion matrices across all 25 folds, compute each class's F1,
then average the three class scores. Accuracy is likewise pooled, calculated
from total correct predictions over 156 clips. These differ from the stored
mean-of-folds metrics. Published comparison scores are attributed separately
to their source. All architectural contrasts use one fixed seed, and each
fold's checkpoint was selected on the same held-out subject subsequently
scored; the comparisons must be interpreted within those limits.

"""

def assemble():
    halves = [(CONTENTS / f'chapter5_short_half{i}.tex').read_text(encoding='utf-8')
              for i in (1, 2)]
    output = CONTENTS / 'chapter5_shortened.tex'
    output.write_text(HEADER + '\n\n'.join(s.rstrip() for s in halves) + '\n', encoding='utf-8')
    print(output)

if __name__ == '__main__':
    assemble()
