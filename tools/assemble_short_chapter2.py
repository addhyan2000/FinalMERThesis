"""Assemble the two Chapter 2 editing halves into one portable LaTeX file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = ROOT / "contents"
HEADER = r"""% Shortened Chapter 2, assembled from chapter2_short_half1.tex and
% chapter2_short_half2.tex by tools/assemble_short_chapter2.py.
% Include this file IN PLACE OF chapter2.tex; do not also include the halves.
\chapter{Background}
\label{ch:background}

Micro-expression recognition combines a weak visual signal with limited, imbalanced training data. The relevant foundations are the phenomenon and its recording, motion magnification, flow and strain, learned feature processing, training and evaluation. The distinctions below identify the assumptions behind these mechanisms and the implementation choices that determine what the architectural comparisons can establish.

"""

def assemble():
    halves = [(CONTENTS / f"chapter2_short_half{i}.tex").read_text(encoding="utf-8")
              for i in (1, 2)]
    output = CONTENTS / "chapter2_shortened.tex"
    output.write_text(HEADER + "\n\n".join(s.rstrip() for s in halves) + "\n",
                      encoding="utf-8")
    print(output)

if __name__ == "__main__":
    assemble()
