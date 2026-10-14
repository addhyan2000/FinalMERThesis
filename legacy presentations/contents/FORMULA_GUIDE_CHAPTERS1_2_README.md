# Formula guide — first batch: Chapters 1 and 2

Open `formula_guide_chapters1_2.tex`. It is a **standalone LaTeX document** with its own embedded nine-paper bibliography. No images or separate bibliography upload are required.

## Use in Overleaf

Upload this file and select it as the **Main document** to compile the guide separately. It can remain in `contents/`. It includes its own document class and preamble, so do not directly `\input` the entire file inside an existing thesis document. The main thesis chapters are unchanged.

## Coverage

- Chapter 1 uses `contents/introduction.tex`; no shortened introduction exists. It contains **no explicit formulas**, which the guide states rather than inventing chapter equations.
- Chapter 2 uses **`contents/chapter2_shortened.tex`**, in its original topic order.
- All **16 displayed equation blocks** are preserved mathematically. The 82 source math spans, including inline calculations, repeated symbols, dimensions and ranges, are mapped to explanation topics by the checker.
- The guide contains **29 numbered equation blocks** overall. Additional blocks promote existing inline formulas or explicitly formalise prose, such as input normalisation and pooling confusion matrices. They are distinguished from equations printed in the chapter.
- Each formula explains its purpose, symbols and relevant assumptions, with its thesis subsection and a paper/page/equation locator where supported.
- Chapters 3–6 are outside this first batch and can be added in subsequent chapter-order batches.

## Reference policy and checks

All nine cited papers are present in `docs/`. Existing thesis keys are reused in the embedded bibliography, in first-use order. Sources include Yan for recording specifications, Bai for the EVM expansion, OFF-ApexNet for flow/ReLU/evaluation definitions, Shreve for strain, Yang for SimAM, Dosovitskiy for attention, Zhang for cross-entropy, Zhao for focal loss and See for pooled UF1.

A paper presenting a formula is not automatically its original inventor. The guide expressly distinguishes those roles. Batch normalisation, label smoothing, convolutional parameter counts, inverse-frequency sampling and exact project normalisations are identified as standard definitions or code-derived expressions where no exact originating paper was verified in the supplied collection. No paper was invented or cited merely because its method uses the same concept.

The guide retains important differences: small-strain versus general finite strain; single versus doubled shear in the scalar magnitude; SimAM's printed `M` variance versus its reference code's `M-1`; image versus clip context; per-head attention scaling; unsmoothed focal probability versus smoothed cross-entropy; and pooled versus mean-fold F1. The explanatory input-normalisation equations explicitly apply standardisation after augmentation.

One existing bibliography error was corrected: `bib/thesis.bib` now names **Mengjiong Bai**, matching the supplied PDF, instead of Mei Bai. The existing citation key and abbreviated `Bai, M.` entry in `BIBLIOGRAPHY.md` remain valid. The guide's embedded reference also uses Mengjiong.

## Validation

Run from the repository root:

```text
python tools/check_formula_guide_chapters1_2.py
```

Checks passed for all sixteen source equations, inline coverage mapping, nine local paper files, bibliography completeness/order, unique labels and balanced LaTeX structure. A report-only source reviewer checked the formula provenance and completed guide; the report is in `tmp/formula_guide/ch2_late_sources.md`.

The local checkout has no LaTeX engine. Final equation wrapping, page layout and table-of-contents pagination must therefore be checked by compiling in Overleaf. Compile to convergence so its contents page and equation links update.
