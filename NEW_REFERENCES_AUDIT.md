# Audit of the 35 added bibliography entries

Audited 23 September 2026 against `bib/thesis.bib` and the six active thesis chapters.

## Decision

**Keep all 35 additions. Removed: 0.** All represent real publications, and each has a defensible purpose in the current thesis. All 35 are cited in the active Chapter 2. The bibliography contains 65 entries in total, all cited in the active manuscript.

These additions mainly supply original or standard sources for the methods explained in the background chapter. A paper need not experiment on CASME II to be the appropriate source for a technique used in the pipeline. Conversely, these general-method papers should not be presented as experimental evidence that a component improves micro-expression recognition.

## Verification and limits

- Queried Crossref by the exact DOI for all **22 additions with DOI fields**. All resolved to matching publications. Checked titles, author lists, publication years, venues and available page/volume information; missing fields in registry records were not treated as errors. The responses are saved in `tools/new_reference_metadata.json`.
- Checked the other **13 additions** against official proceedings, journal pages, author/book sites or author-submitted arXiv records, linked individually below. The arXiv records for Adam, AdamW, SGDR and Mixed Precision Training explicitly confirm their respective ICLR publication years.
- Inspected the citation contexts in the active chapters. Consulted abstracts and relevant passages for method relevance, with closer inspection of spatial dropout and evaluation definitions. This is an existence, metadata and relevance audit, not a claim that every sentence of all 35 works was read or that every equation in the thesis was re-audited.
- Found no substantive bibliographic discrepancy requiring removal or correction. Differences in capitalization, expanded initials, author-name styling, and article numbers used in `pages` do not imply fabricated publications. GELU and Layer Normalization are correctly identified as arXiv preprints, rather than assigned an invented conference venue.
- The local manuscript check reports 65 bibliography entries, 65 cited keys, no invalid keys, no orphan entries, no duplicate labels and no unresolved references. A full Overleaf compilation was not performed.

## One citation correction

**`sokolova2009` is real and relevant, but is not the best citation for the exact displayed macro-F1 formula.** Table 3 gives a macro F-score formed from macro precision and macro recall; the thesis uses the arithmetic mean of per-class F1 scores. Those operations are not generally equivalent. The author's uploaded [full text, Table 3](https://www.researchgate.net/publication/222674734_A_systematic_analysis_of_performance_measures_for_classification_tasks) and the analysis in [Opitz and Burst, Macro F1 and Macro F1](https://arxiv.org/abs/1911.03347) establish the distinction.

Changed only the introductory sentence beside `eq:macro-f1` in `contents/chapter2_shortened.tex` to attribute the mean-of-per-class-F1 definition explicitly to the existing `see2019` reference. The MEGC paper's page 3, Section III.B.1 explicitly describes that averaging procedure; the local source PDF was visually checked. **The formula and results were not changed.** Sokolova and Lapalme remains cited for micro-averaging and stays in the bibliography. Opitz and Burst was used for this audit only and was not added to the bibliography.

## Individual decisions

Line numbers below refer to `contents/chapter2_shortened.tex` at audit time. Every decision is **KEEP**.

### Micro-expressions and facial coding

1. **`ekman1969` — Ekman and Friesen, Nonverbal Leakage and Clues to Deception (1969).** Real: exact [DOI record](https://doi.org/10.1080/00332747.1969.11023575). Relevant to the historical emotional-leakage account at line 28. It does not by itself establish reliable automatic lie detection.

2. **`ekman1978` — Ekman and Friesen, Facial Action Coding System (1978).** Real: the [authors' official FACS page](https://www.paulekman.com/facial-action-coding-system/) confirms the original 1978 manual. Relevant to action units and facial coding at line 57. It is a manual, not a journal article; `@book` is appropriate. The historical citation is valid here; it should not imply the thesis newly coded faces using the current manual.

3. **`porter2008` — Porter and ten Brinke, Reading Between the Lies (2008).** Real: [publisher record and abstract](https://journals.sagepub.com/doi/10.1111/j.1467-9280.2008.02116.x), including the full subtitle omitted from the short Crossref title. Relevant to empirical study of concealed/falsified emotional expressions at line 28. The study's limited observations do not establish that every brief expression proves deception.

4. **`oh2018` — Oh et al., A Survey of Automatic Facial Micro-Expression Analysis (2018).** Real: [publisher full text](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.01128/full). Directly relevant survey of the thesis's recognition problem, databases and methods, cited at line 28. Overlap in coverage with another survey is not a reason to call it irrelevant.

5. **`pfister2011` — Pfister et al., Recognising Spontaneous Facial Micro-Expressions (2011).** Real: matching [DOI](https://doi.org/10.1109/ICCV.2011.6126401) and [author-hosted paper](https://tomas.pfister.fi/files/pfister11microexpressions.pdf). Relevant to temporal interpolation at line 206. The thesis appropriately cites it while explaining that the implemented uniform sampling does not perform TIM.

### Motion processing

6. **`wu2012` — Wu et al., Eulerian Video Magnification for Revealing Subtle Changes in the World (2012).** Real: matching [DOI record](https://doi.org/10.1145/2185520.2185561). Direct source for the magnification mechanism ablated by the thesis, cited at line 97. The original general-video task does not reduce its methodological relevance.

7. **`burt1983` — Burt and Adelson, The Laplacian Pyramid as a Compact Image Code (1983).** Real: matching [DOI](https://doi.org/10.1109/TCOM.1983.1095851) and [author-laboratory abstract](https://persci.mit.edu/pub_abstracts/pyramid83_abs.html). Relevant to the multiscale decomposition used by the magnifier, line 114. Its image-coding origin is not grounds for deletion.

8. **`horn1981` — Horn and Schunck, Determining Optical Flow (1981).** Real: matching [DOI](https://doi.org/10.1016/0004-3702(81)90024-2) and [author-hosted paper](https://people.csail.mit.edu/bkph/papers/Optical_Flow_OPT.pdf). Relevant to brightness constancy and the optical-flow constraint at lines 162 and 167. The thesis does not claim its implemented estimator is Horn-Schunck.

9. **`farneback2003` — Farnebäck, Two-Frame Motion Estimation Based on Polynomial Expansion (2003).** Real: [Springer record](https://link.springer.com/chapter/10.1007/3-540-45103-X_50) confirms author, pages and LNCS volume 2749. Directly relevant to the implemented optical-flow estimator, line 206.

### Neural-network foundations and architecture

10. **`rumelhart1986` — Rumelhart, Hinton and Williams, Learning Representations by Back-Propagating Errors (1986).** Real: exact [DOI record](https://doi.org/10.1038/323533a0). Relevant foundational source for the backpropagation description at line 243. The text need not claim that this paper was the earliest discovery of every form of the chain-rule method.

11. **`lecun1998` — LeCun et al., Gradient-Based Learning Applied to Document Recognition (1998).** Real: exact [DOI](https://doi.org/10.1109/5.726791) and [coauthor's publication record](https://leon.bottou.org/papers/lecun-98h). Relevant to convolution, shared weights and pooling at lines 250 and 261. The document-recognition application does not make the CNN foundations irrelevant.

12. **`lecun2015` — LeCun, Bengio and Hinton, Deep Learning (2015).** Real: exact [Nature DOI record](https://doi.org/10.1038/nature14539). Relevant broad background for neural networks at line 216. This Nature review is a different work from the 2016 textbook despite the identical short title.

13. **`goodfellow2016` — Goodfellow, Bengio and Courville, Deep Learning (2016).** Real: [official book site and recommended BibTeX](https://www.deeplearningbook.org/). Relevant textbook support for units, layers, pooling, capacity and overfitting, lines 216, 226, 261 and 266. Generality is appropriate for those textbook-level explanations.

14. **`nair2010` — Nair and Hinton, Rectified Linear Units Improve Restricted Boltzmann Machines (2010).** Real: [official ICML paper](https://icml.cc/2010/papers/432.pdf). Relevant foundational ReLU reference at line 231. The thesis cites the activation, not a claim that its model is an RBM.

15. **`hendrycks2016` — Hendrycks and Gimpel, Gaussian Error Linear Units (2016).** Real: [arXiv record](https://arxiv.org/abs/1606.08415), originally submitted in 2016. Directly relevant to the GELU activation/formula, line 231. Correctly labelled a preprint.

16. **`ji2013` — Ji et al., 3D Convolutional Neural Networks for Human Action Recognition (2013).** Real: exact [DOI](https://doi.org/10.1109/TPAMI.2012.59) and [IEEE record](https://ieeexplore.ieee.org/document/6165309/). Relevant to the 3D convolution concept at line 279. The 2012 DOI/early-publication date and 2013 journal issue are compatible. The thesis correctly distinguishes its spatial-only kernels from temporal convolution.

17. **`tran2015` — Tran et al., Learning Spatiotemporal Features with 3D Convolutional Networks (2015).** Real: exact [DOI record](https://doi.org/10.1109/ICCV.2015.510). Relevant to spatiotemporal convolution at line 279. It is architectural background, not evidence that the thesis implemented or reproduced C3D.

18. **`ioffe2015` — Ioffe and Szegedy, Batch Normalization (2015).** Real: [official ICML/PMLR record](https://proceedings.mlr.press/v37/ioffe15.html), pages 448–456. Directly relevant to the normalization formula and implemented block, line 286.

19. **`srivastava2014` — Srivastava et al., Dropout (2014).** Real: [JMLR record](https://jmlr.org/papers/v15/srivastava14a.html), volume 15, article 56, pages 1929–1958. Directly relevant to regularization at line 294. The number 56 is the journal's article identifier, not a fabricated issue.

20. **`tompson2015` — Tompson et al., Efficient Object Localization Using Convolutional Networks (2015).** Real: exact [DOI](https://doi.org/10.1109/CVPR.2015.7298664); [paper Section 3.2](https://openaccess.thecvf.com/content_cvpr_2015/papers/Tompson_Efficient_Object_Localization_2015_CVPR_paper.pdf) introduces SpatialDropout. Relevant to dropping whole feature maps at line 294. The paper supplies the channel-wise dropout concept; precise PyTorch `Dropout3d` API semantics are an implementation matter, not a claim that this paper introduced that API.

21. **`ba2016` — Ba, Kiros and Hinton, Layer Normalization (2016).** Real: [arXiv record](https://arxiv.org/abs/1607.06450). Directly relevant to token-wise layer normalization in the transformer/head, line 294. Correctly labelled a preprint.

22. **`he2016` — He et al., Deep Residual Learning for Image Recognition (2016).** Real: exact [DOI](https://doi.org/10.1109/CVPR.2016.90) and [CVPR record](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). Relevant to residual connections at line 328. It is not being cited as evidence that the thesis uses a ResNet backbone.

23. **`vaswani2017` — Vaswani et al., Attention Is All You Need (2017).** Real: [official proceedings](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). Essential source for scaled dot-product attention, multiple heads and sinusoidal positional encoding, lines 318, 324 and 326. The conference was styled NIPS in 2017; the entry's modern NeurIPS abbreviation still identifies the same proceedings.

24. **`xiong2020` — Xiong et al., On Layer Normalization in the Transformer Architecture (2020).** Real: [official ICML/PMLR record](https://proceedings.mlr.press/v119/xiong20b.html), pages 10524–10533. Directly relevant to pre-LN placement at line 328. The paper studies pre-LN and cites earlier proposals; avoid describing it as proof that these authors first invented pre-LN.

### Training and optimization

25. **`he2009` — He and Garcia, Learning from Imbalanced Data (2009).** Real: exact [DOI record](https://doi.org/10.1109/TKDE.2008.239). Relevant to the skewed class distribution and sampling strategies, lines 367 and 384. A DOI assigned in 2008 does not contradict the 2009 journal issue.

26. **`buda2018` — Buda, Maki and Mazurowski, A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks (2018).** Real: exact [journal DOI](https://doi.org/10.1016/j.neunet.2018.07.011) and [author-submitted manuscript](https://arxiv.org/abs/1710.05381). Relevant empirical background for oversampling, line 384. It does not independently establish that the thesis's sampler is optimal on CASME II.

27. **`lin2017` — Lin et al., Focal Loss for Dense Object Detection (2017).** Real: exact [conference DOI](https://doi.org/10.1109/ICCV.2017.324) and [author manuscript](https://arxiv.org/abs/1708.02002). Essential source for focal modulation, line 372. Its object-detection application does not make the loss irrelevant to classification; the thesis separately describes its label-smoothed implementation.

28. **`szegedy2016` — Szegedy et al., Rethinking the Inception Architecture for Computer Vision (2016).** Real: exact [conference DOI](https://doi.org/10.1109/CVPR.2016.308) and [author manuscript](https://arxiv.org/abs/1512.00567). Relevant source for label smoothing, line 401. No Inception backbone is required for that citation to be valid.

29. **`kingma2015` — Kingma and Ba, Adam: A Method for Stochastic Optimization (2015).** Real: [author-submitted record](https://arxiv.org/abs/1412.6980) explicitly confirms ICLR 2015. Relevant to Adam, the basis of AdamW, line 425. The 2014 preprint date does not invalidate the 2015 conference citation.

30. **`loshchilov2019` — Loshchilov and Hutter, Decoupled Weight Decay Regularization (2019).** Real: [author-submitted record](https://arxiv.org/abs/1711.05101) explicitly confirms ICLR 2019. Directly relevant to AdamW and decoupled weight decay, line 425.

31. **`loshchilov2017` — Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts (2017).** Real: [author-submitted record](https://arxiv.org/abs/1608.03983) confirms ICLR 2017. Relevant to the cosine-annealing schedule at line 425. It should not be used to claim that the thesis implemented warm restarts or that a linear warmup is the same as a restart; the present text cites cosine annealing specifically.

32. **`micikevicius2018` — Micikevicius et al., Mixed Precision Training (2018).** Real: [author-submitted record](https://arxiv.org/abs/1710.03740) confirms all eleven authors and ICLR 2018. Directly relevant to mixed precision and loss/gradient scaling, line 427.

### Evaluation

33. **`sokolova2009` — Sokolova and Lapalme, A Systematic Analysis of Performance Measures for Classification Tasks (2009).** Real: exact [publisher DOI](https://doi.org/10.1016/j.ipm.2009.03.002). Relevant to classification metrics and micro/macro averaging. **Keep, with the citation correction explained above:** the reference remains at line 457, while line 452 now uses `see2019` for the exact macro-F1 definition.

34. **`mcnemar1947` — McNemar, Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages (1947).** Real: exact [DOI record](https://doi.org/10.1007/BF02295996). Relevant to the discussion of paired per-clip comparisons at line 480. Classical McNemar compares paired binary outcomes, such as correct/incorrect classifications; it does not directly test a difference in macro-F1. Retain the reference, and avoid implying otherwise when drafting future-work claims.

35. **`varma2006` — Varma and Simon, Bias in Error Estimation When Using Cross-Validation for Model Selection (2006).** Real: exact DOI and [publisher full text](https://link.springer.com/article/10.1186/1471-2105-7-91). Directly relevant to the absence of an inner validation split and optimistic checkpoint selection, lines 471 and 478. The original bioinformatics setting does not limit the relevance of its model-selection bias argument.

## Files changed by this audit

- `contents/chapter2_shortened.tex`: one citation sentence corrected; no formulas or experimental results changed.
- `NEW_REFERENCES_AUDIT.md`: this record of all 35 decisions and their evidence.
- `tools/new_reference_metadata.json`: exact-DOI lookup evidence for the 22 DOI-bearing additions, plus the inspected bibliography fields for all 35 additions. The `no_doi` status means no DOI field was supplied, not that the reference failed verification; those entries are verified through the sources above.

`bib/thesis.bib` was left unchanged by this audit. Its pre-existing additions were retained in full.
