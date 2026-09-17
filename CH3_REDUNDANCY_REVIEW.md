# Chapter 3: redundancy and reduction review

Reviewed 17 September 2026. Analysis only: no thesis `.tex` files were changed.

## Recommendation

Reduce `contents/chapter3_shortened.tex` from approximately **16,934 words to an editorial target of 8,500 words** (about 50%). This is a proposed budget, not an achieved reduction or a compiled page estimate. Most of the saving should come from repeated explanations, method settings, and repeated conclusions; removing unique evidence is not the first step.

The current chapter has **10 sections, 76 subsections, 17 tables and 2 figures**. Sections 3.1, 3.2 and 3.5 alone account for 7,984 words, about 47% of the chapter. The main problem is structural: topics repeatedly follow motivation → mechanism → evidence → limitations → implications → research gap, then repeat the same argument in the synthesis.

Use a shorter pattern: **what relevant studies found → what their evidence does not establish → why that matters for this experiment**. Two or three anchor papers usually suffice; additional citations can support a specific exception without receiving a separate paper summary. SimAM already has only two sources, so its problem is repeated explanation, not excessive source count.

## Scope and count method

Reviewed the current Chapter 3 against `chapter2_shortened.tex`, `chapter4_shortened.tex`, `chapter5_shortened.tex`, `chapter6.tex` and `introduction.tex`. Original unshortened chapters are not treated as the authoritative destination for retained material. This is an editorial and cross-chapter consistency review, not a fresh paper-by-paper authentication of every reference.

Counts use the existing `words()` estimator in `tools/check_short_chapter2.py`: headings and visible table/caption prose included; comments, citation keys, command names and mathematics excluded. They are approximate source counts, not TeXcount or PDF counts. All line numbers below refer to the reviewed checkout. Raw counts are recorded in `tmp/ch3_redundancy/counts.json`.

## Highest-priority cuts

1. **Remove repeated mechanisms and equations from the review.** Chapter 2 already explains magnification, brightness constancy, strain, SimAM, focal loss and F1. Chapter 3 needs the published evidence and transfer limitations, not another derivation.
2. **Remove repeated implementation inventories.** The local EVM/flow/strain/TIM/CNN/SimAM/imbalance decision tables restate choices described in Chapter 2 or 4 and again in the divergence ledger. Keep material source-to-project differences once; keep reproduction settings in methodology.
3. **Replace the nine miniature research-gap endings with short relevance sentences.** Keep one substantial, qualified research-gap synthesis at the end. Preserve a local boundary where it prevents misinterpretation, such as strain not being ablated.
4. **Remove repeated numerical displays.** The dataset figure and label table partly repeat the same distribution; Chapter 5 already contains the seven MEGC CASME II-subset UF1 values; the SLSTT table includes another broad leaderboard where only the Mean/LSTM comparison is central.
5. **Delete repeated debugging history and future-work lists.** Earlier run defects and class-collapse histories already have a methodological home; proposed follow-up experiments are extensively repeated in Chapter 6. Moving duplicated material is unnecessary: delete the extra version.

## Section-by-section decisions and budgets

### 3.1 Corpus and benchmarks — 4,071 → about 1,800 words

Source begins at line 45. This section is disproportionately large for a study evaluating one corpus.

- **Cut heavily:** historical posed-dataset catalogue and seven-corpus survey (51–85); stimulus-panel count, lighting apparatus and registration recipe (87–98); LBP-TOP radius/block tutorial and baseline sweep (143–156); detailed SMIC and CAS(ME)² baseline/transfer detours (158–169).
- **Delete one duplicated display:** use a compact label/version account rather than both the dataset-composition figure (130–137) and detailed taxonomy table (108–126). The model does not consume action-unit criteria, so their full taxonomy is unnecessary here.
- **Delete the repeated MEGC leaderboard (191–208):** its CASME II UF1 column already exists in shortened Chapter 5, lines 431–454. Other columns are not literal duplicates; retain them only if a specific argument actually uses them. Keep a concise account of composite evaluation and source provenance.
- **Replace the large threat table (219–257)** with a short limitations paragraph. Demographics, elicitation and annotation are literature issues; own-study fold statistics and implementation choices belong with the experimental sample/protocol. Preserve any unique statistics actually used later before deleting their only account.
- **Merge 3.1.8 and 3.1.9 (259–273)** into a brief statement of relevance; they restate labels, metrics, scarcity and the full ablation design.
- **Keep:** CASME II suitability, spontaneous elicitation/high-speed recording, label-version differences, 145-versus-156 evaluation mismatch and unresolved reconciliation, subject-disjoint evaluation, demographic/laboratory limitations. Distinguish pooled UF1 from fold averaging without repeating its derivation.
- **Anchor papers:** Yan 2014 (`yan2014`) and See 2019 (`see2019`). Li 2013 (`liX2013`) if the three-class precedent is discussed; Davison 2018 only for a retained SAMM demographic comparison. Each extra fact still needs its own appropriate source.

### 3.2 EVM — 2,247 → about 1,100 words

Source begins at line 277.

- **Keep the main controlled evidence:** Li 2018's magnification-on/off results and non-monotonic strength sweep. Use the compact evidence table or a concise numerical sentence; avoid repeating the same values in prose.
- **Keep Bai as a distinct comparison**, with the caveat that its earlier baseline is not a matched magnification-off arm. Compress the mechanism and parameter tutorial (288–295), already covered in Chapter 2.
- **Delete the standalone apex-paper recap (323–328).** At most one sentence can note that an apex-appearance pipeline uses a different amplification setting. Both Li/Huang/Zhao publications do not need full summaries merely to repeat alpha = 30. The proposed explanation of why that setting differs is a hypothesis, not demonstrated evidence.
- **Merge limitations, implications and gap (330–380).** Keep nuisance-motion/artifact risks and the important boundary that the reviewed EVM evidence largely concerns appearance descriptors, whereas this experiment supplies flow to the learner. Remove the repeated settings table, defect narrative and extended hypothesis rehearsal.
- **Anchor papers:** Li X. 2018 (`liX2018`) and Bai 2021 (`bai2021`); optionally one apex paper for a narrow contrast. Four full paper descriptions are unnecessary.

### 3.3 Optical flow — 1,139 → about 600 words

Source begins at line 383.

- Remove brightness-constancy/aperture exposition (395–400), repeated estimator tutorials, the detailed apex-error anecdote and the local decision table (432–457).
- Condense alignment work to one limitation sentence if needed; it was not evaluated here.
- **Keep:** onset–apex versus sequence flow, the evidence motivating motion input, the fixed estimator and lack of an estimator comparison. End with one sentence motivating the spatial/temporal comparison, not a second full gap statement (460–464).
- **Anchor papers:** OFF-ApexNet (`liong2019a`), Zhao 2021 (`zhaoS2021`), and STSTNet (`liong2019b`) where its channel structure matters. Xu 2017 is an exception citation if alignment or linear interpolation remains.

### 3.4 Optical strain — 1,221 → about 700 words

Source begins at line 468.

- Remove the repeated tensor derivation (480–518) from Chapter 3; Chapter 2 already contains the equations. Keep a concise declaration of the actual magnitude difference.
- Collapse the history of strain as feature/weight/channel (521–528) into one comparative paragraph. Liong 2016 can carry the mature descriptor evidence without separate full accounts of every earlier paper.
- Replace the broad results table (533–550) with the result needed for the argument, or a smaller comparison. **Do not remove the matched flow-versus-strain control:** it prevents the false claim that nobody previously isolated strain.
- Remove the implementation table and repeated future-work/gap paragraphs (563–596), preserving one scope sentence.
- **Keep:** three-channel precedent; ambiguous printed STSTNet formula versus the explicitly declared three-term implementation; differentiation can amplify flow error; strain is fixed in every configuration, so this thesis does not measure its independent benefit.
- **Anchor papers:** Liong 2016 (`liong2016`) and STSTNet (`liong2019b`); Shreve 2011 if retaining the spotting origin. Cite Liong 2014a only for retained details unique to it, such as explicit filtering or discretisation.

### 3.5 Temporal normalisation — 1,666 → about 750 words

Source begins at line 599. Rename the section around temporal sampling/normalisation when editing: the implementation is not manifold TIM.

- Merge the TIM definition/history/ten-frame convention (604–621); remove the 120-frame CAS(ME)² detour and exhaustive list of lengths.
- Retain a short contrast between Li's ten-frame result and Ben's different optimum. It establishes that a setting does not transfer automatically. Detailed Newton scores and multiple repeated onset–apex descriptions are peripheral.
- Remove the local settings table (653–669), file-loader branch detail and repeated follow-up list. Unique timing calculations belong in methodology if still needed, rather than occupying a long literature subsection.
- **Keep:** interpolation synthesises frames while this pipeline samples recorded indices; sampling length is fixed and not tested; sampling precedes EVM and changes the physical meaning of the nominal frequency band; short clips can repeat indices. These constrain the EVM conclusion and must survive somewhere authoritative.
- **Anchor papers:** Li X. 2018, Ben 2021 and Lu 2015 for the TIM definition. Xu 2017 is a useful narrow exception if retaining its multi-corpus/linear-interpolation contrast. Do not imply that three selected papers exhaust all relevant evidence.

### 3.6 CNN — 1,368 → about 750 words

Source begins at line 689.

- Merge the three architecture descriptions (699–708); keep their meaningful differences, not all stream/filter counts.
- Summarise Xia's controlled depth/resolution result in one paragraph (710–719), including its composite-database qualification. Drop the nine-resolution inventory and repeated architecture parameter comparisons.
- Delete the duplicated imbalance/augmentation subsection (721–724), the decision table (742–758) and the repeated limitations/gap ending.
- **Keep:** the distinction between temporal convolution and a spatial-only stem; source-specific rather than universal resolution evidence; the stem-removal question. The no-CNN route still has a learned projection. Do not write that analytical flow/strain guarantees a negative CNN effect.
- **Anchor papers:** STSTNet, Xia 2020 shrinking study (`xia2020b`), Zhao 2021. A short Xia 2020 recurrent-network citation (`xia2020a`) is justified if its counterexample or separate temporal architecture is retained.

### 3.7 SimAM — 961 → about 500 words

Source begins at line 774. This is already a small source set; structural repetition is the problem.

- Delete the SE/CBAM/GC/ECA/SRM parameter catalogue (784–801): these modules are not compared experimentally.
- Delete the repeated energy/variance equations and tutorial (812–827), covered by Chapter 2. Keep the reference-code versus printed-divisor distinction where the equations are owned.
- Merge repeated transfer limits and remove the decision table (851–867) and future-work list.
- **Keep:** natural-image evidence does not validate MER; parameter-free does not mean runtime-free; whole-clip rather than per-frame statistics; unswept lambda; effect conditional on this model's CNN.
- **Anchor paper:** Yang 2021 (`yang2021`). Xia 2020 is optional motivation for parameter-free MER modules, not a validation of SimAM.

### 3.8 Transformer — 964 → about 650 words

Source begins at line 881. Cut less aggressively here: distinguishing the tested component from SLSTT is central.

- Reduce the ViT tutorial (893–898) to one motivation/caution paragraph.
- Reduce the eight-row results table (910–929) to SLSTT-Mean versus SLSTT-LSTM, or one quantitative sentence. The other six models do not need another leaderboard. Do not silently substitute similarly named values from a different source/protocol.
- Remove exact local hyperparameter inventory (945) and repeated future-work/parameter calculation (968).
- **Keep:** SLSTT uses spatial attention plus temporal recurrence; this thesis uses a temporal encoder. Keep the six material differences once, ideally in the existing contrast table (947–964), with only a brief ledger summary. Preserve the interpretation of the Mean/LSTM comparison and the protocol caveat.
- **Anchor papers:** Zhang 2022 (`zhang2022`) and Dosovitskiy 2021 (`dosovitskiy2021`). These are enough for the core argument.

### 3.9 Imbalance — 1,386 → about 650 words

Source begins at line 978.

- Delete the excluded-techniques paragraph (983), repeated class-count motivation (985–988), separate three-intervention tutorial and focal derivation (990–1012), and third metric defence (1019–1022).
- Merge the literature evidence and limitations: Zhao's focal-versus-cross-entropy comparison and Xia's augmentation/balanced-loss ablations deserve concise retention.
- Delete the settings table (1036–1052); Chapter 4 already documents the recipe. Remove the class-collapse/debugging story (1054–1058) from the review; earlier runs are already discussed in methodology.
- **Keep:** difficulty weighting differs from class-frequency correction; the training recipe was held fixed. The final sweep cannot establish the guard's causal benefit or that correcting at exactly one place is universally best.
- **Anchor papers:** Zhao 2021 and Xia 2020 (`xia2020a`). A metric citation belongs in the single protocol discussion, not another full subsection here.

### 3.10 Synthesis — 1,739 → about 900 words

Source begins at line 1070.

- Delete the roadmap and full component recap (1073–1078).
- Merge the four limitations and research-gap statement (1081–1101). Keep the acknowledged prior component ablations; narrow the gap to this joint comparison under this setting.
- Keep a brief experimental response: four switches, twelve architecturally valid configurations, complete 25-subject LOSO, single corpus, shared fixed choices. The matrix and protocol details belong in Chapter 4.
- Delete the three historical-contribution bullets (1111–1115) and final recap (1151–1154).
- Keep one concise divergence ledger (1119–1147) only for material differences that affect interpretation. Avoid duplicating local comparison tables at full length. Update wording to match corrected shortened Chapters 2 and 5.
- No new paper summaries are needed. Cite the retained evidence where useful instead of relying on long chains of internal references.

The section targets total **8,400 words**, plus **100 words for the introduction**, giving **8,500**. They are editorial allocation targets; do not add savings from overlapping cuts to these totals.

## Tables and figures

Aim provisionally for **about 4–6 useful tables**, rather than 17, depending on whether short comparisons read better in prose:

- A compact corpus/protocol comparison if needed.
- The controlled EVM evidence table if retained instead of its prose equivalent.
- A compact strain comparison if its matched control needs a table.
- The SLSTT-versus-thesis architecture contrast.
- A concise consolidated divergence ledger if it does not repeat the preceding table.

The dataset figure can be removed from Chapter 3 if a compact account carries its essential information. The fold-composition figure is methodological: retain it there only if it adds information beyond the existing protocol text. Moving a unique figure does not reduce total thesis pages; deleting duplicated presentations does.

LaTeX's generated list of tables/figures follows retained captions when rebuilt. Update any manual inventories, stale figure/table references and hardcoded section/ledger item numbers during a subsequent edit. Bibliography entries should not be deleted just because their citation disappears from Chapter 3; other chapters may still cite them.

## Claims to narrow during shortening

These are consistency issues, not reasons to discard inconvenient evidence:

- **EVM implementation:** line 291 can imply Butterworth is implemented; shortened Chapter 2 explicitly identifies a hard Fourier mask. Keep the accurate distinction.
- **Temporal normalisation:** the claim at line 684 that none of the length studies includes CASME II contradicts the Xu description at line 632. Also, 30 interpolated frames is not direct validation of 33 sampled images/32 flow fields.
- **Novelty:** line 1099 says the field has never measured stage contributions, but line 1085 explicitly lists prior matched ablations. Describe the specific joint comparison missing from the reviewed set, not the entire field as untested.
- **CNN:** composite-study resolution sensitivity does not prove a universal threshold or explain the observed loss. Shortened Chapter 5, line 346, correctly keeps the mechanism unresolved.
- **Class weighting:** line 1054's “correct at exactly one point” overstates uncontrolled historical observations. Shortened Chapter 5, line 399, explicitly says the guard's causal effect was not isolated.
- **Synthesis:** line 1154's claim that each parameter comes from published sweeps conflicts with unswept or transferred settings. Remove this rhetorical claim; retain the specific evidence and limitations.

## Cross-reference safeguards

An automated scan of **active shortened** chapters found no Chapter 3 label references in shortened Chapters 2 or 5. Some audit notes refer to the original Chapter 5; those old references are not constraints on the shortened manuscript.

Introduction refers to the literature chapter and final gap. Shortened Chapter 4 refers to `sec:corpus`, `sec:label-taxonomy`, `sec:megc-metrics`, `sec:evm-implications`, `sec:imbalance-double-correction` and `sec:research-gap`. Chapter 6 refers extensively to the local implications, gap and divergence sections. Preserve meaningful labels on the merged discussions or retarget references to their authoritative destination. Existing links do not require retaining entire redundant subsections.

Chapter 6 also repeats some older broad claims; it needs a consistency pass after shortening, rather than forcing Chapter 3 to preserve those claims. A final compiled check is necessary to establish actual page savings and catch broken references. This review does not change the thesis or claim it now meets the 100-page goal.
