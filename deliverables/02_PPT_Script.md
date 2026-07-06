# PowerPoint Content Script
## "What Actually Works in Micro-Expression Recognition? A Component Ablation on CASME II"

*Slide-by-slide layout, bullets, and speaker notes. 15 slides + backup. Figures referenced live in `results_weekend/{loso,holdout}/plots/`.*

---

### SLIDE 1 — Title

**Visual layout:** Full-bleed title slide. Centre: title + subtitle. Bottom strip: author, institution, date. Faint background image of an optical-flow face heatmap.

**On-slide text:**
- **What Actually Works in Micro-Expression Recognition?**
- A 12-configuration component ablation on CASME II, under LOSO and hold-out validation
- Addhyan · MER Thesis · 2026

**Speaker notes:** "Micro-expressions are involuntary facial movements lasting under a fifth of a second. Rather than propose yet another architecture and claim it wins, this work asks the more useful question: of the four components we could stack, which ones *actually* earn their place? I answer that by training and evaluating every combination."

---

### SLIDE 2 — The Challenge

**Visual layout:** Left column: 3 icon bullets (brief / faint / rare). Right column: a timeline graphic showing onset→apex→offset over ~0.2 s.

**On-slide text:**
- Micro-expressions: **< 1/5 second**, involuntary, leak concealed emotion
- **Faint** — motion is often sub-perceptual to the human eye
- **Rare & imbalanced** — CASME II: 63 % Negative, 20 % Positive, 16 % Surprise
- **Tiny** — only **156 usable clips**, 25 subjects
- Applications: deception analysis, clinical affect, HCI

**Speaker notes:** "Three things make this hard. The signal is brief and faint — you often can't see it in real time. The data is scarce and heavily imbalanced. And it's spontaneous, so even human coders disagree on labels. Every design decision downstream is a response to one of these three pressures."

---

### SLIDE 3 — The Four Levers (What We Ablate)

**Visual layout:** A 4-block horizontal pipeline diagram: EVM → 3D-CNN(+SimAM) → Transformer → Classifier. Each block a toggle switch icon.

**On-slide text:**
- **A · EVM** (data-level): amplify sub-perceptual motion before feature extraction
- **B · SimAM** (model): *parameter-free* spatial attention on CNN features
- **C · 3-D CNN** (model): three-stream spatial–temporal backbone
- **D · SLSTT Transformer** (model): long-range temporal encoder
- → **2⁴ = 16** combinations; 12 valid (SimAM needs a CNN)

**Speaker notes:** "Four levers. EVM is a data-level switch — it just picks a different precomputed tensor set. The other three add or remove network modules. Toggling all four gives sixteen combinations; four are degenerate because SimAM has nothing to attend to without a CNN, leaving twelve real experiments."

---

### SLIDE 4 — Methodology: The Pipeline

**Visual layout:** Top-to-bottom flow diagram (reuse the ASCII pipeline from the report as a clean graphic). Right margin: the tensor shape `[3, 32, 224, 224]` called out.

**On-slide text:**
- Raw video → face align → **(opt.) EVM** → resample to **32 frames**
- Compute **optical flow (u, v) + optical strain** → 3-channel motion tensor
- 3-D CNN → sequence of 32 × 96-d vectors → temporal encoder → linear head
- **We feed motion, not RGB** — identity-agnostic by construction

**Speaker notes:** "The key methodological choice is on the input: we never feed raw pixels. Each clip becomes three motion channels — horizontal flow, vertical flow, and optical strain, which measures deformation and ignores rigid head motion. This isolates *how the skin moves*, which is what a micro-expression actually is, and as a bonus it makes the network blind to subject identity."

---

### SLIDE 5 — Why These Features & Model (not RGB, not deep nets)

**Visual layout:** Two-column "Chosen ✓ / Rejected ✗" table.

**On-slide text:**
- ✓ **Motion tensors** — signal lives in deformation, not appearance
- ✓ **3-D CNN** — joint spatial-temporal micro-texture
- ✓ **SimAM** — attention at **zero added parameters**
- ✗ RGB + 2-D CNN + LSTM — appearance dominates, leakage risk
- ✗ I3D / R(2+1)D deep nets — far too many params for 156 clips
- ✗ Apex-frame-only — discards the defining temporal dynamics

**Speaker notes:** "Every rejection traces back to data scarcity. Deep video backbones would overfit instantly on 156 clips. RGB pipelines let the model cheat on identity. Apex-only throws away the motion. We deliberately chose the *lowest-capacity* design that still captures spatial-temporal deformation — and even that, as we'll see, turned out to have one component too many."

---

### SLIDE 6 — Why Macro-F1 (Metric Selection)

**Visual layout:** Left: a big callout box with the majority-baseline numbers. Right: mini bar showing accuracy >> macro-F1.

**On-slide text:**
- Classes are 63 / 20 / 16 % → **accuracy is misleading**
- "Always predict Negative" scores: **0.66 acc (LOSO) · 0.75 acc (hold-out)**
- …but only **~0.27–0.29 macro-F1**
- → **Macro-F1 is primary**; accuracy shown only to expose the trap
- Also report per-class P/R/F1 + confusion matrices

**Speaker notes:** "This slide is the crux of the evaluation. A model that does *nothing* — always guessing the majority class — scores 66 to 75 percent accuracy. If we optimised for accuracy we'd reward exactly that useless behaviour. Macro-F1 weights all three classes equally, so ignoring Positive and Surprise is punished. Everything else in the talk is measured against macro-F1."

---

### SLIDE 7 — Why LOSO *and* Hold-out (Validation Strategy)

**Visual layout:** Two side-by-side diagrams. Left: hold-out (one split, 30 % subjects out). Right: LOSO (rotating held-out subject). Bottom: cost vs bias trade-off arrow.

**On-slide text:**
- Both are **strictly subject-disjoint** — zero identity leakage
- **Hold-out (30 % subjects, 52 clips):** fast, higher variance → iterate here
- **LOSO (pilot 20/25 folds, 139 clips):** slow, honest, tests each unseen subject
- Agreement between them = finding is *real*, not a split artefact

**Speaker notes:** "Naive random splits leak identity — the same person in train and test lets the net recognise the face, not the expression. Both our protocols hold out whole subjects. Hold-out is cheap so we iterate on it; LOSO is the gold standard but costs a full training run per subject, so we ran a 20-of-25 pilot. When both protocols tell the same story, we trust it."

---

### SLIDE 8 — The 30 % Question & the "Others" Cut (Data Slicing)

**Visual layout:** A stacked bar showing 255 raw clips → 156 kept (61 %) → 30 % test / 70 % train. Callout: "30 % = smallest test set that keeps all 3 classes measurable."

**On-slide text:**
- Held out **30 % of subjects** for hold-out test (`val_fraction = 0.3`)
- Why 30 %? smallest split that (a) stabilises 3-class F1 on ~52 clips, (b) keeps rare Surprise present on both sides
- Why not 100 %? train/test is zero-sum — 100 % test = **nothing to train on**
- Dropped **"Others" (99 clips, 39 %)** — a residual bucket, not an emotion → removing noise, not cherry-picking

**Speaker notes:** "Two data cuts a reviewer will challenge. First, 30 % held out for test — chosen because with only ~140 clips, a 10 % test set is ~15 clips and one misclassified Surprise swings the score by points; 30 % is the smallest split that keeps every class measurable. You obviously can't test on 100 % — there'd be no training data. Second, we drop the 'Others' class: it's CASME II's junk-drawer of helpless/pain/confused labels with no coherent facial target. Keeping it would train the model on pure noise. Removing it is the field-standard MEGC treatment."

---

### SLIDE 9 — Why Group the Emotions (Target Reframing)

**Visual layout:** Left: funnel diagram 7 raw emotions → 3 groups, with the tiny counts (fear = 2, sadness = 7) circled in red. Right: 3 reasons as numbered cards.

**On-slide text:**
- 7 raw → **3 classes**: Negative (99) · Positive (32) · Surprise (25)
- **① Sample size:** fear = **2 clips**, sadness = **7** → statistically unlearnable alone
- **② Cognitive ambiguity:** disgust/repression/sadness/fear overlap; coders themselves disagree
- **③ Comparability:** matches the MEGC composite standard

**Speaker notes:** "Why not predict all seven emotions? Look at the tail: fear has *two* clips, sadness has seven. Under leave-one-subject-out you can't even split a two-clip class. Beyond arithmetic, these negative states share facial signatures that even expert annotators confuse — asking the model to split them is chasing label noise. Grouping by valence into Negative/Positive/Surprise gives us a tractable, meaningful, and literature-comparable target."

---

### SLIDE 10 — Hyperparameters & the Imbalance Defences

**Visual layout:** Compact 2-column table (param | value) on the left; on the right a highlighted "triple defence" callout.

**On-slide text:**
- Loss: **Focal (γ=2)** + label smoothing 0.05 + **inverse-freq class weights**
- Sampler: **balanced oversampling** → *three stacked* imbalance defences
- AdamW · lr 1e-4 · wd 1e-4 · grad-clip 1.0 · AMP · CosineAnnealing
- **Batch size 2** — hardware-forced (`[3,32,224,224]` ≈ 5 GB VRAM)
- Best checkpoint chosen by **macro-F1**, not accuracy

**Speaker notes:** "On a 63/20/16 split, no single imbalance fix is enough, so we stack three: focal loss down-weights easy majority examples, class weights up-weight rare ones inside the loss, and a balanced sampler over-samples minorities into every batch. Batch size two isn't a scientific choice — it's what fits in VRAM for a 32-frame 224-square clip — so we pair it with a small learning rate and gradient clipping for stability. Crucially, we save the checkpoint with the best macro-F1, never the best accuracy."

---

### SLIDE 11 — Results: The Big Picture

**Visual layout:** Left half: LOSO `accuracy_macro_f1_bar.png`. Right half: hold-out `accuracy_macro_f1_bar.png`. Caption strip: "Blue = accuracy always > Orange = macro-F1."

**On-slide text:**
- Every config: **accuracy bar >> macro-F1 bar** → majority-class pull is universal
- Best **LOSO:** config 5 (SimAM+CNN) — 0.643 / **0.379**
- Best **hold-out:** config 13 (EVM+CNN) — 0.577 / **0.458**
- Both winners: **CNN on, Transformer off**

**Speaker notes:** "Here's the whole ablation in two charts. Notice the blue accuracy bar always dwarfs the orange macro-F1 bar — that's the imbalance signature we predicted. The winners under both protocols share a DNA: they have the CNN and they *don't* have the Transformer. That's the headline."

---

### SLIDE 12 — Finding: CNN Wins, Transformer Hurts

**Visual layout:** A matched-pairs table (Transformer OFF vs ON, Δ macro-F1) with all-red deltas. Small inset: `per_class_f1_grouped.png`.

**On-slide text:**
- **CNN is indispensable** — top-4 configs (both protocols) all have CNN✓, Transformer✗
- **Transformer subtracts macro-F1 in *every* matched pair** (Δ up to −0.28)
- Config 4 (EVM-only, no CNN): hold-out accuracy **0.096** — the floor
- Mechanism: 4-layer/8-head encoder = ~10⁵ params on ~140 clips → overfit

**Speaker notes:** "Two clean findings. One: remove the CNN and everything collapses — EVM-only bottoms out at nine percent accuracy. Two, and more surprising: the Transformer *hurts* in every single matched comparison, by up to 0.28 macro-F1. It's not noise — it's consistent across twelve configs and both protocols. The cause is simple: a four-layer transformer has a hundred-thousand parameters and we have a hundred-forty clips to constrain them."

---

### SLIDE 13 — Finding: More ≠ Better (the Honest Negative Result)

**Visual layout:** Left: `key_configs_confusion_side_by_side.png` (config 8 vs config 5/13). Right: rank callout.

**On-slide text:**
- The fully-loaded **"proposed" config 8** (all 4 on) ranks **10/12 (LOSO), 9/12 (hold-out)**
- Its confusion matrix **smears** across columns; config 5/13 show a clean diagonal
- Training logs: train-loss ↓ while val-F1 flat → textbook overfitting
- **Ablation's value = the diagnosis**, not a rubber stamp

**Speaker notes:** "This is the slide I'm proudest of, even though it's a negative result. Our own proposed all-in model is *not* the winner — it's near the bottom. The side-by-side confusion matrices show why: config 8 smears predictions across columns while the lean CNN configs keep a clean diagonal. A good ablation is supposed to be able to tell you your favourite idea doesn't work at this scale — and this one did."

---

### SLIDE 14 — Finding: EVM is Protocol-Dependent

**Visual layout:** Two mini-podiums: hold-out winner (config 13, EVM✓) vs LOSO winner (config 5, EVM✗). Arrow labelled "amplifies signal *and* noise."

**On-slide text:**
- Hold-out best **uses** EVM (config 13); LOSO best **omits** it (config 5)
- EVM amplifies **motion + noise**
- Fixed split → amplified signal helps; unseen subjects → amplified noise hurts
- A nuanced result, not a null one

**Speaker notes:** "EVM is the subtlest lever. It wins under hold-out but loses under LOSO. The reason is that magnification amplifies genuine muscle motion *and* subject-specific noise. On a fixed split the extra signal helps; when generalising to brand-new subjects, the amplified noise doesn't transfer and it hurts. So EVM's value is conditional — exactly the kind of insight an ablation is for."

---

### SLIDE 15 — Conclusions & Next Steps

**Visual layout:** Left: 4 takeaway checkmarks. Right: roadmap arrow (full LOSO → composite datasets → re-enable Transformer).

**On-slide text:**
- ✅ **3-D CNN spatial backbone is the load-bearing component**
- ✅ **Transformer capacity is mismatched to CASME II scale** (overfits)
- ✅ **Macro-F1 + confusion matrices** exposed majority-class collapse that accuracy hid
- ✅ Effective model = **CNN (+ SimAM / EVM)**, no Transformer *at this scale*
- ▶ Next: full 25-fold LOSO · composite (CASME+SAMM+SMIC) to feed the Transformer · re-test EVM cross-subject

**Speaker notes:** "To close: the CNN is the engine, the Transformer is dead weight at this data scale, and macro-F1 was essential to see the truth. The proposed architecture isn't wrong in principle — its temporal encoder is simply starved. The roadmap is to run the full LOSO for the final number, scale up to a composite dataset so the Transformer has data to learn from, and re-examine EVM's cross-subject behaviour. Thank you — happy to take the hard questions."

---

### BACKUP SLIDES (for Q&A)

**B1 — Full results table.** Both master tables from the report (LOSO + hold-out, all 12 configs, accuracy + macro-F1 + majority baseline).

**B2 — The 74 % myth.** "The earlier 74 % hold-out accuracy ≈ the 0.75 majority baseline — it was a majority-class parrot. Weekend runs with balanced training trade that fake accuracy for real macro-F1."

**B3 — Per-config confusion matrices.** The full `confusion_matrices.png` 3×4 panel for whichever protocol is questioned.

**B4 — Leakage defence.** "Motion-only input + whole-subject-disjoint splits at both feature and split level. `assert train ∩ val = ∅`."

**B5 — Pilot vs full LOSO.** "20/25 folds, flagged `loso_pilot:true`. Full LOSO reserved for the final dissertation number once config + epochs are frozen."

---

*Design tips: keep one figure per results slide at high resolution; use a consistent colour for "config 5/13 = good, lean" vs "config 8 = overloaded". Lead every results slide with the macro-F1 number, never accuracy.*
