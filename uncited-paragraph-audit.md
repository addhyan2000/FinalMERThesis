# Uncited-paragraph audit — *Improving Micro-Expression Recognition through Temporal and Motion Modeling*

Addhyan Pant, BTU Cottbus-Senftenberg. Audit generated 18 September 2026.

## How to read this

Every substantial paragraph (40 words or more) in the body of the thesis that contains no `[n]` citation marker is listed below, 
grouped by category and ordered by page. Page numbers are printed thesis pages, not PDF pages.

**No verbatim or near-verbatim overlap with any online source was found** in the web searches run against the highest-risk passages. 
Nothing in this file is an allegation of copying. What it flags is the separate problem of claims that belong to someone else 
appearing without attribution.

| Category | Count | Action |
|---|---|---|
| **B** — standard-concept explanation | 30 | Add a citation |
| **C** — another author's work or result | 29 | Add or repeat the citation |
| **A** — your own experiment or code | 132 | No action |
| **D** — your own argument or framing | 51 | No action |
| **Total** | 242 | |

The category is assigned by keyword heuristic and then by your judgement. A handful will be miscategorised; treat B and C as a review list, not a verdict.

---

## Category B — Standard-concept explanation with no citation

Correct in substance and in your own words, but the mechanism belongs to a published source. Add a reference.

### B.1 — p. 9, §2.4.2 Optical flow and brightness constancy

> This is one equation for two unknown components. The aperture problem means a local edge constrains motion normal to the edge, along the intensity gradient, but does not determine motion parallel to it. Estimators therefore add neighbourhood assumptions or regularisation. In the discrete pipeline, u and v denote displacement in pixels between selected frames rather than physical velocity per second. Illumination changes and large inter-frame displacement can violate the underlying approximation.

**Suggested reference:** Horn & Schunck (1981); Lucas & Kanade (1981)

- [ ] reviewed  - [ ] citation added

### B.2 — p. 10, §2.4.4 Assembling the three-channel tensor

> Spatial derivatives from NumPy’s gradient supply strain, with unit pixel spacing, central differences in the interior and first-order one-sided differences at boundaries. Temporal derivatives are not used in its construction, and no dedicated strain post-filter is applied. Stacking [u, v, strain] gives a tensor of shape (3, 32, 224, 224) over the resized full frame. The network receives this tensor rather than the original intensity sequence.

- [ ] reviewed  - [ ] citation added

### B.3 — p. 11, §2.5.2 Non-linearity

> Without non-linear operations, a stack of affine layers reduces to one affine map. Non-linearity allows the composition to represent more complex relationships. The convolutional streams use the rectified linear unit, ReLU(x) = max(0, x). The transformer feed-forward layers use the Gaussian error linear unit (GELU). These activations are fixed functions, although they act on features learned by the surrounding layers.

**Suggested reference:** Hendrycks & Gimpel (2016) for GELU; Nair & Hinton (2010) for ReLU

- [ ] reviewed  - [ ] citation added

### B.4 — p. 11, §2.5.3 Learning by gradient descent

> A loss compares predictions with labels. Backpropagation applies the chain rule through the network to compute derivatives of that loss with respect to its parameters. An optimiser uses these gradients to update the parameters; the learning rate controls the update scale. Mini-batches provide successive updates, while an epoch completes the configured sequence of batch draws. With replacement sampling, an epoch need not visit every distinct clip.

**Suggested reference:** Rumelhart, Hinton & Williams (1986)

- [ ] reviewed  - [ ] citation added

### B.5 — p. 11, §2.5.3 Learning by gradient descent

> Evaluation holds parameters fixed and disables training-only stochastic behaviour such as dropout. Low training loss alone does not establish useful learning: the relevant objective is prediction on unseen subjects. A validation set used to select an epoch has a different role from a test set reserved for final assessment.

- [ ] reviewed  - [ ] citation added

### B.6 — p. 11, §2.5.4 Convolution as a learned filter

> A convolution applies the same learned kernel at multiple positions. This weight sharing detects local patterns without learning a separate detector for every image location. Each output channel is a feature map; kernels combine all input channels over their local window. For a two-dimensional layer with one bias per output channel,

**Suggested reference:** LeCun et al. (1998)

- [ ] reviewed  - [ ] citation added

### B.7 — p. 12, §2.5.5 Pooling and spatial reduction

> Pooling replaces a neighbourhood with a fixed summary such as its maximum or mean. A 2 × 2 max-pool with stride two halves spatial dimensions, retaining the strongest response in each window. Adaptive average pooling instead targets a specified output grid. Neither learns weights. Both discard detail while reducing the computation of subsequent layers; a later learned projection remains a separate operation with its own parameters.

- [ ] reviewed  - [ ] citation added

### B.8 — p. 12, §2.5.6 Parameters, capacity and small data

> Capacity describes the range of functions a model can represent. Parameter count is one indicator, but architecture, constraints and training also matter. A model can overfit subject-specific appearance, noise or recording conditions, achieving low training loss without generalising. More parameters than examples do not by themselves prove overfitting, just as few parameters do not guarantee good predictions.

- [ ] reviewed  - [ ] citation added

### B.9 — p. 13, §2.6.1 Three-dimensional convolution and kernel shape

> A three-dimensional convolution accepts (B, C, T, H, W ) tensors: batch, channels, time, height and width. A kernel (kT , kH , kW ) mixes time only when kT > 1. The implemented streams use (1, 3, 3) kernels and (0, 1, 1) padding, followed by (1, 2, 2) max-pooling. Thus the convolutions preserve the temporal length and learn spatial filters; pooling halves height and width.

- [ ] reviewed  - [ ] citation added

### B.10 — p. 13, §2.6.2 Normalisation and regularisation

> Batch normalisation rescales each channel using a mean and variance, followed by learned scale and shift: x − µc x b= p , y = γc x b + βc . σc2 + ϵ BatchNorm3d computes training statistics over (B, T, H, W ) for each channel; evaluation uses running estimates from training. These activation statistics differ from the per-sample input standardisation performed by the dataset loader.

**Suggested reference:** Ioffe & Szegedy (2015)

- [ ] reviewed  - [ ] citation added

### B.11 — p. 13, §2.6.2 Normalisation and regularisation

> Dropout3d randomly removes entire feature channels for a sample during training, rather than masking individual spatial values. Retained channels are rescaled; evaluation disables the random mask. Layer normalisation instead standardises a token’s feature dimension independently of other samples. It appears inside the transformer and in the classifier head, followed there by dropout and a linear projection to class logits.

**Suggested reference:** Srivastava et al. (2014); Tompson et al. (2015) for channel dropout; Ba et al. (2016) for LayerNorm

- [ ] reviewed  - [ ] citation added

### B.12 — p. 14, §2.6.4 Self-attention

> Each input token is projected into a query, key and value. Scaled dot-product attention computes QK T   Attention(Q, K, V ) = softmax √ V. dk Queries and keys determine which values contribute to each output. Scaling moderates dot-product magnitude as key dimension dk grows. Multiple heads learn projections into different subspaces, whose outputs are concatenated and linearly mixed.

**Suggested reference:** Vaswani et al. (2017)

- [ ] reviewed  - [ ] citation added

### B.13 — p. 14, §2.6.4 Self-attention

> Without position information, self-attention is permutation-equivariant: permuting input tokens permutes their outputs correspondingly. It does not identify their temporal order. Fixed sinusoidal position vectors provide that information here. Although the sinusoidal functions can be evaluated at other positions, successful generalisation to unseen sequence lengths is not guaranteed; the implementation uses a finite precomputed buffer.

**Suggested reference:** Vaswani et al. (2017)

- [ ] reviewed  - [ ] citation added

### B.14 — p. 14, §2.6.4 Self-attention

> The temporal encoder alternates attention and position-wise feed-forward sublayers with residual additions. It uses pre-norm placement, applying LayerNorm before each sublayer. Its output is averaged over time before classification. This mean summarises already contex- tualised temporal features and is therefore different from averaging the input tokens without an encoder.

**Suggested reference:** Xiong et al. (2020) on pre-LN transformers

- [ ] reviewed  - [ ] citation added

### B.15 — p. 15, §2.6.5 The component controls

> Pooling and direct temporal averaging have no learned weights. The no-CNN projection does: with its bias it contains 48 × 96 + 96 = 4,704 parameters. The controls preserve the interfaces needed for the experiment while changing the operations and capacity. They do not isolate architecture independently of parameter count or computational cost.

- [ ] reviewed  - [ ] citation added

### B.16 — p. 15, §2.7.1 What skew does to cross-entropy

> Cross-entropy contributes − log pt for an example whose true-class probability is pt . Under natural sampling, a frequent class supplies more terms to the training objective. This can favour majority-class performance, although total gradient magnitude also depends on individual errors and features, not class counts alone. Lower overall loss need not mean that rare classes are handled well.

- [ ] reviewed  - [ ] citation added

### B.17 — p. 16, §2.7.3 Balanced sampling

> The training loader assigns an example from class c weight 1/nc , using counts from that fold’s training split only. Drawing with replacement makes each represented class equally likely in expectation, while individual batches can remain uneven. Each epoch draws as many indices as there are training clips. This changes exposure, not the number of distinct recorded expressions.

- [ ] reviewed  - [ ] citation added

### B.18 — p. 16, §2.7.4 Combining frequency corrections

> Inverse-frequency loss weights and balanced sampling both change the relative influence of classes. Combining them can emphasise rare classes beyond the effect of either alone. The study assigns frequency correction to the sampler: when balanced sampling is active, the runtime guard disables loss-side class weights. Focal difficulty weighting remains active. This is a specific training choice, not a general proof that data-level and loss-level interventions should never be combined.

- [ ] reviewed  - [ ] citation added

### B.19 — p. 17, §2.7.5 Label smoothing

> The implementation uses ϵ = 0.05 before focal modulation. This discourages extreme confidence rather than preferentially weighting a rare class. The focal factor still uses the unsmoothed true-class probability. Neither smoothing strength nor focal exponent is varied in the architectural comparisons.

**Suggested reference:** Lin et al. (2017)

- [ ] reviewed  - [ ] citation added

### B.20 — p. 17, §2.7.6 Augmentation under scarcity

> The active training augmentation is a horizontal flip with probability one half, accompanied by negation of the horizontal-flow channel u. A mirror reverses horizontal displacement, so mirroring alone would give inconsistent motion direction. Subsequent per-channel stand- ardisation preserves the intended sign reversal in the normalised representation. Validation receives neither the random flip nor a random temporal crop.

- [ ] reviewed  - [ ] citation added

### B.21 — p. 18, §2.8.1 Confusion counts, precision, recall and F1

> In a confusion matrix, entry (i, j) counts examples whose true class is i and predicted class is j. The diagonal contains correct predictions. For class c, true positives T Pc count correct predictions of c, false positives F Pc count incorrect predictions of c, and false negatives F Nc count missed examples of c.

**Suggested reference:** a standard reference for macro/micro averaging (e.g. Sokolova & Lapalme 2009)

- [ ] reviewed  - [ ] citation added

### B.22 — p. 18, §2.8.1 Confusion counts, precision, recall and F1

> Precision penalises false alarms and recall penalises misses. F1 is their harmonic mean; a high value requires both to be reasonably high. Degenerate denominators require an explicit convention; this implementation assigns zero rather than omitting the class from the score.

**Suggested reference:** a standard reference for macro/micro averaging (e.g. Sokolova & Lapalme 2009)

- [ ] reviewed  - [ ] citation added

### B.23 — p. 19, §2.8.2 Macro versus micro averaging

> Micro F1 first sums true positives, false positives and false negatives across classes. In single- label multiclass classification, each error supplies one false positive and one false negative, so micro F1 equals accuracy. It can therefore be dominated by a frequent class. Macro F1 and accuracy describe different aspects of performance; reporting both does not remove the need to inspect per-class errors.

**Suggested reference:** a standard reference for macro/micro averaging (e.g. Sokolova & Lapalme 2009)

- [ ] reviewed  - [ ] citation added

### B.24 — p. 19, §2.8.3 Pooled versus per-fold averaging

> F1 is non-linear, so the two procedures need not agree. If a held-out subject has no examples of a class, that fold’s score for the absent class is zero under the fixed-class convention even when every available example is classified correctly. Averaging these fold scores mixes classifier behaviour with fold composition. Pooled scores instead use all held-out predictions together; pooled accuracy is also distinct from an unweighted mean of subject accuracies when subjects contribute different numbers of clips.

- [ ] reviewed  - [ ] citation added

### B.25 — p. 19, §2.8.4 Cross-validation and subject-disjointness

> Cross-validation repeatedly fits a model to a training partition and evaluates it on held-out examples. Leave-one-subject-out (LOSO) assigns all clips from one subject to a fold’s held-out set and trains on the other subjects. This prevents that subject’s clips appearing in both partitions and reduces the opportunity to exploit familiar identity or recording conditions.

**Suggested reference:** the MEGC protocol paper (See et al. 2019) or a CV reference

- [ ] reviewed  - [ ] citation added

### B.26 — p. 19, §2.8.4 Cross-validation and subject-disjointness

> The study covers all 25 usable subjects. Subject separation does not, however, make the final score independent of model selection: choosing an epoch using the held-out subject’s labels also uses information from that subject, even though those labels never enter a gradient update.

- [ ] reviewed  - [ ] citation added

### B.27 — p. 20, §2.9 Conclusion

> The pipeline converts brief facial movement into analytical motion and deformation features, then evaluates learned spatial and temporal processing on those features. Its interpretation depends on the actual preprocessing order, spatial-only convolutional kernels, explicit com- ponent controls and fixed training regime. Pooled metrics and subject-disjoint evaluation address class skew and identity overlap, while checkpoint selection and limited retained predictions still constrain the strength of the conclusions. These mechanisms establish what the architectural experiment can measure without assuming that every named technique is a reproduction of its published form.

- [ ] reviewed  - [ ] citation added

### B.28 — p. 22, §2.10.1 The sequence in words

> 33 evenly spaced frames between the annotated onset and offset. Magnification is applied at this point if enabled. Optical flow is then computed for the 32 adjacent frame pairs, giving horizontal and vertical displacement fields; optical strain is derived from their spatial gradients. The three are stacked and normalised into one saved tensor.

**Suggested reference:** Shreve et al. (2011)

- [ ] reviewed  - [ ] citation added

### B.29 — p. 22, §2.10.1 The sequence in words

> Stage 2 turns motion into a score. A saved tensor is loaded, optionally flipped as augmentation, and standardised per channel. It passes through a convolutional stem, an attention gate, a pooling step that collapses space and leaves time, a temporal encoder, and a three-way classifier head. Training uses balanced sampling and a focal objective with label smoothing. Scoring is leave-one-subject-out over all 25 subjects, with confusion counts accumulated across folds before any per-class score is computed.

**Suggested reference:** Lin et al. (2017)

- [ ] reviewed  - [ ] citation added

### B.30 — p. 22, §2.10.2 What the data is at each step

> After this step The data is Shape Corpus selection a set of clips 156 clips, 25 subjects Frame preparation greyscale frames N × 224 × 224 Uniform sampling a fixed-length sequence 33 × 224 × 224 Magnification the same, amplified 33 × 224 × 224 Flow and strain three motion fields per pair 3 × 32 × 224 × 224 Pooling to a sequence a sequence of feature vectors 32 × 96 Temporal encoding one vector per clip 96 Classifier head class scores 3 Fold aggregation pooled confusion counts 3×3

- [ ] reviewed  - [ ] citation added

---

## Category C — Describes another author's work, dataset or result with no bracketed citation

Often the author is named in prose ("Yang et al. ...") or the citation sits in an adjacent paragraph. Add or repeat the marker so the claim is attributable on its own page.

### C.1 — p. 6, §2.2.4 Which clips enter the pipeline

> Preprocessing selects CASME II rows coded as micro-expressions, marked as having frames, and with an inclusive onset-to-offset length greater than two frames. The loader must still find readable frames; the metadata flag alone does not guarantee successful extraction. The recognition dataset also requires a matching saved tensor and a label in the selected three-class mapping; the residual “others” category is excluded. Preprocessing eligibility and final classification membership are therefore separate filters.

- [ ] reviewed  - [ ] citation added

### C.2 — p. 12, §2.5.6 Parameters, capacity and small data

> The small CASME II training pool motivates controlled architectural comparisons and regularisation. Their effectiveness must be assessed on held-out subjects, while keeping checkpoint selection separate from final testing where possible. The analytical flow and strain transformations already expose motion structure, so the learned network is not starting from raw pixels.

- [ ] reviewed  - [ ] citation added

### C.3 — p. 13, §2.6.1 Three-dimensional convolution and kernel shape

> The use of Conv3d does not establish that temporal convolution was tested. The complete stream can nevertheless depend on multiple frames: BatchNorm during training and SimAM whenever enabled use statistics spanning time. This distinction separates temporal convolution from other forms of whole-clip dependence.

- [ ] reviewed  - [ ] citation added

### C.4 — p. 14, §2.6.3 Parameter-free attention: SimAM

> Here M = T HW within each channel, rather than the original image formulation’s HW , and λ = 10−4 . SimAM acts on each convolutional stream before concatenation. It adds no learned parameters, but still performs reductions and element-wise computations. Whole-clip statistics do not guarantee that apex frames receive the highest weights: unusual activation values may also reflect noise.

- [ ] reviewed  - [ ] citation added

### C.5 — p. 15, §2.6.5 The component controls

> Component Enabled Disabled Spatial stem Three learned convolutional streams Per-frame 4 × 4 average pooling and a learned 48 → 96 linear projection SimAM Per-stream activation re-weighting Unmodified stream outputs; this model enables SimAM only with its CNN stem Temporal encoder Position encoding, transformer lay- Direct mean of the input sequence ers, then temporal mean

- [ ] reviewed  - [ ] citation added

### C.6 — p. 16, §2.7.2 Focal loss: weighting by difficulty

> The implementation computes pt from the hard target label using log-softmax. It then multiplies a label-smoothed cross-entropy term by (1 − pt )γ , followed by the optional class weight. The experiment fixes γ = 2, matching Zhao et al.’s reported exponent; this is an inherited setting rather than a measured optimum for the present corpus.

**Suggested reference:** Szegedy et al. (2016)

- [ ] reviewed  - [ ] citation added

### C.7 — p. 21, §2.10.1 The sequence in words

> Stage 1 turns video into motion. Clips are drawn from CASME II, kept if they are coded as micro-expressions and carry a label in the three-class mapping, and reduced to 156 clips across 25 subjects. Each is loaded in greyscale, resized to 224 × 224, and sampled at

- [ ] reviewed  - [ ] citation added

### C.8 — p. 22, §2.10.3 Where the experiment intervenes

> Four of the boxes in Figure 2.1 are dashed because they are the subject of this thesis rather than fixed infrastructure: motion magnification, the convolutional stem, SimAM and the temporal transformer. Each can be switched off independently. The component-controls table earlier in this chapter states each replacement. Everything else in the diagram is held identical across every configuration and every fold.

- [ ] reviewed  - [ ] citation added

### C.9 — p. 25, §3.1.2 CASME II: construction and annotation

> Two specifics propagate into this thesis. The original release applies a normalisation from which this work departs: 68 Active Shape Model landmarks are detected on the first frame only and the resulting transform applied unchanged to every subsequent frame; this pipeline reads the original recorded frames instead, unregistered and uncropped. And the published paper reports 247 micro-expressions while the distributed coding file contains 255, a post- publication revision rarely acknowledged and one reason sample counts for “CASME II” vary. The present work uses the released 255-sample coding.

- [ ] reviewed  - [ ] citation added

### C.10 — p. 27, §3.1.5 Sibling corpora, and why this study remains single-corpus

> CASME II is used alone because it has the highest temporal resolution of any spontaneous corpus of comparable size, a precondition for magnification in a defined band and for dense frame-to-frame flow; because its onset, apex and offset annotation makes temporal normalisation well-defined rather than heuristic; and because it remains the most widely

- [ ] reviewed  - [ ] citation added

### C.11 — p. 30, §3.1.7 Threats to validity carried by the corpus

> Duration, annotation and version ambiguity. Onset-to-offset lengths range from 31 to 126 frames, median 66 - at 200 fps, 0.155 s to 0.63 s. So every clip is resampled to a fixed length before training. Labels derive from coded action units, self-report and stimulus content: a construct produced by trained coders under a rubric, taken here as given and recorded as a declared limitation rather than as measurement error. And three sample counts circulate under the name “CASME II” - 247, 255 and 145. With both five- and seven-class label sets in use and rarely stated, so N is reported explicitly alongside every figure here.

- [ ] reviewed  - [ ] citation added

### C.12 — p. 36, §3.3.1 Implications for this research

> Two consequences are untested extensions rather than gaps claimed as closed. Flow-based fine alignment, which the estimation discussion above names the most promising unexploited improvement to preprocessing, is not performed here. And Farnebäck against TV-L1 is fixed rather than compared, so no estimator ranking is established. The design likewise does not compare onset–apex against sequence representations.

- [ ] reviewed  - [ ] citation added

### C.13 — p. 40, §3.5.1 Implications for this research

> The length is fixed at 33 frames sampled, giving 32 flow pairs, bounded by the annotated onset and offset so the window spans the expression rather than surrounding neutral frames. That is close to the lower of Ben et al.’s two reported optima, but the correspondence should not be overstated: their 30 frames are interpolated frames under a hand-crafted descriptor on a different corpus, which is not direct validation of 32 sampled flow fields under a learned model here. The length is not swept, and a sweep is declared as further work.

- [ ] reviewed  - [ ] citation added

### C.14 — p. 41, §3.5.1 Implications for this research

> The unresolved question this leaves is what temporal normalisation costs a representation made of motion rather than appearance. Three studies sweep interpolation length and give three answers, only one of them touching CASME II and none using a learned temporal model. This thesis takes the position that follows from its own representation and quantifies what that costs in effective frame rate, but does not sweep the length, compare interpolators, or separate the effect of sequence length from that of the sampling regime.

- [ ] reviewed  - [ ] citation added

### C.15 — p. 42, §3.6.1 Implications for this research

> The component ablated here is a three-stream shallow convolutional backbone, one unshared two-layer stream per input channel, carrying 14,544 parameters - shallow by the corpus’s standards, though roughly nine times STSTNet’s. Where STSTNet’s streams differ in filter count over the same input, these differ in input channel. Removing it leaves not an empty slot but spatial pooling followed by a learned linear projection, so the no-backbone route still contains learned input processing.

- [ ] reviewed  - [ ] citation added

### C.16 — p. 43, §3.6.1 Implications for this research

> A declaration about what this component is. Every kernel in the backbone is spatial only: no filter spans two time steps, so despite the Conv3d implementation it is a purely spatial feature extractor (subsection 2.6.1). The ablation therefore does not test spatio- temporal convolution; it tests a learned spatial stem, and whatever it measures is evidence about this stem - spatial-only, at 224 × 224, on flow and strain input and not evidence that convolution over time is unhelpful for micro-expression recognition, which this study never evaluated. In the field’s own terms the choice is defensible: STSTNet, one of the strongest shallow baselines on the CASME II subset, likewise performs no temporal convolution.

- [ ] reviewed  - [ ] citation added

### C.17 — p. 44, §3.7 Parameter-Free Attention: SimAM

> Yang et al. evaluate SimAM with several network families on CIFAR and ImageNet, reporting competitive performance against learned attention modules, and their timing results show that unchanged parameter totals do not imply unchanged inference speed. For CIFAR they sweep λ from 10−1 to 10−6 and select 10−4 ; they repeat the selection for ImageNet and choose 0.1. Thus 10−4 is a dataset-selected operating point, not a universal setting justified by the formula.

- [ ] reviewed  - [ ] citation added

### C.18 — p. 44, §3.7.1 Implications for this research

> SimAM is applied separately to each convolutional stream before concatenation, introducing no trainable weights or biases anywhere. Two choices diverge from the source. Mean and variance span (T, H, W ) within each channel with divisor T HW − 1, rather than one (H, W ) plane, so an activation is judged distinctive relative to its whole clip rather than its own frame; this could emphasise a brief contraction, but does not guarantee low weights at onset or high weights at the apex, since the score depends on squared deviation and not on expression labels. And λ = 10−4 is retained without re-tuning on CASME II. Neither context choice is compared experimentally.

- [ ] reviewed  - [ ] citation added

### C.19 — p. 46, §3.8 The Temporal Encoder: Transformers

> Published SLSTT combines three stages. First, onset-referenced optical flow describes sampled frames relative to the onset rather than to their immediate predecessors, which Zhang et al. argue gives a more structured trajectory towards and away from the apex; their experiments use it throughout, without a matched short-term-flow ablation establishing its independent gain. Second, a spatial ViT encoder processes each flow field as patches, using ImageNet-pre-trained ViT-B/16 with twelve layers at 384 × 384. Third, an LSTM temporal aggregator combines the per-frame features, with a mean aggregator supplying the comparison variant. Self-attention is spatial in this architecture; recurrence handles time.

- [ ] reviewed  - [ ] citation added

### C.20 — p. 47, §3.8.2 Implementation and six declared divergences

> Six matched pairs evaluate the encoder with other component choices fixed within each pair. They test the whole encoder: positional encoding, feed-forward layers and normal- isation included, against direct mean pooling, and do not isolate self-attention from those accompanying operations or compare it against another learned temporal model. Three follow-ups follow from Table 3.4: testing an LSTM aggregator, comparing onset-referenced with consecutive-frame flow, and replacing fixed positions with learned embeddings. All three remain untested, and changing the input representation requires re-training before any benefit can be claimed. What this study can establish is a matched comparison of the implemented encoder against an order-blind baseline within this pipeline; it neither validates published SLSTT nor establishes a general ranking of attention, recurrence and convolution.

- [ ] reviewed  - [ ] citation added

### C.21 — p. 48, §3.9 Training Under Severe Class Imbalance

> what matters: γ re-weights by difficulty and is treated as a constant, “set as 2 in practice”, while α re-weights by class frequency and is “treated as a hyper-parameter to set by cross validation”. Conflating the two is the error discussed below. Zhao et al. compare focal loss against cross-entropy across a sweep of α, but hold γ fixed, so the focusing exponent remains unvalidated on this task.

**Suggested reference:** Lin et al. (2017)

- [ ] reviewed  - [ ] citation added

### C.22 — p. 48, §3.9.1 Implications for this research

> Focal loss is adopted with γ = 2.0, matching Zhao et al.’s reported setting. The class-frequency term is implemented as an optional per-class α vector but switched off at run time, because frequency correction is assigned to a weighted sampler active in every configuration. Label smoothing of 0.05 is applied inside the focal loss, declared as a choice this corpus supports

**Suggested reference:** Lin et al. (2017)

- [ ] reviewed  - [ ] citation added

### C.23 — p. 50, §3.10 Synthesis and Research Gap

> Read as a whole, the reviewed literature settles a good deal. The corpora are small, demo- graphically narrow, laboratory-bound and irreducibly imbalanced by what can be elicited, and CASME II remains the primary single-corpus benchmark under its native leave-one- subject-out protocol. Motion rather than appearance is the appropriate input; a strain field adds a further cue, though on CASME II only weakly and under a protocol that is not subject-disjoint; and both the magnification factor and the sequence length have interior optima located only by sweep. At this data size, capability bought without parameters outper- forms capability bought with them, and self-attention is the strongest-performing mechanism reported. Accuracy alone is inadequate under skew and must give way to macro-averaged measures accumulated across folds before averaging.

- [ ] reviewed  - [ ] citation added

### C.24 — p. 51, §3.10.1 Four limitations that recur across every topic

> (c) The strongest results depend on regimes a single-corpus study cannot access. The best CASME II figures come from composite-database training (subsection 3.1.6) and, for SLSTT, ImageNet pre-training, neither a property of the architecture being credited. Almost no evidence exists on what these components do when trained from scratch in a small corpus, which is the situation most practitioners are in.

- [ ] reviewed  - [ ] citation added

### C.25 — p. 51, §3.10.1 Four limitations that recur across every topic

> (d) Preprocessing parameters have interior optima and are transferred without re-validation. The magnification factor, the sequence length and the input resolution all rise then fall, and none has a principled selection rule. Values are carried between pipelines whose downstream models differ entirely: α = 30 moves from a single-frame appearance encoder to other settings, ten-frame interpolation propagates from one baseline paper to the whole field, and λ = 10−4 transfers from CIFAR though its own authors re-tuned it for ImageNet.

- [ ] reviewed  - [ ] citation added

### C.26 — p. 52, §3.10.3 How this thesis addresses the gap, and what it does not

> What it does. This thesis treats the pipeline as a factorial experiment rather than a proposal: four components varied independently across every architecturally valid cell of a 24 matrix, twelve configurations, each evaluated under complete 25-fold leave-one-subject-out on CASME II with every non-varied factor held identical. That supplies the joint measurement (a) identifies as missing, reporting the protocol in full addresses (b) for this study at least, and training from scratch is the regime (c) identifies as unevidenced. Each component’s marginal contribution is read from pairs of runs differing in one factor alone, and the distribution of those four effects is the substantive answer to the gap. Two further contributions follow from the review rather than the experiment: a repaired measurement, where an early data-routing defect left the magnification switch inert and the review’s evidence of a clear, repeatable magnification effect is what made bit-identical results detectable as a defect rather than a null (subsection 3.2.3); and two opposite class-collapse failures that no reviewed paper reports (subsection 3.9.1).

- [ ] reviewed  - [ ] citation added

### C.27 — p. 52, §3.10.3 How this thesis addresses the gap, and what it does not

> What it does not do. Three of the four limitations are only partially addressed. On (b), this study reports its own protocol completely but cannot repair the field’s inconsistency, and its own N = 156 differs from MEGC’s 145. On (c), it establishes what these components do from scratch on one corpus, which is a different question from, not an answer to, how they behave under composite training. On (d), the thesis inherits parameter values rather than sweeping them: α = 10, T = 32, λ = 10−4 and γ = 2.0 are fixed throughout, so the ablation measures architecture at one point in preprocessing space rather than across it.

- [ ] reviewed  - [ ] citation added

### C.28 — p. 53, §3.10.4 A consolidated ledger of declared divergences

> not a verdict on convolution over time or on SLSTT. The limits of what can be claimed are therefore fixed before any number is reported, a small, single-seed, demographically narrow corpus measured against richer published regimes, with these divergences standing throughout.

- [ ] reviewed  - [ ] citation added

### C.29 — p. 54, §3.10.4 A consolidated ledger of declared divergences

> § Divergence Consequence 3.1.3 Working set is N = 156, including sadness Comparisons against challenge figures are not and fear in Negative; MEGC’s CASME II exactly like-for-like subset is N = 145 3.1.2 The corpus’s own registered release is not No landmarking or alignment of any kind pre- used; the pipeline reads the original recor- cedes the network; tolerance to pose and scale ded frames, unregistered and uncropped must be learned from 156 clips 3.2.3 Temporal band is the unnarrowed 5–25 Hz The pass-band admits the low-frequency range duration rule, where Bai et al. narrowed it they rejected to 15–25 Hz as noisy 3.2.3 Band-pass is realised by an FFT coefficient A different filter shape, with different transition mask, not the Butterworth filter of the re- behaviour at the band edges viewed description √ 3.4.1 Strain magnitude counts shear once, not Shear weighted lower by 2 relative to normal twice as in the published Frobenius norm strain; untested 3.4.3 No dedicated strain filtering, where the lit- Spatial differentiation can amplify flow- erature applies Wiener and Gaussian filters estimation noise, and nothing downstream sup- presses it 3.5.1 Temporal normalisation is uniform index Avoids fabricating motion; forfeits the ability sampling, not manifold-based TIM; no to lengthen short clips frames are synthesised 3.5.1 Magnification is applied after subsampling, Realised pass-band is clip-dependent and below at a nominal 200 fps the intended 5–25 Hz 3.6.1 Backbone kernels are spatial only - no tem- The ablation tests a learned spatial stem, not poral mixing spatio-temporal convolution 3.6.1 Backbone input is 224 × 224, against Xia Whatever the stem is measured to contribute et al.’s guidance of ≤ 100 × 100 - though may be a resolution effect as much as an archi- Bai et al. and Li, Huang and Zhao both tecture effect use 224 × 224 3.7.1 SimAM statistics pooled over the whole clip Neurons judged against the clip, not the frame; rather than per frame as in the source; λ defensible but untested not re-tuned 3.8.2 Six divergences from published SLSTT The temporal encoder is not a reproduction of (temporal not spatial attention, mean not SLSTT LSTM aggregation, from scratch, smaller, sinusoidal encoding, short-term flow) 3.9.1 Label smoothing of 0.05, named in the cor- A choice the micro-expression corpus supports pus only by Dosovitskiy et al. and never neither way, held constant and unmeasured examined on this task

**Suggested reference:** Szegedy et al. (2016)

- [ ] reviewed  - [ ] citation added

---

## Category A — Describes your own experiment, code, configuration or results

No citation expected. Listed only so the audit is complete.

| Page | Section | Opening words |
|---|---|---|
| 1 | 1 Introduction | The systems built to attempt it are not single models. They are pipelines, assembled stage by stage from techn… |
| 2 | 1.1 Research Motivation | That arrangement is manageable while a pipeline is being published and awkward as soon as one is being built. … |
| 2 | 1.1 Research Motivation | The study is therefore designed as a factorial ablation. Four main components: Eulerian mag- nification, a con… |
| 3 | 1.1 Research Motivation | What that design yields is a set of matched pairs. Any two of the twelve configurations that differ in exactly… |
| 3 | 1.2 Structure of the Thesis | • Chapter 2 - Background sets out the mechanisms the thesis depends on, in the order the pipeline applies them… |
| 3 | 1.2 Structure of the Thesis | • Chapter 4 - Methodology specifies the study as carried out: the experimental design, the corpus and its prep… |
| 6 | 2.2.3 Frame selection and preparation | Frames are loaded in greyscale and resized by bilinear interpolation to 224 × 224. Resizing the whole rectangu… |
| 14 | 2.6.3 Parameter-free attention: SimAM | (xi − µ)2 1 Ei−1 = + , ei = xi sigmoid(Ei−1 ). x 4(v + λ) 2 The reciprocal energy measures distinctiveness; th… |
| 20 | 2.8.5 What the evaluation apparatus must retain | Third, aggregate metrics do not preserve which clips two models disagree on. Per-clip predictions are needed f… |
| 21 | 2.10 The Pipeline End to End | Figure 2.1: The pipeline from corpus to score. Stage 1 runs once per clip; Stage 2 runs once per configuration… |
| 24 | 3 Literature Review | This chapter reviews the literature bearing on each component of the recognition pipeline evaluated in this th… |
| 26 | 3.1.3 The label taxonomy and the sparsity it creates | Seven-class classification is not viable on this corpus. Fear is represented by two clips in the entire databa… |
| 27 | 3.1.4 The original baseline, and what it does not establish | Accuracy also moves by several points across a purely descriptive radius sweep, with no change to data, split … |
| 28 | 3.1.6 How CASME II is evaluated: MEGC 2019 and the metric convention | What matters is not the algebra but the construction: UF1 accumulates true positives, false positives and fals… |
| 29 | 3.1.7 Threats to validity carried by the corpus | Subject-level non-uniformity. Subject 17 alone contributes 33 clips, 21 % of the working set, while three subj… |
| 30 | 3.1.7 Threats to validity carried by the corpus | which subjects fall in the test set can move the result by more than any architectural change under study. Sub… |
| 30 | 3.1.8 Implications for this research | Two further properties determine settings rather than threatening validity: the 200 fps capture rate fixes the… |
| 34 | 3.2.3 Implications for this research | intensity change by α does not simply scale the recovered displacement by α. It changes the conditioning of th… |
| 34 | 3.2.3 Implications for this research | What the literature leaves open is threefold: every evaluation places magnification in front of an appearance … |
| 34 | 3.3 Motion Representation: Dense Optical Flow | Dense optical flow supplies the horizontal and vertical displacement channels. The relevant literature concern… |
| 36 | 3.3.1 Implications for this research | A temporal encoder addresses a different task: modelling relations among successive displace- ment fields, whi… |
| 36 | 3.4 Deformation Representation: The Optical Strain Tensor | Optical strain supplies the third input channel. Its literature establishes a deformation descriptor derived f… |
| 38 | 3.4.3 Implications for this research | Strain extends the analytical-feature argument of subsection 3.3.1: the input contains a hand-specified differ… |
| 40 | 3.5.1 Implications for this research | A declaration about what this pipeline implements. The module performing this step is named for the Temporal I… |
| 41 | 3.5.1 Implications for this research | Magnification is applied after subsampling, which shifts the realised pass-band. Magnification runs on the 33 … |
| 44 | 3.7 Parameter-Free Attention: SimAM | deformation, noise or an artefact. So its suitability for flow-derived features requires evaluation rather tha… |
| 47 | 3.9 Training Under Severe Class Imbalance | The 99 : 32 : 25 distribution of subsection 3.1.3 is a property of what can be elicited in a laboratory, not a… |
| 48 | 3.9 Training Under Severe Class Imbalance | Two techniques belonging to a fuller account of this pipeline are excluded from the imple- mented system and r… |
| 49 | 3.9.1 Implications for this research | This is worth reporting because no reviewed paper warns that combining corrections can invert the intended eff… |
| 49 | 3.9.1 Implications for this research | A second declaration follows. Because every configuration shares the same sampler, loss and γ, this thesis mea… |
| 49 | 3.9.2 Relevance and scope | The corrections themselves are well established. What the corpus does not settle is how two corrections compos… |
| 51 | 3.10.2 The research gap | This matters practically as well as scientifically. A practitioner assembling this pipeline today has no publi… |
| 52 | 3.10.4 A consolidated ledger of declared divergences | Reviewing the literature closely enough to write this chapter surfaced twelve points at which the implemented … |
| 52 | 3.10.4 A consolidated ledger of declared divergences | Three of these: the strain formula, the magnification ordering and the input resolution, were identified by th… |
| 55 | 4 Methodology | This chapter specifies the study as it was carried out: the experimental design, the corpus and its preparatio… |
| 56 | 4.1.2 The four variables | Each flag is a strict on/off switch, with no intermediate setting and no equivalent implement- ation substitut… |
| 56 | 4.1.3 Twelve configurations, not sixteen | Not every generated cell is trained. Each candidate passes through AblationConfig.is_valid(), which rejects an… |
| 57 | 4.1.5 A stated limitation | Every configuration is trained exactly once, under a single fixed random seed (42); none is retrained under mu… |
| 58 | 4.2.2 From 255 rows to 156 clips | The pipeline’s label table, Processed Data/master thesis labels.csv, holds 255 rows, all CASME II and all code… |
| 58 | 4.2.3 The three-class grouping | The grouping is the field convention reviewed in subsection 3.1.3, adopted rather than re- argued, and stated … |
| 59 | 4.2.4 Subjects and the leave-one-subject-out protocol | The distinction matters directly to the protocol. Leave-one-subject-out cross-validation (section 2.8) is run … |
| 59 | 4.2.4 Subjects and the leave-one-subject-out protocol | Fold sizes are uneven by construction, since each held-out set is one subject’s clips and CASME II subjects do… |
| 60 | 4.3.1 The twelve trained configurations | name evm simam cnn transformer descriptor config 1 pure base F F F F no component active; raw motion tensor po… |
| 60 | 4.3.2 The four excluded cells | Four of the sixteen generated cells are rejected by AblationConfig.is_valid() and never trained, all sharing u… |
| 62 | 4.4.2 Input and the three streams | The model receives the tensor of subsection 4.2.5 with a batch dimension prepended, giving [B, 3, 32, 224, 224… |
| 63 | 4.4.3 The convolutional stem | No kernel in this stem spans the temporal axis, for the reason subsection 2.6.1 sets out, so the sequence leng… |
| 65 | 4.4.9 Parameter counts | Four components combine to give any configuration’s total. Their individual counts are: the convolutional back… |
| 65 | 4.4.9 Parameter counts | configuration type example parameters no CNN, no transformer config 1, config 4 5,187 CNN, no transformer conf… |
| 66 | 4.5 Training Procedure | Every one of the twelve valid configurations is trained by the same procedure - the same loss, sampler, optimi… |
| 66 | 4.5.2 The class-weighting rule | ExperimentConfig exposes an optional per-class α weight vector for the loss, and use class weights defaults to… |
| 66 | 4.5.2 The class-weighting rule | use balanced sampler also defaults to True (subsection 4.5.3) and no reported run overrides either default, so… |
| 66 | 4.5.2 The class-weighting rule | The rule is not an oversight. The α term and the balanced sampler correct for the same skew at different point… |
| 67 | 4.5.3 Sampler | The training loader for every fold and configuration uses a WeightedRandomSampler. Each training example is as… |
| 67 | 4.5.4 Optimiser and learning-rate schedule | Every configuration is optimised with AdamW, learning rate 1 × 10−4 and weight decay 1 × 10−4 , applied unifor… |
| 67 | 4.5.5 Batch size | ExperimentConfig declares batch size: int = 2, annotated in the source as deliber- ately small because transfo… |
| 67 | 4.5.5 Batch size | The mechanism matters to anyone reconstructing the command from source. The run of record does pass through th… |
| 68 | 4.5.7 Seeding and replication | A single global seed, 42, is reset via random.seed, numpy.random.seed, torch.manual seed and torch.cuda.manual… |
| 68 | 4.5.7 Seeding and replication | It guarantees nothing about how sensitive the outcome is to that state. Every configuration was trained exactl… |
| 69 | 4.5.8 No hyperparameter search | Every value in this section: the focal loss parameters, learning rate and weight decay, warmup length and anne… |
| 69 | 4.6 Evaluation Protocol | Every configuration is scored by the same protocol: the same fold structure, checkpoint- selection rule, cross… |
| 69 | 4.6.1 Leave-one-subject-out folds | Evaluation uses leave-one-subject-out cross-validation. Since the 156-clip three-class working set retains 25 … |
| 70 | 4.6.2 Checkpoint selection and the absent inner validation split | Stated plainly, because it is the single most significant methodological limitation of this evaluation: there … |
| 70 | 4.6.2 Checkpoint selection and the absent inner validation split | The magnitude cannot be quantified from the stored output (subsection 4.6.6), but its scale follows from the f… |
| 70 | 4.6.2 Checkpoint selection and the absent inner validation split | One further value should not be mistaken for an alternative in use: val fraction = 0.2 exists on ExperimentCon… |
| 71 | 4.6.4 The primary metric: pooled macro F1 | The primary reported metric is pooled macro F1 which is the mean of the three values in the stored per class f… |
| 71 | 4.6.5 Why mean-of-folds macro F1 is rejected | Pooled macro F1 is primary, and the stored macro f1 field is not, because of how 156 clips distribute across 2… |
| 72 | 0.7 cannot serve as the headline result. | The same composition biases mean-of-folds accuracy in the same direction without a hard ceiling, since one pre… |
| 72 | 4.6.6 What is stored | For each configuration, Ablation Study/results/<config folder name>/ holds final results.json (the metrics of … |
| 72 | 4.6.6 What is stored | Two further artefacts are written per configuration but are not aggregates: training metrics.csv and checkpoin… |
| 72 | 4.6.6 What is stored | Per-clip predictions are not saved at all. No file records which fold evaluated a given clip, what was predict… |
| 73 | 4.6.7 Earlier evaluations, and why they are excluded | run protocol clips configs support Neg/Pos/Sur EVM pairs identical top-ranked confi tion A holdout, 60 epochs … |
| 73 | 4.6.7 Earlier evaluations, and why they are excluded | Every score in Table 4.4 is pooled macro F1, recomputed from each run’s stored per class f1 array; the macro f… |
| 73 | 4.6.7 Earlier evaluations, and why they are excluded | The imbalance treatment also changed across these runs, and that is the largest difference between them and ru… |
| 73 | 4.6.7 Earlier evaluations, and why they are excluded | Runs A and B carry the data-routing defect of subsection 3.2.3: all six magnification matched pairs are bit-id… |
| 74 | 4.6.7 Earlier evaluations, and why they are excluded | Coverage is incomplete in two further senses: run B scores only eight of the twelve configura- tions, so the m… |
| 74 | 4.7.1 A gap in the record, stated before anything else | No file produced by this study’s training runs records what hardware they ran on. final results.json, configur… |
| 74 | 4.7.2 What is recorded, and for one fold only | What each configuration summary.txt does record is a training time and a peak VRAM figure, written by ResultWr… |
| 75 | 4.7.2 What is recorded, and for one fold only | whichever is in hand when the summary is written rather than accumulated. What survives is whichever fold ran … |
| 75 | 4.7.2 What is recorded, and for one fold only | configuration last-fold train time (s) peak VRAM (MB) config 1 pure base 56.47 165.5 config 4 motion amp base … |
| 75 | 4.7.3 The convolutional stem, not the transformer, sets the cost | Table 4.5 splits cleanly in two. The four configurations without the 3D-CNN backbone run their last fold in 56… |
| 75 | 4.7.3 The convolutional stem, not the transformer, sets the cost | This is set by the stem, not the transformer, even though the transformer holds about 96 % of a full configura… |
| 76 | 4.7.4 GPU-hours: an extrapolation, not a measurement | The analysis scripts report approximately 50.6 GPU-hours across the sweep, of which about 48.9 hours, 96.6 %, … |
| 76 | 4.7.4 GPU-hours: an extrapolation, not a measurement | The arithmetic is reproducible from Table 4.5: the eight CNN-bearing configurations’ stored fold times sum to … |
| 76 | 4.7.5 Mixed precision, and what a reproduction would need | Mixed precision was enabled and, as subsection 4.5.6 records, takes effect only on CUDA. The peak-VRAM figures… |
| 76 | 4.7.5 Mixed precision, and what a reproduction would need | What is missing for exact reproduction is everything subsection 4.7.1 lists: the GPU model, driver version, CU… |
| 77 | 4.8 Conclusion | the remaining twelve the object of study, each effect read off matched pairs differing in exactly one flag, fo… |
| 77 | 4.8 Conclusion | These sections also fix the bounds within which the results may be interpreted. A single fixed seed, with no r… |
| 78 | 5 Results | The results use the twelve complete LOSO evaluations stored under Ablation Study/results/. The primary metric … |
| 78 | 5.1.2 The headline result | Configuration 2 is best on pooled macro F1: 0.7122. Its transformer receives motion features through spatial p… |
| 79 | 5.1.2 The headline result | Rank Config. EVM SimAM CNN Transf. Macro F1 Accuracy Correct 1 config 2 – – – ✓ 0.7122 0.7436 116 2 config 8 ✓… |
| 79 | 5.1.3 A reference point: the all-Negative classifier | An input-independent classifier predicting Negative for every clip obtains accuracy 99/156 = 0.6346. Its Negat… |
| 79 | 5.1.3 A reference point: the all-Negative classifier | Figure 5.2 contrasts pooled and mean-of-folds scores. With all three classes retained and undefined class F1 s… |
| 81 | 5.2.3 What consistency across six of six means, and does not | Even the smallest transformer gain exceeds the absolute mean effect of every other component. The repeated pos… |
| 82 | 5.2.3 What consistency across six of six means, and does not | Pair (off → on) ∆ pooled macro F1 config 1 → config 2 +0.2786 config 16 → config 8 +0.2467 config 4 → config 1… |
| 83 | 5.2.5 Interpretation | The gain supports the encoder as implemented. It does not isolate the contribution of temporal order from atte… |
| 83 | 5.3.2 The six pairs | Pair (off → on) Other enabled blocks ∆ pooled macro F1 config 9 → config 7 CNN + transformer +0.0795 config 6 … |
| 84 | 5.3.4 A pattern, not a conclusion | Three of four stem-bearing pairs improve with EVM; 5 → 16 declines by 0.0109. The two stem-free pairs split on… |
| 85 | 5.4.1 The effect | The mean effect is +0.0034 pooled macro F1 across four pairs: three positive, one negative, ranging from −0.02… |
| 86 | 5.4.3 Why the mean is not the finding | Table 5.5 shows the two pairs with the largest changes. For 9 → 6, Surprise F1 rises from 0.3000 to 0.5902, bu… |
| 86 | 5.4.5 Cost | SimAM adds no learned parameters but increases stored last-fold training time by approx- imately 11% and peak … |
| 87 | 5.5.2 The four pairs | Pair (off → on) Components also present ∆ pooled macro F1 config 4 → config 13 EVM +0.0094 config 12 → config … |
| 88 | 5.5.4 The cost | Table 5.7 compares stored peak VRAM and estimated training cost. The eight stem-bearing configurations account… |
| 89 | 5.6.1 Per-class F1 for all twelve configurations | Configuration Negative Positive Surprise Pooled macro F1 config 2 0.8068 0.6506 0.6792 0.7122 config 8 0.8458 … |
| 90 | 5.6.2 No configuration abandons a class | Every configuration achieves nonzero pooled F1 on all three classes; the minimum is configur- ation 4’s Surpri… |
| 90 | 5.6.3 config 9 is the most imbalanced configuration in the study | Configuration 9 has the highest Negative F1, 0.8491, and the second-lowest Surprise F1, 0.3000. Its best-to-wo… |
| 91 | 5.6.6 A caution about minority-class numbers | One changed prediction can noticeably alter a minority-class score, but its effect on F1 depends on the existi… |
| 92 | 5.7.3 Where this study lands | The study’s best score, 0.7122, is numerically above LBP-TOP (0.7026) and Quang et al. (0.7068), and below the… |
| 93 | 5.7.4 Three reasons the comparison is not like-for-like | • Evaluation population. MEGC’s CASME II subset has 145 clips; this study has 156. The working mapping include… |
| 93 | 5.7.4 Three reasons the comparison is not like-for-like | • Checkpoint selection. This study selects the epoch using the same held-out subject subsequently scored, with… |
| 94 | 5.8.2 The headline tension, and what it means for this thesis | The all-components system is outperformed on the primary metric by configuration 2, while retaining a one-clip… |
| 95 | 5.8.2 The headline tension, and what it means for this thesis | Figure 5.9: All twenty matched pairs, grouped by component. Each bar is one pair differing in exactly one swit… |
| 97 | 6 Conclusion | This thesis investigated which stages of the standard micro-expression recognition pipeline are responsible fo… |
| 97 | 6 Conclusion | This chapter closes the account. section 6.1 states what the study contributes; section 6.2 sets out the limit… |
| 97 | 6.1 Contributions | subsection 3.10.2 states the gap this study closes: the field has settled on a standard pipeline for micro-exp… |
| 97 | 6.1 Contributions | • A matched-pair factorial measurement of the standard pipeline. The reviewed corpus evaluates components only… |
| 98 | 6.1 Contributions | an otherwise identical pipeline, and none varies several jointly so that their interactions can be read. Twelv… |
| 98 | 0.7122 pooled macro F1 against config 8 proposed unified’s 0.6659 (subsec- | tion 5.1.2); and the component consuming most of the sweep’s compute is the only one whose mean effect is nega… |
| 98 | 0.7122 pooled macro F1 against config 8 proposed unified’s 0.6659 (subsec- | • The shape of the effect distribution across the four components. The con- tribution is not any individual me… |
| 98 | 0.7122 pooled macro F1 against config 8 proposed unified’s 0.6659 (subsec- | • A documented account of what was excluded. subsection 4.6.7 records four earlier evaluations of the same mat… |
| 98 | 6.2 Limitations | subsection 5.8.3 gathers the limitations bearing on the results themselves, and its closing bullet, ”that each… |
| 99 | 6.2 Limitations | The ablation measures architecture at one point in preprocessing space. sub- section 3.10.3 records that α = 1… |
| 99 | 6.2 Limitations | Twelve declared divergences from the methods reviewed. subsection 3.10.4 ledgers them; three bear directly on … |
| 99 | 6.2 Limitations | One corpus, one demographic, trained from scratch, with no composite training. subsection 3.1.5 states the cos… |
| 99 | 6.2 Limitations | These three bound what the study is valid for. The measurements report what four components contribute to this… |
| 99 | 6.3 Further Work | section 5.8 closes by asking what a stronger design would still need to establish. The answer falls into three… |
| 100 | 6.3 Further Work | Making the numbers testable. Three changes repair the evaluation rather than the model, and none requires an a… |
| 100 | 6.3 Further Work | What the findings point at. Of the experiments the findings themselves motivate, subsection 3.6.1 puts one fir… |
| 101 | 6.3 Further Work | ledger: flow-based fine alignment, which that section names the most promising unexploited improvement and whi… |
| 101 | 6.3 Further Work | Beyond the matrix, the corpus itself is the binding constraint. subsection 3.1.5 states the cost of remaining … |
| 102 | 6.4 Final Remarks | This thesis asked a narrow question of a standard pipeline. Not how good a score it can be made to produce, bu… |
| 102 | 6.4 Final Remarks | subsection 3.10.2 puts the practical complaint directly: a practitioner assembling this pipeline today has no … |
| 102 | 6.4 Final Remarks | The design is the more durable contribution. Building the matrix rather than the proposal is what made that or… |

---

## Category D — Your own argument, framing or transition

No citation expected. Listed only so the audit is complete.

| Page | Section | Opening words |
|---|---|---|
| 1 | 1.1 Research Motivation | To build a system that is capable of recognising these movements automatically is not about just pointing a cl… |
| 2 | 1.1 Research Motivation | information that distinguishes one emotion from another. It is delicate, so the information that these frames … |
| 2 | 1.1 Research Motivation | Systems built for the task have responded to those constraints in a consistent way. Rather than a single model… |
| 4 | 2 Background | Micro-expression recognition combines a weak visual signal with limited, imbalanced training data. The relevan… |
| 8 | 2.4.1 Why motion rather than pixels | Flow makes displacement explicit instead of asking the network to infer it from appearance alone. This can red… |
| 10 | 2.5.1 Units, layers and the forward pass | A unit computes a = ϕ(wT x + b), where w and b are learned weights and bias, ϕ is an activation function and a… |
| 12 | 2.6 Network Building Blocks | The model combines a spatial stem, optional activation re-weighting, an optional temporal encoder and a common… |
| 16 | 2.7.2 Focal loss: weighting by difficulty | For γ > 0, the modulating factor reduces the contribution of confident, correct examples relative to difficult… |
| 20 | 2.8.5 What the evaluation apparatus must retain | Second, a nested evaluation separates fitting, checkpoint selection and final testing. The present procedure c… |
| 20 | 2.10 The Pipeline End to End | The pipeline runs in two stages, and the division matters because the two are executed a very different number… |
| 28 | 3.1.6 How CASME II is evaluated: MEGC 2019 and the metric convention | Two further observations from the challenge bear on this design. The top three submissions all chose optical f… |
| 29 | 3.1.7 Threats to validity carried by the corpus | Scale and imbalance. 156 usable clips across three classes, distributed 99 : 32 : 25, small enough that parame… |
| 30 | 3.1.8 Implications for this research | The consequence that shapes interpretation rather than construction is that the small- sample regime predicts … |
| 33 | 3.2.3 Implications for this research | Behind those settings lies a hypothesis the corpus does not test, stated because it frames the interpretation.… |
| 34 | 3.2.3 Implications for this research | A defect the review made detectable. In this project’s earliest evaluations every magnification-enabled config… |
| 36 | 3.3.1 Implications for this research | The central interpretive point is that optical flow is an analytical spatio-temporal feature extractor : it es… |
| 37 | 3.4 Deformation Representation: The Optical Strain Tensor | and describe robustness to moderate head translation, which does not imply immunity to arbitrary head movement… |
| 39 | 3.4.3 Implications for this research | system substitutes for it. A deterministic transformation of u and v adds no independent observation; its pote… |
| 44 | 3.7 Parameter-Free Attention: SimAM | The limits of transfer are specific. The source experiments concern natural images, different training-set siz… |
| 45 | 3.7.2 Relevance and scope | The study evaluates SimAM in a motion-input, small-corpus setting through four matched comparisons under compl… |
| 46 | 3.8.2 Implementation and six declared divergences | The evaluated component receives 32 tokens of width 96, from either the convolutional stem or spatial pooling … |
| 49 | 3.9.1 Implications for this research | What this project contributes is a negative result about composing corrections. The configuration is resolved … |
| 51 | 3.10.2 The research gap | The reviewed literature has established a standard pipeline for micro-expression recognition: magnify, compute… |
| 55 | 4.1.1 The form of the study | This is not the proposal and evaluation of a single architecture. It is a factorial ablation: four components … |
| 57 | 4.1.4 Matched pairs and their counts | Because the retained cells form a near-complete factorial design, each flag’s effect is es- timated by matched… |
| 57 | 4.1.4 Matched pairs and their counts | The counts differ by flag, and the asymmetry is arithmetic rather than oversight. Valid- ity depends only on t… |
| 58 | 4.2.2 From 255 rows to 156 clips | Others is excluded before any model is trained, leaving 156 clips. It is not an emotion label but the residual… |
| 58 | 4.2.3 The three-class grouping | The 156 retained clips are mapped to three coarser classes by GROUPED EMOTION MAP, which assigns {"Negative": … |
| 61 | 4.3.3 Naming is not contiguous | Configurations 1 through 8 carry the descriptive names of the original eight-cell design that preceded the ful… |
| 62 | 4.3.5 Matched pairs | noisier estimate of the same quantity than one over six; any comparison between the size of a four-pair effect… |
| 62 | 4.4.1 Which implementation this is | AblationMERModel is one nn.Module whose forward pass conditionally executes or skips each of the three model-l… |
| 63 | 4.4.5 The transformer | The input sequence is formed from the concatenated feature map [B, 96, 32, 112, 112] by AdaptiveAvgPool3d((32,… |
| 64 | 4.4.5 The transformer | It is a fixed buffer, not a learned parameter, and contributes nothing to the parameter counts. Although the p… |
| 64 | 4.4.7 Fallbacks for disabled components | • use cnn=False → RawPatchEmbedding. The raw input [B, 3, 32, 224, 224] is average- pooled with AdaptiveAvgPoo… |
| 64 | 4.4.7 Fallbacks for disabled components | • use transformer=False → TemporalPooling. The 32-step sequence is collapsed to one vector by a plain mean (or… |
| 65 | 4.4.8 Weight initialisation | Every configuration is initialised by the same routine, applied once at construction and inde- pendent of whic… |
| 65 | 4.4.9 Parameter counts | In a full configuration the transformer accounts for roughly 96 % of total parameters (348,736 of 363,763), wh… |
| 68 | 4.5.6 Numerical safeguards | Gradients are clipped to a global norm of 1.0 via clip grad norm , applied after the AMP scaler’s unscale step… |
| 74 | 4.6.7 Earlier evaluations, and why they are excluded | A different configuration ranks first in each of the five runs. That is not offered as evidence that protocol … |
| 76 | 4.8 Conclusion | This chapter has specified, rather than argued for, what is needed to reproduce these experiments and read the… |
| 79 | 5.1.2 The headline result | Configuration 8 leads on pooled accuracy, 0.7500 versus 0.7436, corresponding to 117 versus 116 correct clips.… |
| 80 | 5.1.5 The baseline, and how many configurations fall below it | The designated baseline is configuration 4, EVM with spatial pooling, a learned projection and temporal averag… |
| 84 | 5.3.3 The most consequential observation | Adding EVM to the best configuration, 2 → 12, produces its largest loss, −0.0541, over three and a half times … |
| 85 | 5.3.5 Interpretation | Two implementation choices offer possible explanations. First, clips are sampled to 33 frames before magnifica… |
| 86 | 5.4.4 What this supports, and what it does not | The largest-loss pair, 13 → 16, declines on all three classes, including Surprise from 0.4035 to 0.3934. SimAM… |
| 87 | 5.5.3 The largest stem-associated loss | Three differences have absolute magnitude below 0.01. The fourth, 2 → 9, reduces the best configuration from 0… |
| 89 | 5.5.5 Interpretation | The no-CNN route still learns a projection, and toggling the stem changes capacity as well as processing. The … |
| 91 | 5.6.4 config 2 balances high class scores | Relative to configuration 8, configuration 2 improves Positive F1 from 0.5556 to 0.6506 and Surprise from 0.59… |
| 91 | 5.6.5 Negative-class F1 tracks the transformer split | Negative F1 separates transformer configurations (0.7485–0.8491) from those without (0.3500– 0.5641), with a g… |
| 94 | 5.7.5 What the comparison does support | The comparison locates the study’s score within the range of published results while exposing the protocol dif… |
| 94 | 5.8.1 The four component findings, recapitulated | The transformer has the largest mean and improves every tested pair. EVM is mixed, including a loss for the be… |

---

## Separately: passages that will flag in a similarity checker but are not plagiarism

- The German *Eidesstattliche Erklärung* (front matter). Standard BTU template text, present in every thesis from the faculty. Exclude it from the scan or tell your examiner.
- The Task Description page. Supervisor-supplied text.
- Standard equations (softmax attention, batch-norm, focal loss, F1). Formulas match across every source by definition.
- The reference list itself. Bibliographies always register high similarity.

## Unrelated fix before submission

The PDF metadata still carries the LaTeX template defaults:

```
Title:    Bachelorarbeit
Subject:  e.g. Document type
Author:   Max Mustermann
```
Set `\title`, `\author` and the `hyperref` `pdfinfo` fields before you submit.
