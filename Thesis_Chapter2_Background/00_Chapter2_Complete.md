# Chapter 2 — Background

## Scope and conventions

This chapter sets out the concepts and techniques needed to follow the methodology in Chapter 4 and the results in Chapter 5. It explains **what each mechanism is and how it works**; Chapter 3 surveys who has used each one, with what result, and what gap remains. Where a fact is needed in both places it is established here and cross-referenced there.

**Scope is limited to what the system actually does.** Every technique described is present in the implemented pipeline. Techniques a reader might expect in a background chapter but which this project does not use — quantisation, pruning and knowledge distillation among them — are not covered, and their absence is noted at the point where it would otherwise be assumed (§2.5).

**Sources.** Every work cited is a paper held in the project's `docs/` corpus. Standard machine-learning material for which the corpus holds no source is presented without citation rather than with an invented one; the reference list states which topics are handled that way. Works cited only inside the reviewed papers are attributed in the text to the paper that reports them.

## Contents

| § | Topic |
|---|---|
| 2.1 | The phenomenon: micro-expressions as involuntary, brief, low-intensity movement |
| 2.2 | From video to motion: optical flow and optical strain |
| 2.3 | Motion magnification |
| 2.4 | Network building blocks |
| 2.5 | Learning under scarcity and skew |
| 2.6 | Evaluating on a small corpus |

---

## 2.1 The Phenomenon: Micro-Expressions as Involuntary, Brief, Low-Intensity Facial Movement

A micro-expression is a brief facial movement that leaks a felt emotion the person is actively trying to conceal (Yan et al., 2014). It is best understood by contrast with an ordinary, or macro-, expression, on three axes rather than one. Macro-expressions typically last upwards of half a second, and may run to several seconds, and can be produced at will (Qu et al., 2016); micro-expressions, by contrast, are "characterized by short durations, involuntary generation and low intensity" (Qu et al., 2016). The third axis is the one most easily overlooked: a micro-expression cannot be performed on instruction. It is a symptom of concealment rather than a communicative act. That single property governs how such data can be gathered at all, and the consequences for corpus construction are taken up in §3.1.

### Duration: the first constraint

The field's working definition fixes an upper bound on duration. Yan et al. (2014) state that "the generally accepted upper limit of the duration is 1/2 s." Brevity is the first source of the difficulty in recognising the phenomenon automatically, and the mechanism is simple: whatever signal distinguishes one emotion from another must be carried by however many video frames fit inside that half-second window. At a modest frame rate that window may contain only a handful of frames, several of which are near-identical to their neighbours; the sequence available to a classifier is short by construction, not by any deficiency of the recording.

### Low intensity: the second constraint

Duration alone would still leave a tractable problem if the movement were large. It is not. Yan et al. (2014) observe that micro-expressions are "usually low in intensity – it might be so brief for the facial muscles to become fully-stretched with suppression", with the consequence that "because of the short duration and low intensity, it is usually imperceptible or neglected by the naked eyes". In practice the movement of interest may change only a few grey levels over a small patch of skin — around an eyebrow, or the corner of a mouth — while the rest of the face stays static. This is a distinct obstacle from brevity: a longer recording does nothing to help it, because the difficulty is not how many frames are available but how far the signal in each frame sits above the noise floor of ordinary video. Duration limits how much evidence exists; intensity limits how visible that evidence is once it does.

### Onset, apex and offset

Because a micro-expression is a movement rather than a static configuration, it is conventionally described by three temporal landmarks. The **onset** is the frame at which the face begins to depart from its neutral baseline; the **offset** is the frame at which it returns to neutral; and the **apex** is the frame at which the movement is judged most intense — the point at which, as Ekman puts it, a "snapshot taken at [the] point when the expression is at its apex can easily convey the emotion message" (as reported by Li, Huang & Zhao, 2018). All three landmarks are available in the corpus used here. This thesis, however, uses only the onset and offset frames to bound each clip passed into the pipeline; the apex frame is never read. This is a deliberate design choice rather than an oversight, and its consequences — for what information the model can and cannot see, and for how its results relate to apex-frame methods in the literature — are taken up in Chapters 3 and 4.

### Action units, and the label actually used

The Facial Action Coding System (FACS) describes any facial movement, however subtle, as a combination of discrete Action Units (AUs), "an objective method for labeling facial movements in terms of component actions" (Yan et al., 2014). An AU is a description of movement, not of emotion, and the mapping between the two is not one-to-one; corpora therefore assign emotion categories using AUs together with other evidence, by a procedure that differs between datasets and is described for the corpus used here in §3.1.2.

It is worth being precise about how much of that machinery survives into this thesis: none of it. The work reported here uses only the resulting emotion label. The Action Units recorded for each clip are present in the label table as inherited metadata from the original annotation — carried along because the released coding includes them — but they are never read by any stage of the data pipeline and never presented to the model. Nothing in this thesis performs, or claims to perform, action-unit recognition; the sole target throughout is the emotion category.

---

---

## 2.2 From Video to Motion: Optical Flow and Optical Strain

### 2.2.1 Why motion, not pixels

A micro-expression is a movement, so the natural quantity to hand a classifier is a description of movement rather than of appearance. Raw pixel intensity carries the signal of interest — a few grey levels of change around an eyebrow or a mouth corner — embedded in everything else the camera also recorded: who the subject is, how the scene is lit, the fixed geometry of a particular face. None of that is discarded by presenting a network with frames directly, and on a corpus this narrow in identity and illumination a model has every incentive to fit the wrong part of the image. A displacement field is different in kind: it records how each point moved, not what colour it is, so identity and illumination do not enter it at all. That principle motivates the rest of this section; the comparative case for it is made in §3.3.1.

### 2.2.2 Optical flow: the brightness-constancy constraint

Optical flow estimates, for every pixel, the apparent two-dimensional displacement between two frames. The estimate rests on the brightness-constancy assumption: a point on a moving surface is taken to have the same intensity in the next frame as in the current one, so that any change in a pixel's grey level is attributed to something having moved past that location, never to the thing itself changing colour or shade (Liong et al., 2019a).

Writing $I_t(x,y)$ for the image intensity at position $(x,y)$ and time $t$, and supposing the point there moves to $(x+\delta x,\, y+\delta y)$ by time $t+1$, brightness constancy states

$$I_t(x,y) = I_{t+1}(x+\delta x,\, y+\delta y), \qquad \delta x = u_t\,\delta t,\;\; \delta y = v_t\,\delta t,$$

where $u_t(x,y)$ and $v_t(x,y)$ are the horizontal and vertical components of the flow at that point. Expanding the right-hand side by a first-order Taylor series and substituting into the constancy equation, the intensity terms cancel and division by $\delta t$ leaves the **optical flow constraint equation**:

$$u_t(x,y)\,\frac{\partial I}{\partial x} + v_t(x,y)\,\frac{\partial I}{\partial y} + \frac{\partial I}{\partial t} = 0.$$

This single scalar equation holds at every pixel but contains two unknowns, $u_t$ and $v_t$: the image gradients are measurable directly from the frames, while the displacement is not determined by them alone. This under-determinacy is the **aperture problem** — a local patch of image constrains the component of motion perpendicular to an edge or gradient, but says nothing about motion parallel to it. One equation per pixel cannot fix two unknowns per pixel, so every practical estimator closes the system by adding a further assumption relating the flow at neighbouring pixels, typically that the field varies smoothly across the image. That added assumption is what distinguishes one flow algorithm from another; this thesis does not adjudicate between them (§3.3.3) and simply adopts one, described in §2.2.4. The output, for a pair of frames, is the pair of scalar fields $u(x,y)$ and $v(x,y)$ — the horizontal and vertical displacement at every pixel — forming the first two channels of the motion representation used throughout this thesis.

### 2.2.3 Optical strain: the finite strain tensor

Flow answers where a point moved; it does not by itself distinguish a patch of skin that moved together with its neighbours from one that stretched or sheared relative to them. Optical strain makes that distinction by taking the spatial derivative of the flow field rather than the field itself.

Writing the displacement at a point as $\mathbf{u} = [u, v]^{\mathsf T}$, the **finite strain tensor** is defined as the symmetric part of the displacement gradient (Shreve et al., 2011):

$$\varepsilon = \tfrac{1}{2}\left[\nabla\mathbf{u} + (\nabla\mathbf{u})^{\mathsf T}\right] = \begin{bmatrix} \varepsilon_{xx} & \varepsilon_{xy} \\ \varepsilon_{yx} & \varepsilon_{yy} \end{bmatrix}, \qquad \varepsilon_{xx} = \frac{\partial u}{\partial x},\;\; \varepsilon_{yy} = \frac{\partial v}{\partial y},\;\; \varepsilon_{xy} = \varepsilon_{yx} = \frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right).$$

$\varepsilon_{xx}$ and $\varepsilon_{yy}$ are the **normal strain** components, describing stretching or compression along each axis; $\varepsilon_{xy}$ is the **shear strain** component, describing the change of angle between two initially perpendicular directions on the surface. The tensor is reduced to a single scalar per pixel:

$$\varepsilon_{\text{mag}} = \sqrt{\varepsilon_{xx}^{2} + \varepsilon_{yy}^{2} + \varepsilon_{xy}^{2}}.$$

The property that makes this useful, rather than a second copy of the flow field, follows from its definition as a *gradient*. If a region of the face translates rigidly — every pixel moving by the same $u$ and the same $v$, as a small head movement produces — then $u$ and $v$ are locally constant there, and their spatial derivatives, hence every component of $\varepsilon$, are zero: a rigid translation contributes nothing to the strain field, however large it is. A localised muscle contraction, by contrast, moves neighbouring points of skin by different amounts, so the displacement field has a non-zero gradient exactly where the deformation occurs, and $\varepsilon_{\text{mag}}$ is large there. Strain is thus sensitive to non-rigid deformation specifically, insensitive by construction to whatever rigid motion the flow field also contains.

### 2.2.4 Assembling the three-channel tensor

For each clip, the frames spanning onset to offset are uniformly resampled to a fixed sequence of 33 frames, and dense optical flow is computed between every adjacent pair using OpenCV's implementation of Farnebäck's method (Zhao et al., 2021) — `cv2.calcOpticalFlowFarneback`, with pyramid scale 0.5, three pyramid levels, window size 15, three iterations, a polynomial neighbourhood of size 5 and a polynomial smoothing of 1.2, applied to 8-bit greyscale frames. This yields **32 flow fields** per clip, each a $(u, v)$ pair over the 224 × 224 face region. Strain is computed from each field's spatial gradients using `np.gradient` (central differences at unit pixel spacing, over the two spatial axes only; no temporal gradient is used), giving $\varepsilon_{xx}$, $\varepsilon_{yy}$, $\varepsilon_{xy}$ and hence $\varepsilon_{\text{mag}}$ for each of the 32 pairs.

The three quantities — $u$, $v$ and $\varepsilon_{\text{mag}}$ — are stacked as three channels, giving a tensor of shape $(3, 32, 224, 224)$ in channel order $[u, v, \text{strain}]$. When Eulerian video magnification is applied (§2.3), flow and strain are computed on the magnified frames rather than the originals; the construction is otherwise unchanged.

Normalisation is applied in two separate stages, and both remain active. At extraction time, each of the three channels is independently min–max normalised to $[0, 1]$ across the whole clip, so that a channel's own range — which differs enormously between a flow component and a strain magnitude — does not by itself determine its influence. Separately, at training load time (`Ablation_Study/dataset.py`, with `normalize=True`), each channel is z-scored per sample over its $T \times H \times W$ extent. Neither stage substitutes for the other: the first is a per-clip rescaling done once, during data preparation; the second is a per-sample standardisation applied every time a clip is loaded, centring and scaling each channel to zero mean and unit variance. Both are applied to every clip the model sees.

---

---

## 2.3 Motion magnification

### 2.3.1 The Eulerian idea

A moving object can be described in two ways. A *Lagrangian* description follows a point on the object as it moves — this is what optical flow and feature tracking do: locate a point in frame $t$, find its correspondence in frame $t+1$, and report the displacement. A *Eulerian* description instead fixes attention on a location in the image and asks how the signal observed there changes over time. Eulerian video magnification (EVM) takes the second view: no correspondence between frames is computed. At every pixel, it treats the sequence of intensity values over time as a 1-D signal, isolates the part occurring in a chosen temporal frequency band, multiplies it by a factor $\alpha$, and adds the result back to the original sequence.

The justification for why this amplifies *motion*, and not merely intensity, is a first-order argument. If a small one-dimensional image profile $f(x)$ translates by a time-varying displacement $\delta(t)$, the observed intensity at a fixed location is $I(x,t) = f(x+\delta(t))$. A first-order Taylor expansion around $x$ gives

$$I(x,t) \approx f(x) + \delta(t)\,\frac{\partial f(x)}{\partial x},$$

so the temporal variation at a fixed pixel is approximately proportional to the true displacement $\delta(t)$, scaled by the local spatial gradient. Amplifying that temporal variation by $\alpha$ before adding it back is therefore, to first order, equivalent to synthesising a signal consistent with a displacement of $(1+\alpha)\delta(t)$ rather than $\delta(t)$: motion is amplified as a consequence of amplifying intensity change, with no tracking step anywhere in the process. This is the account given by Bai et al. (2021, following Wu et al.).

### 2.3.2 Spatial decomposition: the Laplacian pyramid

The Taylor approximation above only holds where the spatial gradient is well-behaved relative to the displacement, so in practice the frame is first decomposed into a set of spatial-frequency bands and the argument is applied band-by-band. The standard tool is the **Laplacian pyramid**. Each level is built by blurring the current image with a small low-pass kernel, downsampling by a factor of two, upsampling the result back to the original resolution, and subtracting it from the pre-downsampling image. The subtraction discards everything the blur-and-downsample step could reconstruct, leaving only the spatial detail specific to that scale — the Laplacian band. The downsampled image feeds the next iteration, and the recursion terminates in a coarse, heavily blurred residual (a Gaussian, not a Laplacian, band). A pyramid of depth $L$ yields $L$ Laplacian bands, each isolating a different range of spatial frequencies, plus one coarsest residual that carries no band-specific detail and exists only to seed reconstruction.

### 2.3.3 Temporal filtering

Each spatial band is treated as a stack of scalar time series, one per pixel, and passed through a **temporal band-pass filter** that keeps only the variation falling inside a chosen frequency range and discards the rest. The filtered signal at each band is multiplied by $\alpha$ and added back to that band's original values; the bands are then recombined by repeating the pyramid construction in reverse — upsampling the coarsest residual and successively adding each (now-amplified) Laplacian band back in. The published amplitude-based method realises the band-pass step with a **Butterworth filter**, a smooth infinite-impulse-response design (Bai et al., 2021).

### 2.3.4 The amplification trade-off

The magnification factor $\alpha$ trades sensitivity against artefact. A larger $\alpha$ makes a fainter in-band motion visible, but the same multiplication applies indiscriminately to everything the band-pass filter passes: sensor noise, compression artefacts, and any other real motion whose temporal frequency happens to fall inside the chosen band are amplified exactly as the motion of interest is. The method has no way to distinguish a wanted signal from an unwanted one occupying the same frequency range — the filter only knows how fast a signal oscillates, not what it is. Raising $\alpha$ therefore does not monotonically improve the result; past some point amplified noise and induced displacement artefacts dominate whatever gain in visibility was sought. This tension is intrinsic to the method rather than a defect of any one implementation, and it recurs, with numbers, in Chapter 3.

### 2.3.5 Departures from the textbook method in this implementation

The magnifier used in this thesis (`Stage1_DataPipeline/evm_magnifier.py`) implements the mechanism above with a genuine four-level Laplacian pyramid — a $[1,4,6,4,1]/16$ blur kernel, downsample-by-two, upsample, subtract — followed by a coarsest Gaussian residual, matching §2.3.2 exactly. It departs from the amplitude-based method as described by Bai et al. (2021) in four respects.

First, the temporal band-pass is an **ideal filter**: a hard binary mask applied directly to the discrete Fourier transform of each pixel's time series (via `scipy.fftpack`), rather than a Butterworth (IIR) filter. Frequencies inside $[\omega_{\text{low}}, \omega_{\text{high}}]$ pass with a gain of exactly one and everything else is zeroed, with no smooth roll-off at the band edges.

Second, all four Laplacian bands are amplified by the **same scalar $\alpha$**; the published method reduces $\alpha$ at higher spatial frequencies to limit noise amplification, and no such damping is applied here. The coarsest residual is never filtered or amplified — it only seeds reconstruction.

Third, the implementation works directly on the greyscale intensity tensor, so there is no chromatic attenuation step.

Fourth, and a property of pipeline ordering rather than of the magnifier itself: the calling code (`tensor_pipeline_manager.py`) resamples each clip to a fixed 33 frames *before* invoking the magnifier, and passes the corpus's nominal recording rate (200 fps) as the filter's frame-rate parameter regardless of the resampled sequence's true frame spacing. The consequence — a temporal filter whose frequency axis no longer matches the signal it is filtering — is analysed in §3.5.7, not here.

The default operating point used throughout is $\alpha = 10$, a band of 5–25 Hz, and 4 pyramid levels; the amplified output is clipped to $[0,255]$ and quantised back to 8-bit before being handed to the downstream flow-and-strain extractor.

---

---

## 2.4 Network Building Blocks

Chapter 3 argues why each of the following components was chosen and what its measured contribution was. This section stays one level below that argument, setting out the mechanism each component computes.

### 2.4.1 Three-dimensional convolution and kernel shape

A `Conv3d` layer generalises image convolution to a five-dimensional tensor of shape $(B, C, D, H, W)$ — batch, channel, and three spatial-like axes, here depth $D$ (time), height $H$ and width $W$. A kernel of shape $(k_D, k_H, k_W)$ slides over all three of the non-channel, non-batch axes at once, at each position computing a weighted sum over every input channel and every position it currently covers. Stacking $C_{\text{out}}$ such filters produces $C_{\text{out}}$ output channels; stride and padding along each axis control how the output's $D$, $H$ and $W$ compare to the input's.

The detail that matters most for what follows is the kernel's temporal extent, $k_D$. If $k_D > 1$, the filter reaches across multiple frames at every application, so a single output value is a genuine function of more than one time step — a spatio-temporal filter in the literal sense. If $k_D = 1$, the filter at time step $t$ only ever reads input at time step $t$; no output value depends on any other frame. A stack of layers with kernel shape $(1, k_H, k_W)$ is therefore a two-dimensional convolution applied identically and independently to every frame, merely expressed using `Conv3d` operations so the tensor never leaves its $(C, D, H, W)$ layout. This is the case the model in this thesis uses throughout its convolutional stem: every kernel is shaped $(1, 3, 3)$, with padding $(0, 1, 1)$ preserving the spatial extent while the temporal axis is neither padded nor touched. The temporal dimension is carried through the stem unchanged in length; only $H$ and $W$ are affected, first by the convolutions' own padding and then by a spatial-only max-pool of shape $(1, 2, 2)$ that halves both.

### 2.4.2 Normalisation and regularisation

`BatchNorm3d` normalises each channel independently: for channel $c$ it computes a mean and variance over every other axis at once — the batch dimension and all spatial-temporal positions $(D, H, W)$ — and rescales,
$$\hat{x} = \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad y = \gamma_c \hat{x} + \beta_c,$$
where $\gamma_c$ and $\beta_c$ are learned per-channel scale and shift parameters. Training uses batch statistics; inference uses a running estimate accumulated during training, so the layer behaves exactly as its two-dimensional counterpart does, only pooled over one further axis.

`Dropout3d` differs from ordinary `Dropout` in what unit it zeroes. Ordinary dropout masks individual scalar activations independently at random, which is a weak regulariser on a convolutional feature map, since neighbouring positions in $H$, $W$ and $D$ are highly correlated and the network can read a missing value off an adjacent one. `Dropout3d` instead zeroes an entire channel — every $(D, H, W)$ position of it, for a given sample — with the given probability, removing a whole feature detector rather than a single correlated pixel.

`LayerNorm` normalises across the feature dimension of a single position for a single sample, independent of the batch, which is why it is the default choice inside transformer blocks rather than `BatchNorm`. Here it appears inside every transformer encoder layer and once more at the end of the classifier head, immediately before the final linear projection to the output classes.

### 2.4.3 Parameter-free attention: SimAM

SimAM (Yang et al., 2021) re-weights a feature map without learning any new parameters, by scoring each neuron on how distinctive it is from its surrounding context and using that score directly as a multiplicative gate. For a neuron with activation $x$ in a channel whose mean and variance are $\mu$ and $v$ (estimated by pooling over that channel's own spatial extent), the closed-form energy reduces to
$$\text{energy}(x) = \frac{(x-\mu)^2}{4(v+\lambda)} + 0.5,$$
where $\lambda$ is the module's single hyperparameter. The refined output is then $x \cdot \sigma(\text{energy}(x))$, a monotonic sigmoid gate applied element-wise. What makes this mechanism unusual among attention modules is exactly what the formula shows: every quantity it needs — the mean, the variance, the resulting energy — is read off the feature map itself, at inference as much as at training time, so no weight matrix, bias, or reduction layer is introduced anywhere. The gate is a fixed function of the activations it is applied to, not a learned one.

The implementation in this thesis follows that formula exactly, with $\lambda = 10^{-4}$, but estimates $\mu$ and $v$ by pooling over the entire spatio-temporal volume $(D, H, W)$ of a stream's feature map rather than a single frame's spatial extent alone, so a neuron's distinctiveness is judged against the whole clip.

### 2.4.4 Self-attention

Self-attention relates every position in a sequence to every other position by learned linear projection rather than a fixed spatial neighbourhood. Each input vector is projected three ways, into a query $Q$, a key $K$ and a value $V$; attention weights are the scaled dot product of queries against keys,
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$
with the $\sqrt{d_k}$ scaling present to stop the dot products, and hence the softmax, from saturating as the key dimension $d_k$ grows. Multi-head attention runs several such projections in parallel, each into a smaller subspace, concatenating their outputs before a final linear mixing layer, so different heads can attend to different relationships within the same sequence.

Because this is a weighted sum over the whole sequence with weights derived purely from content, it is invariant to the order of the input positions: permuting the sequence permutes the output identically, with nothing in the mechanism encoding *where* a token sits. A positional encoding is added to each input vector to break that symmetry. Two schemes are common. A sinusoidal encoding assigns each position a fixed vector built from sine and cosine functions at geometrically varying frequencies across the embedding dimensions — deterministic, requiring no training, and evaluable at sequence lengths never seen during training. A learned encoding instead assigns each position index its own trainable embedding, which can fit the data more closely but cannot extrapolate beyond the longest position trained on. The encoder in this thesis uses the fixed sinusoidal form, held as a constant buffer rather than a parameter.

A second placement choice concerns normalisation relative to each sublayer. Post-norm — the original arrangement — applies `LayerNorm` after the residual addition that follows each sublayer; pre-norm applies it beforehand, to the sublayer's input, so the residual path carries an un-normalised signal straight through the stack, which is generally reported to train more stably since gradients reach earlier layers unimpeded. This is the placement used here. Beyond what the encoder layer itself contributes, no additional residual or skip connection is introduced anywhere else in the architecture.

### 2.4.5 The fallbacks

Each learned component here can be switched off, and each switch has a specific, near parameter-free replacement rather than simply removing that stage. With the convolutional stem off, every frame's motion channels are instead reduced by adaptive average pooling to a fixed $4\times4$ spatial grid regardless of input resolution, flattened per frame and passed through a single linear layer into the model's working dimension. With the transformer off, the sequence of per-frame feature vectors is instead collapsed by a plain mean over the time axis, with no parameters and no representation of frame order at all.

Both fallbacks add as little capacity as possible: the pooling operations introduce no parameters, and the one linear layer needed to match dimensions in the no-CNN case is negligible next to what it replaces. That near-absence of added capacity is what makes each fallback a fair control, isolating what the learned component contributes without also changing how much the network is capable of fitting.

---

## 2.5 Learning Under Scarcity and Skew

This section sets out the training machinery — the loss function, the sampling scheme, regularisation, and the optimiser — at the level of what each mechanism computes and how it is configured here. Chapter 3 (§3.9) reviews the literature motivating these choices, gives the corpus's class counts, and reports the failure mode that follows from combining two of them carelessly; that narrative is not repeated below. What follows is the mechanism and the configuration actually used.

### 2.5.1 What skew does to cross-entropy

Standard cross-entropy sums a per-example term $-\log p_t$, where $p_t$ is the model's predicted probability for the true class, weighting every example equally. Under a skewed class distribution that equal weighting is not equal in effect: a class's total gradient contribution is its example count times the per-example gradient, so a majority class dominates the sum purely by outnumbering the rest. A model facing this loss has a cheap route to a lower value — predict close to the prior on ambiguous cases — because that is correct often enough on the majority class to outweigh being wrong on every minority example. Loss can fall steadily while the model learns comparatively little about classes it rarely sees. The remainder of this section responds to that problem.

### 2.5.2 Focal loss: re-weighting by difficulty, not by frequency

Focal loss (Lin et al., 2017) inserts a modulating factor in front of the log term:

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^{\gamma} \log p_t.$$

Zhao et al. (2021) apply this formulation to micro-expression recognition and state the mechanism plainly: $\gamma$ is "the balance factor for loss" and $\alpha_t$ "the weight balance factor for samples". The distinction matters for what follows. The factor $(1-p_t)^{\gamma}$ shrinks toward zero as the model becomes confident and correct on an example, re-weighting by how *difficult* it currently finds that example, irrespective of its class — an easy majority-class example contributes almost nothing once learned; a hard example, of any class, keeps contributing. Nothing in that factor looks at how many examples of a class exist. Class frequency enters only through the separate $\alpha_t$ term, an explicit per-class multiplier ordinarily set near inverse class frequency.

The implementation here (`Ablation_Study/losses.py`) follows this exactly: it computes `log_softmax`, gathers $\log p_t$ for the true class, exponentiates to recover $p_t$, and forms `focal_weight = (1 - p_t) ** gamma`. Label smoothing (§2.5.5) is applied first, blending the negative log-likelihood term with a uniform term over classes, so the focal weight multiplies the *smoothed* loss rather than the raw NLL. The optional $\alpha$ vector, where supplied, multiplies the result afterwards, indexed by each example's true class. $\gamma = 2.0$ throughout, following Zhao et al.'s reported setting, with label smoothing at $0.05$.

### 2.5.3 Balanced sampling

The second lever acts on the data a batch is drawn from rather than on the loss computed over it. A `WeightedRandomSampler` assigns each training example a weight equal to the inverse of its class's count, computed over the training split of the current fold only. Sampling then proceeds with replacement, drawing `num_samples = len(train)` indices per epoch, so each class is presented with roughly equal probability regardless of its true rarity. This sampler is attached to the training loader alone; the validation loader uses no sampler and iterates with `shuffle=False`, so validation scores reflect the split's true distribution.

### 2.5.4 Why combining both is a choice, not a default

Focal loss's $\alpha$ term and the balanced sampler both correct for class frequency, at different points in the pipeline, so running both at once corrects for the same imbalance twice. This project resolves that at run time by an explicit rule: `use_loss_weights = use_class_weights and not use_balanced_sampler`. Both flags default to `True`, so in every default configuration the sampler wins and the loss-side $\alpha$ term is switched off. Focal loss, as actually run here, therefore operates through difficulty-focusing and label smoothing only — the class-frequency correction is already made upstream, by the sampler. §3.9.7 (cross-referenced rather than repeated here) explains why this rule exists and what happens when it is violated; the fact to carry forward is simply that the two mechanisms are not additive by default in this codebase.

### 2.5.5 Label smoothing

Label smoothing replaces the one-hot target with a softened one, mixing a fraction $\epsilon$ of uniform probability mass across all classes into the true-class target. Inside the focal-loss implementation this is a weighted combination of the ordinary negative log-likelihood term and a term averaging $-\log p_c$ over every class $c$, with $\epsilon = 0.05$. The effect is to discourage the model from driving the true-class logit arbitrarily high — regularisation against over-confidence, applied uniformly regardless of class, and distinct from a correction for imbalance. AdamW, mixed precision, gradient clipping, and label smoothing itself have no dedicated treatment in this project's `docs/` corpus and are used here as standard practice.

### 2.5.6 Augmentation under scarcity

With few labelled clips per class, augmentation acts before the loss or the sampler is reached at all, and is applied to the training split only. Xia et al. (2020a) motivate this for micro-expression data directly: "temporal data augmentation strategies as well as a balanced loss are jointly used for our deep network" to address "limited and imbalanced training samples". Two transformations are used here (`Ablation_Study/dataset.py`): a random temporal crop of the input window (centred at evaluation), and a horizontal flip applied with probability one half. Because the input is optical flow rather than raw pixels (§2.2), the flip cannot be a plain mirror — reversing the scene horizontally reverses the sign of horizontal displacement — so the flip is paired with a negation of the flow tensor's $u$ (horizontal) channel. Flipping without negating $u$ would hand the network motion vectors pointing the wrong way for the geometry shown, a quiet corruption motion-tensor augmentation must specifically guard against.

### 2.5.7 Optimisation

Training uses AdamW, with decoupled weight decay applied uniformly to every parameter — no separate no-decay group for biases or normalisation terms. Learning rate is $10^{-4}$, weight decay $10^{-4}$. The schedule combines a short linear warmup with cosine annealing: `LinearLR` ramps the rate from a tenth of target to full over five epochs, after which `CosineAnnealingLR` anneals it to a floor of $10^{-7}$ over the remaining epochs; the two are joined by `SequentialLR` and stepped once per epoch, for 50 epochs in total. Cosine annealing is adopted elsewhere in the micro-expression literature on similar small-data grounds — Zhang et al. (2022) use it in a spatio-temporal transformer for this same task.

Gradient norms are clipped to $1.0$ after the AMP scaler's gradients are unscaled, via `clip_grad_norm_`, guarding against the occasional large update a small, skewed dataset can produce. Training runs under mixed precision (`torch.amp.GradScaler`, `autocast`), enabled whenever CUDA is available. Before each optimiser step, every parameter's gradient is checked for non-finite values; if any is found, the step is skipped entirely, gradients are zeroed, and a warning logged, rather than letting a corrupted update reach the weights. Model selection keeps the checkpoint with the highest validation macro F1, not accuracy, consistent with the metric argument of §3.9.5 that accuracy is a poor guide under skew; ties favour the later epoch. There is no early stopping — every configuration trains the full 50 epochs regardless of when its best epoch occurred.

One category of technique is absent by design: quantisation, pruning, and knowledge distillation are used nowhere in this project. Efficiency is measured rather than engineered here — §3.6 reports the compute cost of the architectural choices actually made.

---

## 2.6 Evaluating on a Small Corpus

Chapter 3 argues why leave-one-subject-out evaluation and pooled macro F1 suit this corpus, and what those choices cost here (§3.1.6, §3.1.8). This section gives the mechanism underneath: the confusion matrix and per-class precision, recall, F1; averaging scores versus pooling counts; the easily conflated distinction between averaging across folds and within them; what cross-validation buys and what leave-one-subject-out protects against; and, to close, an audit of what this thesis's own measurement apparatus does and does not supply.

### 2.6.1 The confusion matrix, and precision, recall, F1 for one class

For a single class $c$ treated as positive against all others, every prediction falls into one of four counts: true positives $TP_c$ (correctly predicted $c$), false positives $FP_c$ (predicted $c$ but truly something else), false negatives $FN_c$ (truly $c$ but predicted otherwise), and true negatives. Arranging all classes this way at once gives the confusion matrix: entry $(i,j)$ counts clips whose true label is $i$ and predicted label is $j$, so the diagonal holds correct predictions and every off-diagonal cell names a specific error — which true class is mistaken for which other.

From those counts, precision and recall isolate two failure modes:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}.$$
Precision penalises false alarms: it falls whenever the classifier calls $c$ on a clip that is not $c$, regardless of how many true $c$ instances it also catches. Recall penalises misses: it falls whenever a genuine $c$ instance is assigned elsewhere, regardless of how many false alarms accompany the hits. Neither failure mode is visible from the other, which is why F1, the harmonic mean,
$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$$
is the usual single-number summary: being harmonic, it stays low whenever either input is low, so a model cannot buy a good F1 by trading recall for precision.

### 2.6.2 Macro versus micro averaging

Combining per-class F1 scores into one number can be done two ways. Macro averaging computes $F1_c$ per class and takes an unweighted mean, $\text{Macro-F1} = \tfrac1C\sum_c F1_c$: every class counts equally regardless of size, so ignoring a rare class is penalised as heavily as ignoring a common one. Micro averaging instead pools the raw counts across classes first — summing every class's $TP$, $FP$ and $FN$ — and computes one F1 from those totals.

For single-label multi-class classification, where every clip gets exactly one predicted label, this pooling has a consequence worth stating plainly: every misclassification is simultaneously a false positive for the class it was wrongly assigned to and a false negative for its true class, so summed $FP$ and summed $FN$ equal each other and the total error count, and pooled micro-F1 reduces algebraically to $TP_{\text{total}}/N$ — plain accuracy. Micro-F1 and accuracy are not two different metrics here; they are the same number computed two ways. The contrast that actually matters is therefore not "macro versus micro" but **macro-F1 versus accuracy**: an unweighted per-class average against a count a majority class can dominate.

### 2.6.3 Pooled versus per-fold averaging

A further averaging choice sits underneath macro-F1 once evaluation is split into folds, and it governs how every result in this thesis is reported. One option accumulates the confusion counts — $TP_c$, $FP_c$, $FN_c$ — across every fold first, then computes one set of per-class F1 scores, and one macro-F1, from that pooled total. The other computes a complete macro-F1 inside each fold from that fold's own predictions, then averages the per-fold values. See et al. (2019) define the Unweighted F1 (UF1) of the MEGC 2019 protocol the first way: the macro-averaged F1 obtained by accumulating true positives, false positives and false negatives over all folds of a leave-one-subject-out run before computing and averaging per-class F1.

These are estimators of two different quantities, not two ways of writing the same one: pooled macro-F1 describes this class's F1 across the whole evaluation, once every held-out prediction sits in one confusion matrix together, while per-fold-averaged macro-F1 describes the typical per-fold macro-F1 — a quantity that depends on how classes happen to fall within each fold, not only on the classifier. When a fold's held-out set does not contain every class, its per-class F1 for the missing class is degenerate, and averaging such folds in with fully-populated ones changes what the number reflects for reasons unrelated to classifier quality; on a small, unevenly distributed corpus this is a live possibility, and fold composition can bound the per-fold estimator below what the pooled one would report. §3.1.6 and §3.1.8 work through how the fold structure of this study does exactly that; the point to carry here is only that the two quantities are not interchangeable, and which one a reported number is becomes apparent only once its computation is stated.

### 2.6.4 Cross-validation and subject-disjointness

$k$-fold cross-validation partitions the data into $k$ disjoint subsets, trains $k$ models — each on $k-1$ folds — and evaluates each on the one fold it never saw, so every example is scored exactly once by a model that never trained on it. Leave-one-subject-out (LOSO) is the case where $k$ equals the number of subjects, and each fold's held-out set is precisely one subject's clips, every other subject's clips forming that fold's training set.

The property this buys is subject-disjointness: no fold's training and validation sets ever share a subject. Without it, a model evaluated on a clip from a subject it has already trained on can succeed by recognising that person's face or recording conditions rather than the expression class itself — an identity leak inflating the score without the model having learned anything transferable to an unseen person. Enforcing subject-disjointness in every fold closes that channel.

### 2.6.5 What this apparatus does and does not provide

The pipeline behind every result in this thesis computes, per fold, a confusion matrix and per-class precision, recall and F1 via scikit-learn; across folds it reports `accuracy` and `macro_f1` in `final_results.json` as the **mean of the per-fold values**, while summing the confusion matrices and computing `per_class_f1` and `micro_f1` from that **summed** matrix. Two consequences follow directly, stated without softening. First, the pooled macro-F1 — the UF1 quantity of §2.6.3 — is never computed by the training code itself; the field named `macro_f1` in its output is the per-fold average, not the pooled quantity, which must be derived afterwards as the mean of the stored `per_class_f1` values. Second, `micro_f1` equals accuracy here, as the code's own comment records; unweighted average recall (UAR), the balanced-accuracy counterpart to UF1, is never computed at all, though the per-class recall values it would be built from are stored.

The LOSO folds enforce subject-disjointness by construction — a held-out subject's clips contribute nothing to that fold's training set — and the holdout variant asserts explicitly that its train and validation subject sets are disjoint. One property of the apparatus falls short of an analogous guarantee and must be named as a limitation rather than left implicit: there is no inner validation split. The held-out subject's fold is used both to select the best training checkpoint and as the final scored set for that same fold, so a reported fold result is not blind to the data it is checked against, as a nested train/validation/test design would keep it. This is an optimistic bias of unknown size in every reported number, a property of the apparatus itself rather than of any one configuration.

Finally, no confidence interval, significance test or variance estimate is computed anywhere in the pipeline, and per-clip predictions are not saved — only aggregate metrics, the confusion matrix, and training curves persist. The consequence is concrete: with no per-clip prediction record surviving, a paired significance test between two configurations cannot be constructed from the stored artefacts, whatever the aggregate scores say.

---

## References

*Every source listed is a paper held in the project's `docs/` corpus. Standard machine-learning background — the optical flow constraint equation, batch normalisation, dropout, three-dimensional convolution, scaled dot-product attention, AdamW, label smoothing, gradient clipping, mixed-precision training, and the definitions of precision, recall, F1 and k-fold cross-validation — is presented without citation, since no source for it is held in the corpus. Works cited only inside the reviewed papers — Ekman and Friesen on nonverbal leakage, the FACS manual, the original Eulerian magnification formulations of Wu et al. and Wadhwa et al., the Farnebäck flow estimator, the Transformer of Vaswani et al., and the focal loss of Lin et al. — are attributed in the text to the paper that reports them and are not listed here.*

---

Bai, M., Goecke, R., & Herath, D. (2021). Micro-expression recognition based on video motion magnification and pre-trained neural network. *IEEE International Conference on Image Processing (ICIP 2021)*, 549–553.

Li, Y., Huang, X., & Zhao, G. (2018). Can micro-expression be recognized based on single apex frame? *IEEE International Conference on Image Processing (ICIP 2018)*.

Liong, S.-T., Gan, Y. S., Yau, W.-C., Huang, Y.-C., & Tan, L. K. (2019a). OFF-ApexNet on micro-expression recognition system. *Signal Processing: Image Communication*, 74, 129–139.

Qu, F., Wang, S.-J., Yan, W.-J., & Fu, X. (2016). CAS(ME)²: A database of spontaneous macro-expressions and micro-expressions. In M. Kurosu (Ed.), *Human-Computer Interaction, HCI 2016, Part III*, LNCS 9733 (pp. 48–59). Springer.

See, J., Yap, M. H., Li, J., Hong, X., & Wang, S.-J. (2019). MEGC 2019 — The second facial micro-expressions grand challenge. *14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019)*.

Shreve, M., Godavarthy, S., Goldgof, D., & Sarkar, S. (2011). Macro- and micro-expression spotting in long videos using spatio-temporal strain. *IEEE International Conference on Automatic Face and Gesture Recognition (FG 2011)*, 51–56.

Xia, Z., Hong, X., Gao, X., Feng, X., & Zhao, G. (2020a). Spatiotemporal recurrent convolutional networks for recognizing spontaneous micro-expressions. *IEEE Transactions on Multimedia*, 22(3), 626–640.

Yan, W.-J., Li, X., Wang, S.-J., Zhao, G., Liu, Y.-J., Chen, Y.-H., & Fu, X. (2014). CASME II: An improved spontaneous micro-expression database and the baseline evaluation. *PLoS ONE*, 9(1), e86041.

Yang, L., Zhang, R.-Y., Li, L., & Xie, X. (2021). SimAM: A simple, parameter-free attention module for convolutional neural networks. *38th International Conference on Machine Learning (ICML)*.

Zhang, L., Hong, X., Arandjelović, O., & Zhao, G. (2022). Short and long range relation based spatio-temporal transformer for micro-expression recognition. *IEEE Transactions on Affective Computing*, 13(4), 1973–1985.

Zhao, S., Tao, H., Zhang, Y., Xu, T., Zhang, K., Hao, Z., & Chen, E. (2021). A two-stage 3D CNN based learning method for spontaneous micro-expression recognition. *Neurocomputing*, 448, 276–289.
