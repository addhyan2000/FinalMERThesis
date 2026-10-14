# Chapter 1 — Introduction

In the moment after a difficult question lands, a face can give something away. A tightening at the brow, a pull at one corner of the mouth — there and gone again before the person has registered producing it, and at odds with the composed expression held over the top of it. Movements of this kind are micro-expressions: brief, involuntary facial movements that betray an emotion its owner is working to keep hidden [1]. They last under half a second, and they are faint enough that an observer watching in real time will ordinarily miss them entirely.

That combination — informative, and effectively invisible — is what has drawn sustained effort towards recognising them by machine. Yan et al. [1] motivate the CASME II corpus on the grounds that a robust automatic recogniser "would have broad applications in national safety, police interrogation, and clinical diagnosis"; the capability is likewise reported as promising for lie detection, business negotiation and psychoanalysis [9]. Those are the field's motivating claims rather than records of anything deployed, but they are why the problem has kept its audience.

The systems built to attempt it are not single models. They are pipelines, assembled stage by stage from techniques that arrived in the field separately and are now adopted together. This thesis takes one such pipeline apart and measures its stages one at a time.

§1.1 sets out the practical problem that makes such a measurement worth making, and the form of study built to produce it. §1.2 describes how the remainder of the thesis is organised.

---

## 1.1 Research Motivation

Building a system that recognises these movements automatically is not a matter of pointing a classifier at a face. Three properties of the phenomenon work against it. The movement is short, so only a handful of video frames carry whatever distinguishes one emotion from another. It is faint, so what those frames carry sits close to the noise floor of ordinary video. And it cannot be produced on instruction, so the recordings that contain it have to be elicited under controlled laboratory conditions and are small by construction.

Systems built for the task have responded to those constraints in a consistent way. Rather than a single model reading raw video, they are assembled as a sequence of stages: the small motion is amplified, the sequence of pixel intensities is re-expressed as a description of movement, and the clip is resampled to a fixed duration before any learned component sees it; a spatial extractor then reduces each frame, and a temporal model reads the resulting sequence. Each stage entered the field with its own justification and its own reported gain, and each tends to be adopted along with the rest.

That arrangement is manageable while a pipeline is being published and awkward as soon as one is being built. Anyone assembling this pipeline works against a finite budget — of compute, of implementation effort, of the patience required to tune a preprocessing constant that has no principled setting — and has to decide which stages deserve that budget. The published record does not settle the question, because a component is almost always reported from inside a complete system that differs from its predecessor in more than one respect, so the reported improvement cannot be separated from everything else that changed alongside it. Chapter 3 makes that argument component by component and states the resulting gap at §3.10.3.

The aim of this thesis is to attribute the performance of a standard micro-expression recognition pipeline to its individual stages, by measuring what each one separately contributes under a single fixed protocol.

The study is accordingly designed as a factorial ablation. Four components — Eulerian magnification, a convolutional spatial stem, parameter-free attention, and a temporal transformer — are switched on and off independently, giving sixteen possible combinations of which twelve are architecturally valid and are trained. Each configuration is trained from scratch, with no pre-training and no auxiliary data, which is the regime a single small corpus actually affords. Each is then evaluated under complete 25-fold leave-one-subject-out validation on CASME II: 156 clips once the residual `Others` category is set aside, grouped into three classes carrying 99, 32 and 25 examples. One fold is held out per subject, so no subject's clips ever appear in training and test data together. Every factor not under variation is held identical across the twelve runs, and all twelve are scored by the same primary metric, pooled macro F1.

What that design yields is a set of matched pairs. Any two of the twelve configurations that differ in exactly one component bracket that component with everything else fixed, so the difference between their scores estimates what it contributed. Read across every such pair, the twelve runs return a marginal contribution for each of the four stages, measured on one corpus, under one protocol, with a single metric. The four stages are not found to contribute equally; Chapter 5 takes each of them in turn.

---

## 1.2 Structure of the Thesis

The thesis proceeds in six chapters, each taking the pipeline as its organising object and treating it from a different side.

- **Chapter 1 — Introduction** introduces the phenomenon and the practical difficulty that motivates the work, states the question the thesis asks, and outlines the design assembled to answer it.

- **Chapter 2 — Background** sets out the mechanisms the thesis depends on, in the order the pipeline applies them: the phenomenon itself, what the recording supplies, magnification and motion representation, neural-network fundamentals and the blocks built from them, training under class skew, and evaluation on a small corpus.

- **Chapter 3 — Literature Review** reviews the literature bearing on each pipeline component as a single continuous argument, closing at §3.10 with the consolidated research gap and every declared divergence from the reviewed methods.

- **Chapter 4 — Methodology** specifies the study as carried out: the experimental design, the corpus and its preparation, the twelve configurations, the architecture and its ablatable components, the training procedure, the evaluation protocol and the computational environment — together with §4.6.7's account of the earlier evaluations that are excluded, and why.

- **Chapter 5 — Results** reports what the ablation found, taking each ablated component in turn over the matched pairs that isolate it, and then drawing the findings together against per-class behaviour and the published literature.

- **Chapter 6 — Conclusion** states the contributions, the limits within which they hold, the work the design leaves undone, and what a reader should take away.

---

