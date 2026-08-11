# Speaker Notes — Final Thesis Presentation

**Deck:** `MER_Thesis_Final_Presentation.pptx` (13 slides) · **Target length: 15 minutes** · Leave 5 minutes for questions.

For every slide this document gives you four things:

- **Purpose** — why this slide exists in the argument. Read this while preparing, not while presenting.
- **On screen** — what you can point at.
- **Say this** — a word-for-word script. It is written to be spoken, not read silently. Say it in your own words once you know it; the wording is here so you are never stuck.
- **Then move on** — the sentence that gets you to the next slide.

---

## Before you start

### The three things your audience must remember

If they forget everything else, they should walk out with these:

1. **75 % accuracy under full leave-one-subject-out** — ten points above the best published result on this dataset.
2. **Of my four components, the Transformer does nearly all the work** (+0.217 macro F1, positive in all six matched comparisons). The 3D-CNN is worth −0.031 and consumed 97 % of the compute.
3. **How you evaluate decides what you conclude.** The same twelve configurations, evaluated five different ways, produced five different winners — two of them architecturally opposite.

Say each of these at least twice during the talk. Repetition is what survives.

### Timing plan

| Slide | Topic | Target | Running total |
|---|---|:--:|:--:|
| 1 | Title | 0:40 | 0:40 |
| 2 | The problem and the data | 1:00 | 1:40 |
| 3 | The four technologies | 1:10 | 2:50 |
| 4 | Baseline and the ablation ladder | 1:30 | 4:20 |
| 5 | How everything is tested (LOSO) | 1:30 | 5:50 |
| 6 | Which number is the number | 1:20 | 7:10 |
| 7 | All 12 results | 1:25 | 8:35 |
| 8 | config_8 vs config_2 | 1:20 | 9:55 |
| 9 | What each component contributed | 1:25 | 11:20 |
| 10 | Per-class behaviour | 0:55 | 12:15 |
| 11 | Cost | 0:45 | 13:00 |
| 12 | Literature and protocol history | 1:15 | 14:15 |
| 13 | Conclusions and caveats | 1:10 | 15:25 |

**If you are running late**, cut slide 10 (per-class) and slide 11 (cost) down to one sentence each — "nothing abandons a class, and the CNN ate 97 % of my compute for a negative contribution" — and go straight to slide 12. Never cut slides 5, 6, or 12; they are what makes the result defensible.

### Delivery reminders

- **Slow down on slides 4, 5 and 6.** These are definitions and method. Everyone can follow them if you go slowly; nobody can follow them if you rush.
- **Never say a number without saying what it means.** "0.6659" means nothing to a listener. "Two-thirds — where always guessing the common answer gets you a quarter" means something.
- **Own the awkward result on slide 8 rather than defending it.** Examiners respect the candidate who raises the problem first.
- **Point at the screen** when you say "look at this column". Do not assume they found it.
- **Pause after each of the three key messages.** Silence is what marks something as important.

---

# Slide 1 — Title

**Purpose.** Establish the topic and put your headline number in the room immediately, so everything that follows is evidence for a claim they have already heard.

**On screen.** Title, the four big stat numbers (156 / 25 / 300 / 50), and the headline bar.

**Say this.**

> Good morning, and thank you for coming. My thesis is on micro-expression recognition — teaching a model to read those split-second flickers of emotion on a person's face that people can't control.
>
> The short version of what I did: I took four techniques the literature suggests should help, and I tested every meaningful combination of them — twelve configurations — under the strictest evaluation protocol available for a dataset this small. That came to three hundred separate model trainings and about fifty GPU-hours.
>
> The headline is seventy-five per cent accuracy, which is ten points above the best published result on this dataset. But the more interesting finding is *which* of my four components actually earned its place — and I'll get to that.

**Then move on.** "Let me start with what the problem actually is."

---

# Slide 2 — The problem and the data

**Purpose.** Make a non-specialist understand the task in thirty seconds, and plant the class imbalance — 99 negative clips out of 156 — because every result later has to be read against it.

**On screen.** The explanation text on the left; the dataset table and the three class-size boxes on the right.

**Say this.**

> A micro-expression is a flicker of emotion on someone's face that lasts a fraction of a second. People can't control them, which is why they're interesting — in deception research, in clinical settings, in anything where what someone says and what they feel might differ.
>
> My model has to sort each clip into three buckets: negative, positive, or surprise.
>
> Now, the one thing to remember from this slide: **the model never looks at the actual video.** I convert everything to motion first — where things moved, and how the skin stretched. Why? Because with only 156 clips, if I gave it real pixels it would just memorise the twenty-five faces instead of learning what an expression looks like. Motion doesn't care who you are or how the room is lit.
>
> One more thing, and it comes back to bite us later — look at the imbalance. Ninety-nine negative clips, but only twenty-five surprise. Hold on to that number, ninety-nine out of 156.

**Then move on.** "So that's the input. Here's what I built on top of it."

---

# Slide 3 — The four technologies

**Purpose.** Introduce the four switches, and plant the parameters-versus-cost inversion that becomes the punchline on the cost slide.

**On screen.** Four lettered cards on the left (A through D); the pipeline diagram on the right; the "odd fact" box underneath it.

**Say this.**

> Four techniques, four on/off switches. That's the whole experiment.
>
> **EVM** is a magnifying glass for movement — it makes tiny facial twitches ten times bigger before I measure anything. Important detail: it happens *before* training, so it changes the data, not the model. That's why the diagram shows Stage 1 running twice — once magnified, once raw — and the EVM switch just picks which folder to read from.
>
> **SimAM** is a spotlight. It works out which parts of the face are statistically unusual and turns up their importance — and it's free, it adds no weights to learn. With 156 clips, free is exactly what I want.
>
> **The 3D-CNN** looks at shapes — small patches of space and time together. **The transformer** looks at the story over time — it sees all thirty-two frames at once.
>
> And one odd detail worth remembering, bottom right: the transformer is *huge* in size but almost free to run. The CNN is tiny but it ate ninety-seven per cent of my GPU time. That flips the way you'd expect, and it matters later.

**Then move on.** "Now — twelve configurations, and where they all start from."

---

# Slide 4 — The baseline and the ablation ladder

**Purpose.** This is the slide that makes your study *an ablation study* rather than a pile of results. Two jobs: establish that config_4 (EVM only) is the baseline, and establish the ladder ordering that every later table uses. Do not rush it.

**On screen.** Left: the baseline architecture in code, and the config_4-vs-config_1 table. Right: the twelve-row ladder table, Group A and Group B.

**Say this.**

> Four switches would give sixteen combinations, but four of them are nonsense — SimAM re-weights the CNN's output, so with the CNN off there's nothing to re-weight. That leaves twelve. All twelve were run.
>
> Now the framing that matters, and I want to be precise about it. **My baseline is config_4 — EVM switched on, nothing else.** EVM is not one of the things I'm testing away; it's part of my data pipeline. It's the starting point. Everything else in this study is that baseline *plus* network components.
>
> The baseline model itself is deliberately simple — you can read it on the left. It shrinks each frame to a blurry four-by-four thumbnail, averages all thirty-two frames into one, and runs a linear classifier. About five thousand numbers to learn, against nearly four hundred thousand for the full model. And because it averages the frames, it literally cannot tell you whether a smile was starting or ending.
>
> The column next to it, config_1, is the *same model with EVM switched off*. That's not a second baseline — it's the control that isolates EVM. The gap between the two, about half a point, is the cleanest EVM measurement I have.
>
> Last thing, and it governs every table from here on. **I order my tables as a ladder, not by config number.** Start at the baseline, add the CNN, add the transformer, add SimAM — walking down Group A is my proposed model being assembled one piece at a time. Group B is the exact same ladder with EVM switched off, lined up row for row, so any pair of rows tells you what EVM did at that rung.

**Then move on.** "Before any of those numbers mean anything, I need to tell you how they were measured."

---

# Slide 5 — How everything is tested (LOSO)

**Purpose.** Justify the evaluation protocol. This is the slide that earns you the right to compare against published work — and it sets up the metric problem on slide 6.

**On screen.** Left: why a simple split fails, then the four numbered steps. Right: the 25-fold schedule diagram. Bottom: the fixed-settings table.

**Say this.**

> Normally you'd hold back a chunk of your data and test on it. With only twenty-five people, that doesn't work — and it fails in two ways.
>
> First, your score depends on *which* faces you happened to hold back. Faces differ enormously. Draw easy subjects and the number flatters you; draw hard ones and it's pessimistic — and nothing tells you which happened. Second, if clips from the same person land in both halves, the model can just memorise that face.
>
> So instead I refuse to pick a split at all. Hold out **one person**. Train a completely fresh model on the other twenty-four. Test on that one person. Then **delete the model** and do it again for the next person. Twenty-five times.
>
> At the end, every single one of my 156 clips has been predicted by a model that had never seen that person's face. There's no lucky split to argue about, and cheating is impossible by construction. The cost is that I train twenty-five models instead of one — which is where the fifty GPU-hours went.
>
> One thing to notice in the picture on the right: the folds are very lopsided. Subject seventeen alone supplies thirty-three of the 156 clips. Three subjects supply one clip each. And ten of the twenty-five folds contain only a single emotion class — that causes a measurement problem, which is exactly the next slide.
>
> Everything in the table at the bottom is held identical across all twelve configurations, with the seed re-applied at the start of every fold. The runs are bit-for-bit reproducible.

**Then move on.** "So — which number should you actually look at?"

---

# Slide 6 — Which number is the number

**Purpose.** Prevent misreading. Two of the four numbers in your results files are booby-trapped, and one of them is mathematically incapable of hitting your own target. Saying this yourself, before an examiner finds it, is worth a great deal.

**On screen.** Left card: the two metrics you use. Right card: the two you refuse. Bottom: the floors table, and the always-Negative warning.

**Say this.**

> One minute of definitions, and then we're into results. Bear with me, because this is the slide that stops everything else being misread.
>
> **Accuracy** is easy: out of 156 clips, how many did it get right.
>
> **Macro F1** is the one that actually decides which model is better. It scores each of the three emotions separately, then averages them *equally* — so a model that quietly ignores the rare surprise class gets punished, even if its overall accuracy looks fine.
>
> Here's why that matters. If I just always guessed "negative", I'd score sixty-three per cent accuracy — because ninety-nine of the 156 clips *are* negative. That sounds respectable. But macro F1 gives that same trick **0.26**. That's the gap between the two numbers, and it's why I never quote accuracy on its own.
>
> Last thing. Two of the four numbers in my results files are computed per-fold and then averaged, and both are broken. One of them over-rates my proposed model by six points. The other is *mathematically capped* at 0.627 — which is below my own target of 0.68 — because ten of my folds contain only one class, so two of the three scores are forced to zero in those folds. Not even a perfect classifier could reach my target on that column. So I don't use it.

**Then move on.** "Right — the results."

---

# Slide 7 — All 12 results

**Purpose.** The main results table. The job here is to stop them reading twelve rows and instead read the *pattern*: climbing the ladder, and the transformer split.

**On screen.** The ladder table (Group A on top, Group B below); three summary cards on the right.

**Say this.**

> Twelve configurations, all twenty-five folds, all 156 clips — ordered the way the experiment was designed rather than by config number.
>
> Group A on top is the ladder with EVM. Reading down: baseline, add the CNN, add the transformer, add SimAM. That walk is my proposed model being assembled. The last two rows of the group are the side branches — what happens if you leave the transformer out, and what happens if you leave the CNN out.
>
> Group B is the identical ladder with EVM switched off, lined up row for row.
>
> Now watch the far-right column as you climb. Add the CNN — **nothing**, less than a hundredth of a point. Add the transformer — **plus 0.22**. Add SimAM — three thousandths.
>
> And here's the pattern across both groups. Everything containing a transformer scores between 0.58 and 0.71. Everything without one scores between 0.42 and 0.45. **No overlap, twelve out of twelve, no exceptions.** That is the single clearest signal in my entire thesis.
>
> Two rows to notice before I move on. My full model, config_8, takes the best accuracy in the study — seventy-five per cent, 117 clips. But the transformer *on its own*, at the bottom of group B, takes the best macro F1 — and it's the only configuration to clear both of my targets.

**Then move on.** "That's awkward for a thesis that proposes four components — so let me deal with it head-on."

---

# Slide 8 — config_8 vs config_2

**Purpose.** Confront the fact that your simplest transformer configuration matches your proposed model. Handled well, this is the most credibility-building slide in the deck. Handled defensively, it is the most damaging.

**On screen.** The two-column comparison table; the two explanation cards; the berry bar at the bottom with the confidence interval.

**Say this.**

> I'd much rather present this than have someone find it.
>
> My four-component model gets 117 clips right. The transformer on its own gets 116. **One clip.** With only 156 clips, the ninety-five per cent confidence interval is about plus or minus seven points — so statistically these two are the same model, and I'm not going to claim otherwise.
>
> What genuinely differs is *how* they're right. My model is a specialist in the common class — eighty-five of the ninety-nine negatives, the best negative score anywhere in the study. Accuracy loves exactly that behaviour, because negatives are most of the dataset, which is why it takes the accuracy crown. But it only recovers fifteen of the thirty-two positive clips — and that single weakness is what costs it the macro-F1 target, which it misses by fourteen thousandths.
>
> The transformer alone spreads its errors much more evenly across all three emotions. And it does that on seven per cent of the computing power and under one per cent of the memory.
>
> One more thing I want on the record. In my *earlier* experiments this same model scored a flat zero on the positive class — it never predicted it once. That turned out to be a bug in how I was correcting the class imbalance, not a limit of the architecture. Fixed here.

**Then move on.** "So which component is actually doing the work? That's what the ablation was for."

---

# Slide 9 — What each component contributed

**Purpose.** The scientific core. Every number here comes from matched pairs, which is the only fair way to attribute an effect. Also where you disclose the EVM defect you found and repaired.

**On screen.** The four-row contribution table; the transformer-split figure bottom left; three explanation cards bottom right.

**Say this.**

> Every number on this slide comes from a **matched pair** — two configurations that are identical except for one switch. That's the only fair way to say what a component is worth.
>
> **The transformer: plus 0.217**, and positive in all six comparisons, without exception. The figure shows it — the group with a transformer and the group without are separated by a gap that is completely empty.
>
> Why does it work? It's the only piece that sees all thirty-two frames at once. A micro-expression is a little story — neutral, peak, back to neutral. The transformer can compare the beginning with the middle. Everything else just averages the frames together, which wipes out the peak entirely.
>
> **The 3D-CNN: minus 0.031.** Negative — and it consumed ninety-seven per cent of my GPU budget. I'll be honest about why: the input is *already* a motion representation, so I'm asking a convolutional feature extractor to redo work that optical flow already did — and to learn it from 156 clips.
>
> EVM and SimAM sit near zero on average. But the averages hide something real. Look at config_9: it catches only six of twenty-five surprise clips. Add SimAM, and surprise goes from 0.30 to 0.59. Add EVM instead, and it goes to 0.65. Both of the components that look useless on average fix the *same specific failure*.
>
> And one correction to my own earlier work, bottom right. In every previous run, EVM was doing literally nothing — both settings were reading the same files, so half my matrix was duplicated. I found that and fixed it. **This is the first time EVM has actually been measured in this project.**

**Then move on.** "Let me show you how that plays out per emotion."

---

# Slide 10 — Per-class behaviour

**Purpose.** A short slide. It closes off the "does your model just ignore the rare classes?" question before it can be asked.

**On screen.** The per-class ladder table; three cards on the right.

**Say this.**

> This is the same ladder, but scored per emotion.
>
> The important result here is a negative one: **none of my twelve models gives up on a class.** The worst single score anywhere in the whole table is 0.246. In my earlier experiments the proposed model scored a flat zero on positive — never predicted it once. That's completely gone.
>
> And a warning about reading precision on its own: one of my models has *perfect* precision on the negative class. Sounds impressive. It got there by only ever risking that label twenty-one times out of 156. Its recall is 0.21. That's not a good model — that's a model refusing to answer.

**Then move on.** "Briefly, what all of this cost."

---

# Slide 11 — Cost

**Purpose.** One point, forty-five seconds. The best model is nearly the cheapest, and the expensive component is the one that doesn't work.

**On screen.** The cost ladder table; three cards on the right.

**Say this.**

> One point on this slide.
>
> The best-scoring configuration I ran is also nearly the **cheapest** one I ran — top macro F1 in about half an hour of GPU time and a hundred and seventy megabytes of memory. My proposed model got one more clip right for thirteen times the time and a hundred and fifteen times the memory.
>
> And the cost of climbing the ladder is entirely one rung: going from the baseline to *plus the CNN* costs fourteen times the time and ninety times the memory, for nine thousandths of a point. Adding the transformer on top of that is free by comparison — and buys **plus 0.21**.

**Then move on.** "So how does this compare to everyone else — and to my own earlier attempts?"

---

# Slide 12 — Literature and protocol history

**Purpose.** Two jobs. Place your result against published work, and then deliver what is arguably the most transferable finding in the thesis — that the evaluation protocol, not the architecture, was driving your earlier conclusions.

**On screen.** Top left: the literature table. Right: the five-protocol history table and the explanation card.

**Say this.**

> Against published work: seventy-five per cent accuracy is ten points above the best comparable result on this dataset, and five points above my own target. And config_2 clears both targets outright.
>
> The word that matters there is *comparable*. My earlier report couldn't make this comparison at all, because its best number came from one small thirty-nine-clip split, while every published number is leave-one-subject-out — that would have been comparing an easier task to a harder one. That objection is gone now. I should also flag honestly that these literature rows still need checking against the primary papers, and none of them report macro F1.
>
> Now the table on the right, which I'd argue is the most important thing in this talk.
>
> These are the **same twelve configurations**, evaluated five different ways over the life of this project. **The winner is different every single time.** Under the old method, the best model had no transformer at all. Under proper leave-one-subject-out, the best model is the transformer on its own. Those are architecturally *opposite* conclusions — from identical code.
>
> Why did it flip? The old test set had thirty-nine clips and contained *exactly one* surprise example. That one clip was worth a third of the score. Getting it right isn't a measurement of architecture — it's a coin flip weighted at thirty-three per cent.

**Then move on.** "Let me finish with what I can claim, and what I can't."

---

# Slide 13 — Conclusions and caveats

**Purpose.** Land the three key messages, state your limits before anyone asks, and end on the transferable finding rather than on a limitation.

**On screen.** Achievements on the left, caveats on the right, the teal bar at the bottom.

**Say this.**

> To wrap up.
>
> I hit my targets, using an evaluation method strict enough to actually support the claim. Seventy-five per cent accuracy — ten points above the best published result on this dataset.
>
> But the more interesting finding is that of my four components, **one does nearly all the work**. The transformer is worth 0.217. The CNN is worth minus 0.03, and cost me ninety-seven per cent of my computing budget.
>
> Now the limits, stated plainly rather than buried. My top two configurations differ by **one clip**, so I can't claim one beats the other — only that anything with a transformer beats anything without one. I ran a **single seed**, so I have proven determinism but no error bars; treat any difference under about five points as unresolved. And because I only saved aggregate confusion matrices rather than per-clip predictions, **I can't run the statistical test that would settle the first point**. That fix costs essentially nothing and is first on my list.
>
> If you take one thing away, make it the line at the bottom. On a dataset this small, **a single train/test split will pick a winner essentially at random** — and I have five evaluation regimes on identical code proving it picks architecturally *opposite* winners depending on the draw. That's a finding about how this field should evaluate, not just about one pipeline.
>
> Thank you. I'm happy to take questions.

---

# Question preparation

Answer in this shape: **direct answer first, one sentence of evidence, then stop.** Do not keep talking. If you don't know, say "I don't know — here's how I'd find out."

### "config_2 beats your proposed model. Isn't your thesis a failure?"

> No — and I'd push back on "beats". They differ by one clip out of 156, and the confidence interval is plus or minus seven points, so they aren't distinguishable. What my ablation actually establishes is *which component is responsible*, and that's the transformer. A finding that three of my four components don't pay for themselves is a real result, and it's only visible because I ran the full matrix rather than just the proposed model.

### "Why only a single seed? Doesn't that invalidate the comparisons?"

> It limits them, and I say so on the final slide. It's why I refuse to claim any difference under about 0.05 macro F1. What I do have is bit-exact reproducibility — I have a duplicate results directory with byte-identical metrics. Multi-seed runs are the first thing I'd spend compute on, and I'd take it from the 3D-CNN, which used forty-nine of my fifty GPU-hours for a negative contribution.

### "Why can't you report statistical significance?"

> Because I saved aggregate confusion matrices per configuration, not per-clip predictions. With per-clip predictions I could run a McNemar test on the config_2 versus config_8 pair and settle it directly. It's a one-line change to the results writer and costs nothing — it's my top recommendation for the next iteration.

### "Did you actually tune the 3D-CNN? Maybe it just needs better hyper-parameters."

> That's a fair challenge, and I can't rule it out. What I can say is that it was given the same budget, schedule and data as every other configuration, and its failure has a mechanical explanation rather than a tuning one: the input is already an optical-flow representation, so the spatio-temporal features it would learn have largely been extracted already — and 156 clips is very little data to learn a convolutional extractor from scratch. A fairer test would be pre-training it on a larger corpus, which I didn't have.

### "Why group seven emotion labels into three?"

> Because the original labels are unusably sparse — fear has two clips in the entire dataset and sadness has seven. You cannot do subject-disjoint cross-validation on a two-clip class. Grouping by emotional valence raises the smallest class from two to twenty-five, and it matches the three-class setup the published baselines use, which is what makes the comparison on slide 12 valid.

### "Why is your baseline EVM rather than nothing at all?"

> Because EVM is part of my data pipeline, not part of the network under test. It runs offline, before optical flow. So the honest baseline for an *architecture* ablation is "my data pipeline, simplest possible classifier" — that's config_4. I do also run the EVM-off version, config_1, but its role is as a control that isolates EVM, and the gap between the two is the cleanest EVM measurement in the study.

### "Your accuracy is 75 %, but always guessing 'negative' gets 63 %. Is 12 points impressive?"

> On accuracy alone, that's the right scepticism, and it's exactly why I don't quote accuracy alone. The trick model scores 0.26 macro F1 and mine scores 0.67 — that's the comparison that reflects actual capability, because macro F1 refuses to reward ignoring the rare classes. And none of my twelve configurations abandons a class.

### "Would this work on real-world video, outside CASME-II?"

> I can't claim that from this study — everything here is lab-recorded at 200 frames per second with the onset and offset labelled. What I would say transfers is the architectural finding: the temporal model is what matters, and it's the cheapest component to run. Testing on a second dataset — SAMM or CAS(ME)² — is the obvious next step, and cross-dataset evaluation would be a stronger claim than anything in this thesis.

### "Why focal loss *and* a balanced sampler? Isn't that double-correcting?"

> That's precisely the bug I found. In the earlier runs I had inverse-frequency class weights in the loss *and* no sampler, and the model collapsed to predicting a single class — my proposed model scored zero on positive. In this run the balanced sampler is on and loss class-weighting is deliberately disabled. Focal loss stays because it addresses a different problem — down-weighting easy examples — not the imbalance itself.

### "How do you know the EVM switch is working now, given it was broken before?"

> Two ways. First, all six matched pairs now produce different numbers, where previously they were identical to four decimal places — including identical confusion matrices, which can't happen from training noise. Second, the confusion matrices themselves differ, so genuinely different tensors are being read. The repository has a verification helper for exactly this check, and my recommendation is to wire it in as a startup assertion so the defect can't recur silently.

### "What would you do differently?"

> Four things, in order. Save per-clip predictions so significance testing is possible. Run multiple seeds — funded by dropping the 3D-CNN. Assert the EVM tensor directories at startup. And test on a second dataset. The first three cost almost nothing; I simply didn't know I needed the first one until the results were in.

---

## The one-sentence version

If someone stops you in a corridor and asks what your thesis found:

> Under the strictest evaluation available, my pipeline hits 75 % on micro-expression recognition — ten points above published work — but the ablation shows the temporal transformer does nearly all the work, and the expensive convolutional component actually hurts.
