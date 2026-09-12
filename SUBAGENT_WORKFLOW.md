# How subagents are used on this thesis

A record of the working method, written after four chapters. It supplements `AGENT_PROMPTS.md`, which holds the original five prompt templates; this document covers what changed once those templates met a large amount of real work.

---

## 1. The shape

Three stages, not five.

```
SCOPE CHECK  →  WRITER  →  VERIFY
(report only)   (edits)     (scripts + one narrow agent)
```

**Scope check.** Before any new section, an agent reads the code and reports the exact facts a writer will need — parameter values, tensor shapes, call order, file:line — and says plainly which proposed topics have **no footprint in the project**. This is the stage that pays for itself. It has caught: that the pipeline performs no face registration at all; that a proposed §2.2 would have duplicated three existing sections; that the batch size actually used was 4 while the config default is 2; that the code builds sixteen configurations, not twelve.

Skip it when the section is standard background with no project footprint — §2.5 (neural-network fundamentals) needed no scope check, because the real question was what §2.6 already assumed, answered by reading §2.6 for a few thousand tokens instead of dispatching an agent for ~160k.

**Writer.** One agent, one or two files, edited **one at a time**. Gets the verified facts as a flat list, an explicit no-duplication constraint naming what other chapters own, and the exact values to use.

**Verify.** Mostly scripts. See §2.

---

## 2. What moved from agents to scripts, and why

The original workflow had three audit agents per section: facts, references, duplication. Most of that work is mechanical, and a regex does it better than a reader — deterministically, in seconds, for nothing.

| Was an agent | Now |
|---|---|
| Cross-reference audit | `tools/ch3_check.py` — every `§x.y`, `Table n.m`, `Figure n.m` resolves |
| Citation audit | `tools/bibliography.py check` — every key exists, no orphans either direction |
| No-drift audit, mechanical half | `tools/ch3_verify.py` — every number, quotation ≥12 chars, heading and table row preserved against a git ref |
| No-drift audit, **judgment half** | **still an agent**, scoped to the newly written prose only |

This cut the Chapter 3 estimate from ~3.2M tokens to ~1.45M. It also made verification *more* reliable, not less: a script checked all ten sections identically every time, where an agent's attention varied.

**What cannot be scripted, and must stay an agent or a human:** does this cross-reference point at a section that actually says what the sentence claims? Does this new sentence introduce a claim the source text does not support? Is this heading honest? Those need reading.

---

## 3. The brief

A good brief is long. The best ones in this project ran 800–1,500 words and contained every value the writer needed, so the writer never had to recall or re-derive anything.

**Always include:**

- **Where the section sits** and what it is for, in one paragraph.
- **The verified facts as a flat list**, with values quoted exactly. Mark the load-bearing ones.
- **An explicit no-duplication constraint** naming the files that own what must not be restated — and telling the writer to read them.
- **Target length**, as a range.
- **Hard constraints** repeated every time: no commits, no `git add`, touch only these files, never the generated `00_Chapter*_Complete.md`, do not reflow paragraphs you are not otherwise changing.
- **Verification commands to run**, with expected output.
- **A report contract**: what to return, and what to write to a report file instead.

**The single most valuable line, used in every brief:**

> Before writing "no concerns", run a check that would disprove it.

Agents that were told this found real problems. Agents that were not returned "Concerns: None" while shipping defects — three rounds in a row, in one case.

**Two more that earned their place:**

> This section is already fact-audited. You are not re-verifying its claims. Any factual change is a defect, **including one that looks like a correction**. If you believe a fact is wrong, report it — do not fix it.

> If any number in this brief is contradicted by the source data, do not write it — report the discrepancy instead.

That second one is not a formality. See §4.

---

## 4. The briefs are not a trusted source either

Writers caught errors in the controller's own briefs **three times** while writing Chapter 5 alone:

- A brief claimed "the two cheapest configurations both carry a transformer". False — `config_1` (56.47 s) and `config_4` (58.12 s) are the two cheapest and neither has one. The writer checked the timing table, declined the sentence, and wrote the accurate version.
- A brief wrote "config_9 scores 0.7308, sixth of twelve" — conflating an accuracy value with a macro-F1 rank. By accuracy it ties third. The writer disambiguated rather than reproducing the confusion.
- A brief pasted §3.9's real bridge sentence as a "register example". The writer recognised it as another section's actual text and wrote a fitting one instead of duplicating it.

The lesson is not that the writers are excellent — it is that **dense numeric briefs are exactly where the controller makes mistakes**, and the instruction to check rather than comply is what surfaces them. Write briefs expecting to be wrong in them.

---

## 5. Reviewing what comes back

**Never accept a reported finding without checking it.** Roughly two in twenty-five findings have been wrong in this project — in both directions. Examples: an auditor reported a PDF missing from `docs/` that was present under a misleading filename; a verification grep of mine returned a false negative because it searched for "never invoked" while the text said "never called".

**Check the claim, not the report.** When an agent said a section stated something, the answer came from opening the file. When an agent said the code had no weight-initialisation routine, the answer came from a better grep — and the agent was right, my first grep was wrong.

**Watch for agents exceeding the brief.** One agent, told to add four masking patterns, added six — and the two extras hid six real citations and masked the very problems the check existed to surface. It took three fix rounds. Unrequested additions are not initiative; they are scope creep with a plausible rationale.

**Judge headings separately from bodies.** One writer produced a correct paragraph under the heading "Two baselines, not one" — the exact overclaim a prior audit had removed. Headings are what a skimming examiner reads.

---

## 6. Batching

**Two files per writer, not four.** A writer holding two files drifts far less, and a bad batch costs a redo of two sections rather than four. The extra tokens are worth it.

**Within a batch, one file at a time** — finish and report the first before opening the second. Context fills as a batch proceeds, and the later file is where drift appears.

**Verify per file, not in aggregate**, so a drifting file cannot hide behind a clean one. Revert a drifting file alone with `git checkout -- <file>`; never re-run a whole batch.

**At most three agents in one message.** Five heavy agents at once hit a rate limit and one died mid-task.

---

## 7. Cost, measured

| Role | Observed |
|---|--:|
| Scope check | 76k – 182k |
| Writer, two sections | 79k – 168k |
| Reviewer / auditor | 87k – 156k |
| Controller doing it directly | 5k – 20k |

Do it yourself when the work is deterministic and you can verify it entirely — a renumbering, a scripted substitution, four heading numbers and a bridge sentence. §2.1 took ~5k in-session against ~150k for a dispatch, for four numbers and one sentence.

Dispatch when the work needs judgment across material you would otherwise have to read into context — writing prose from verified facts, auditing a 29,000-word chapter for duplication, checking 46 cross-references against their targets.

**A rate limit is not necessarily lost work.** One figure-generation agent died after 61 tool calls; checking the filesystem showed it had completed everything and failed only while writing its report. Check the state before re-dispatching.

---

## 8. What is never delegated

- **Deciding what to build.** Structure, scope, and which option to take.
- **Ruling on a conflict** between a finding and the plan.
- **Applying fixes after a review.** The reviewer reports; the controller decides and edits. An auditor that edits produces silent changes you cannot weigh.
- **The final judgment on prose that will carry someone's name.** Every section in this thesis was read after the agent finished, and roughly one in three needed a correction the agent did not flag.
