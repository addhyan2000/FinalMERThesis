# Subagent prompt templates

The five agent roles used to produce Chapters 2 and 3. Replace every `{{PLACEHOLDER}}`.
Referenced from `HANDOVER.md` §5.

---

## The orchestration pattern

Per section, four stages. Stages 1 and 2 are sequential; stage 3's three audits run **in parallel** (one message, three tool calls).

```
1. SCOPE CHECK      → is this actually in the project? what are the exact facts?
2. WRITER           → write from those facts only
3. AUDITS (parallel) → (a) facts  (b) references  (c) duplication
4. ORCHESTRATOR      → apply fixes, stamp verified, print
```

**Rules that made this work.**
- Auditors get `Report only — do NOT edit`. Only the writer and the orchestrator write. Prevents two agents fighting over one file.
- Every audit prompt carries the **metric warning** (§1 of `HANDOVER.md`). One audit reported a correct figure as an error because it read the wrong JSON key.
- Every audit asks for a `VERDICT: ACCURATE / NEEDS FIXES (n)` line and `Say explicitly if a category is empty` — otherwise silence is ambiguous.
- Never accept a reported error without checking it. One agent claimed a PDF was missing from `docs/`; it was present under a misleading filename.
- Give agents the extraction command rather than letting each rediscover how to read a PDF.
- Batch: 3–5 agents per message. Five heavy agents at once hit a rate limit and one died mid-task.

**Shared preamble** — paste at the top of every prompt:

```
**Project root:** `/Users/addhyanpant/Desktop/Thesis5/FinalMERThesis` (run commands from here)

**PDF extraction:**
`{{VENV}}/bin/python {{SCRIPT}}/ex.py "docs/<file>.pdf" [max_pages]`
PDF extraction glues table columns and mangles maths — re-read the surrounding lines
before declaring a number wrong.

**CRITICAL METRIC WARNING.** `Ablation_Study/results/config_*/final_results.json`
contains TWO macro-F1 quantities that can differ IN SIGN:
- key `macro_f1` = MEAN-OF-FOLDS. The thesis explicitly rejects this (capped at 0.6267).
- POOLED macro F1 = mean of the `per_class_f1` array. **This is the primary metric.**
Always compute pooled unless the text says otherwise. Report both if they differ.

Note: the `docs/` file `CASME2_ADatabaseofSpontaneous...pdf` is Qu et al.'s **CAS(ME)²**
paper, not CASME II — do not report it missing.
```

---

## 1. Scope check

> Senior ML scientist specialising in micro-expression recognition. SCOPE CHECK before a thesis section is written. Report only — do not write or edit anything.
>
> {{SHARED PREAMBLE}}
>
> ## The proposed section
> §{{N}} "{{TITLE}}", intended to cover:
> {{BULLET LIST OF TOPICS}}
>
> ## Your job
> For EACH bullet, answer: **is this actually relevant to and used by this project**, or would it be padding?
>
> "Used by the project" means the concept has a concrete footprint — in the data, the code, or the experimental design. Check {{FILES TO READ}}.
>
> Report the EXACT implementation facts a writer would need: parameter values, tensor shapes, formulas as written in code, call order. Do not paraphrase the code — quote the values.
>
> Then answer specifically:
> {{NUMBERED QUESTIONS — the things you genuinely don't know}}
>
> ## Available evidence
> Run `ls docs/`. Name which PDFs would support each bullet. **Flag any bullet for which NO `docs/` paper provides support**, since the writer must not invent a citation.
>
> ## Output
> Brief markdown: **IN SCOPE** (bullet → code evidence file:line → supporting PDF), **OUT OF SCOPE / PADDING** (with why), **MISSING** (what the list omits), **ANSWERS** to the numbered questions. No prose padding.

**Why it earns its place.** This stage found that the apex frame is never read, that Action Units are metadata only, that focal loss's α term is inactive, and that there is no inner validation split. Four factual errors that would otherwise have been written confidently.

---

## 2. Writer

> Senior ML scientist writing one section of {{CHAPTER}} of an MSc thesis on micro-expression recognition. Write the file directly.
>
> **Write to:** `{{PATH}}`
> {{SHARED PREAMBLE}}
>
> ## What this section is
> §{{N}} "{{TITLE}}". {{ONE LINE ON THE CHAPTER'S JOB — e.g. "Background explains what things are and how they work; Chapter 3 surveys who did what."}}
>
> Target length: **{{RANGE}} words.** {{CALIBRATION NOTE — e.g. "Concise and definitional. Resist padding."}}
>
> ## Content, as fixed by a scope check of the actual project
> Cover exactly these, in this order:
> {{NUMBERED CONTENT SPEC — from the scope check}}
>
> ## VERIFIED IMPLEMENTATION FACTS — use these exactly, do not re-derive
> {{THE SCOPE CHECK'S FINDINGS, as a flat list of facts}}
> {{Flag the load-bearing ones: "State this plainly — a writer who mentions only one is factually wrong."}}
>
> ## References — strict
> - Cite ONLY papers in `docs/`. Run `ls docs/`.
> - **Do not invent a citation.** If you cannot verify a claim from a PDF you have actually opened, drop it or state it uncited as standard background.
> - Keep citations to a **minimum — aim for {{N}} or fewer.**
> - Any quotation must be verbatim from a PDF you opened.
> - Where a claim originates in a work NOT in `docs/` ({{EXAMPLES}}), attribute it in the text to the `docs/` paper that reports it — "as reported by X (year)" — rather than citing the original.
> - {{WHICH SPECIFIC PDFs COVER WHICH CLAIM}}
> - End with `## References for §{{N}}`, matching the style of `{{REFERENCE LIST PATH}}`.
>
> ## HARD CONSTRAINT — no duplication
> Read `{{OTHER CHAPTER FILES}}` before writing. That chapter owns, and you must NOT restate:
> {{EXPLICIT LIST}}
> You may cross-reference it but must not carry its content. {{ONE LINE DRAWING THE BOUNDARY — e.g. "§2.1 is about the phenomenon in the world; §3.1 is about the dataset."}}
>
> ## Style
> Markdown, start `## {{N}} <title>`. `###` subsections. British spelling. LaTeX for equations. Precise, plain, no hype. Match the register of `{{EXEMPLAR FILE}}` — read it first to calibrate tone.
>
> ## Report back
> Word count, the citations you used, and one line on what you deliberately left out to avoid duplication.

---

## 3a. Fact checker

> Senior ML scientist fact-checking one section of a thesis. Report only — do NOT edit.
>
> **File:** `{{PATH}}`
> {{SHARED PREAMBLE}}
>
> ## Check every claim
> 1. **Every double-quoted passage must be verbatim** from the PDF it is attributed to. Flag paraphrase-as-quote, spliced quotes without ellipsis, and truncations that change meaning.
> 2. **Every factual assertion about a source** must be supported by a `docs/` PDF you actually opened. Where the section attributes an out-of-corpus work through a `docs/` paper, verify the `docs/` paper really does report it.
> 3. **Every claim about THIS PROJECT** must be verified against code and data:
> {{ENUMERATED CLAIMS, each naming the file to check}}
> 4. **Overreach** — flag any sentence stated more strongly than its source supports.
>
> ## Output
> - **VERDICT**: ACCURATE or NEEDS FIXES (n issues)
> - **ERRORS** — claim as written, what the source actually says, file/table/line evidence, the minimal correction
> - **UNSUPPORTED** — claims with no verifiable source; drop or de-cite?
> - **VERIFIED** — compact list of what you positively confirmed
> Say explicitly if a category is empty. Be brief.

---

## 3b. Reference auditor (ghost references)

> Auditing the references of one thesis section for FABRICATION and ACCURACY. Report only — do NOT edit.
>
> **File:** `{{PATH}}`
> {{SHARED PREAMBLE}}
>
> ## Your job — in this order
> 1. **Run `ls docs/`.** For EVERY reference-list entry and EVERY in-text citation, determine whether a corresponding PDF exists. Name the matching filename. Any citation with no matching PDF is a **GHOST REFERENCE** — report it prominently.
> 2. **Open each matching PDF and verify its bibliographic details** against the entry: author list AND ORDER, year, title, venue, volume, pages. Check author order against the PDF's own title-page byline, **not against your prior knowledge**.
> 3. **Check every in-text citation resolves** to a listed entry, and every entry is actually cited. Report orphans in both directions.
> 4. **Check the citation format matches** `{{REFERENCE LIST PATH}}` — same author–year style, same punctuation, and the same year-suffix convention (2019a/2019b, 2014a/2014b, 2020a/2020b) where a paper already carries a suffix there.
> 5. **Check out-of-corpus attribution.** Where the text attributes a work not in `docs/`, confirm (a) it is NOT given its own reference entry, and (b) the `docs/` paper actually reports it.
> 6. Where a PDF is an arXiv preprint or author manuscript without printed volume/page data, mark the entry **UNCONFIRMED** rather than accepting or rejecting it.
>
> ## Output
> - **VERDICT**: CLEAN or PROBLEMS (n)
> - **GHOST REFERENCES** — state each explicitly. Say "none" if none.
> - **BIBLIOGRAPHIC ERRORS** — with the correct value from the PDF
> - **ORPHANS** — cited but unlisted, or listed but uncited
> - **FORMAT / SUFFIX MISMATCHES**
> - **UNCONFIRMED** — and what is needed
> - **VERIFIED** — compact list
> Be brief and specific.

**Point 2 matters.** Asking for title-page verification rather than recall caught OFF-ApexNet cited as "Gan et al." when Liong is first author.

---

## 3c. Duplication checker

> Checking a newly written thesis section for DUPLICATION against an existing chapter. Report only — do NOT edit.
>
> **New file:** `{{PATH}}`
> **Existing chapter:** `{{OTHER CHAPTER DIR}}` — read `{{ASSEMBLED FILE}}`, and `{{MOST RELEVANT SECTION}}` in particular.
>
> ## The intended division of labour
> - **{{CHAPTER A}}** explains *what things are and how they work*.
> - **{{CHAPTER B}}** surveys *who did what, with what result, and what gap remains*.
>
> The two will be read back to back by the same examiner. Any fact stated in both is a defect unless the second occurrence is doing genuinely different work.
>
> ## What to look for
> 1. **Verbatim or near-verbatim sentences** shared between them. Quote both, give locations.
> 2. **Repeated facts**, even where wording differs. {{CHAPTER B}} owns and the new section must not restate: {{EXPLICIT LIST}}.
> 3. **Repeated quotations** — the same passage quoted from the same paper in both. Flag every instance; this is the most visible form of duplication.
> 4. **Repeated arguments** — the same reasoning made twice, even with different evidence.
> 5. **Legitimate overlap** — a definition one chapter must give and the other legitimately uses is fine PROVIDED the second does not re-define it. Note these separately, and say which chapter should own each.
>
> ## Also check the reverse direction
> Is there anything in the new section that the existing chapter currently explains but should NOT, because it is now established here? Name the passage that could be cut and replaced with a cross-reference.
>
> ## Output
> - **VERDICT**: CLEAN or DUPLICATION FOUND (n instances)
> - **VERBATIM / NEAR-VERBATIM OVERLAP** — both texts quoted, with locations
> - **REPEATED FACTS** — the fact, where it appears in each, which should own it
> - **REPEATED QUOTATIONS**
> - **REPEATED ARGUMENTS**
> - **LEGITIMATE OVERLAP** — with a note on ownership
> - **REVERSE DIRECTION** — passages that could now defer
> Say explicitly if a category is empty. Be brief and specific.

**The reverse direction is the valuable half.** It produced six edits to Chapter 3 after Chapter 2 was written — §3.1.1, §3.3.2, §3.4.2, §3.6.6, §3.9.3, §3.1.6 now defer to Background instead of re-deriving.

---

## 4. De-duplication editor (used once, across a finished chapter)

> Senior academic editor working on one continuous chapter that was drafted as {{N}} standalone sections. Remove cross-section repetition. You WILL edit files directly.
>
> **Your files:** {{LIST}}
>
> ## The principle
> These files are ONE chapter read front to back, not {{N}} independent documents. A fact established in an earlier section must not be re-explained in a later one. Replace restatement with a cross-reference, or delete it where the sentence works without it.
>
> ## Canonical ownership map
> Each recurring fact belongs to exactly one section. Outside its home it may appear ONLY as a brief cross-reference — no re-derivation, no restated numbers, no re-argument.
>
> | Fact | Canonical home |
> |---|---|
> {{THE MAP — and mark which ones are IN the agent's own files: "this is YOUR section, keep it in full"}}
>
> ## What to do in each file
> {{PER-FILE INSTRUCTIONS, naming specific passages}}
>
> ## Hard constraints
> - **Do not remove any citation, quotation, table, or verified number from its canonical home.**
> - Do not merge or renumber subsections. Headings stay as they are.
> - Preserve each section's closing gap statement and its implications table, tightening only where they repeat the body.
> - Aim to cut {{X}}% of the words, entirely from repetition.
>
> ## Report back
> Per file: word count before and after, and a bullet list of what you removed or converted to a cross-reference.

**Measure first.** Before dispatching, count the repetition with a script — regex probes per recurring fact, one column per file. That produces the ownership map and the before/after evidence. This pass cut ~3,300 words from Chapter 3 and merged ten reference lists into one.

---

## Combined auditor (when running low on budget)

3a + 3b + 3c can be folded into one agent with three explicitly separated jobs and a structured report per job. Used for the Chapter 3 re-audit. Slightly weaker on duplication — that job needs to read the whole other chapter, which competes for the agent's attention.

---

## What to avoid

- **Don't let an auditor edit.** Findings you can weigh beat silent changes you can't see.
- **Don't skip the scope check** to save a round. It is the stage that catches "we don't actually do that".
- **Don't trust a reported error.** Two of roughly twenty-five findings were wrong.
- **Don't ask for "check everything".** Enumerate the specific claims; vague briefs produce vague reports.
- **Don't run five heavy agents in one message.** Three to four is the safe batch.
