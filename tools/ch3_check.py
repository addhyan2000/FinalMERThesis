#!/usr/bin/env python3
"""Structural gate for Chapter 3: cross-references, tables, figures, word count.

NOTE: citation checking moved to tools/bibliography.py when the thesis adopted a
single keyed bibliography. This tool no longer inspects citations at all.

Read-only. Never edits. Run from the repository root:
    python3 tools/ch3_check.py
    python3 tools/ch3_check.py --json
"""
import argparse
import json
import re
import sys
from pathlib import Path

CH3 = Path("Thesis_Chapter3_LiteratureReview/Tech-wise")
CH2 = Path("Thesis_Chapter2_Background")
CH4 = Path("Thesis_Chapter4_Methodology")
CH5 = Path("Thesis_Chapter5_Results")
SECTION_GLOB = "[0-9][0-9]_3.*.md"


def _numbered(d):
    """Section files of a chapter directory, excluding the generated 00_ bundle."""
    return [p for p in sorted(d.glob("[0-9][0-9]_*.md"))
            if not p.name.startswith("00_")] if d.exists() else []


# A reference-list entry after renumbering: "[12] Yan, W.-J., ..."
ENTRY_RE = re.compile(r"^\[(\d+)\]\s+[A-ZÄÖÜ]")
# A reference-list entry before renumbering: "Yan, W.-J., ... (2014). Title."
LEGACY_ENTRY_RE = re.compile(r"^[A-ZÄÖÜ][^\n]*\((?:19|20)\d{2}[a-c]?\)\.")
# In-text numeric citation: [12] or [7, 8]
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Any parenthesised year, used to surface residual author-year citations.
YEAR_RE = re.compile(r"\((?:[^()]{0,80}?)(?:19|20)\d{2}[a-c]?\)")
# Cross-references.
SECREF_RE = re.compile(r"§(\d+(?:\.\d+)*)")
TABLEREF_RE = re.compile(r"Table (\d+\.\d+)")
FIGREF_RE = re.compile(r"Figure (\d+\.\d+)")
# Definitions.
TABLEDEF_RE = re.compile(r"^\*Table (\d+\.\d+) —", re.M)
FIGDEF_RE = re.compile(r"^\*Figure (\d+\.\d+) —", re.M)
HEADING_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\s", re.M)

# Masking patterns for non-citation bracketed numbers.
FENCE_RE = re.compile(r'^```.*?^```', re.S | re.M)
INLINE_CODE_RE = re.compile(r'`[^`\n]+`')
MATH_RE = re.compile(r'\$[^$\n]+\$')
DQUOTE_RE = re.compile(r'[\"“][^\n]*?[\"”]')


def mask(text, quotes=True):
    """Blank out spans where a bracketed number is not a citation.

    Replaces each masked span with spaces of equal length, so line numbers
    and character offsets are preserved for reporting.
    """
    regexes = [FENCE_RE, INLINE_CODE_RE, MATH_RE]
    if quotes:
        regexes.append(DQUOTE_RE)
    for rx in regexes:
        text = rx.sub(lambda m: re.sub(r"\S", " ", m.group(0)), text)
    return text


def section_files():
    return sorted(CH3.glob(SECTION_GLOB))


def load_allowlist():
    """Surface forms that must NOT be converted: out-of-corpus attributions.

    Read from citation_map.tsv when it exists (bucket == 'out-of-corpus' or
    'not-a-citation'); empty before Task 2 has run.
    """
    path = CH3 / "citation_map.tsv"
    if not path.exists():
        return set()
    allow = set()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] in ("out-of-corpus", "not-a-citation"):
            allow.add(parts[0])
    return allow


def collect_headings():
    """Every section number that exists as a heading, in Chapters 2 to 5.

    Chapters 4 and 5 are included because Chapter 3 refers forward into them;
    without those headings a valid forward reference is reported as broken.
    """
    found = set()
    for path in (list(section_files()) + sorted(CH2.glob("*.md"))
                 + _numbered(CH4) + _numbered(CH5)):
        for num in HEADING_RE.findall(path.read_text(encoding="utf-8")):
            found.add(num)
            # A reference to §3.1 is satisfied by the heading "## 3.1".
            parts = num.split(".")
            for i in range(1, len(parts) + 1):
                found.add(".".join(parts[:i]))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not CH3.exists():
        print(f"error: {CH3} not found — run from the repository root", file=sys.stderr)
        return 2

    bodies = {p: p.read_text(encoding="utf-8") for p in section_files()}
    all_body = "\n".join(bodies.values())
    refs_text = ""

    # --- citations -------------------------------------------------------
    masked_all = mask(all_body, quotes=True)
    cited = set()
    for group in CITE_RE.findall(masked_all):
        for n in group.split(","):
            cited.add(int(n.strip()))

    entries = [int(m.group(1)) for line in refs_text.splitlines()
               if (m := ENTRY_RE.match(line))]
    legacy = [l for l in refs_text.splitlines() if LEGACY_ENTRY_RE.match(l)]
    reference_entries = len(entries) if entries else len(legacy)

    allow = load_allowlist()
    residual = []
    for path, text in bodies.items():
        masked = mask(text, quotes=False)
        for i, line in enumerate(masked.splitlines(), 1):
            for m in YEAR_RE.finditer(line):
                token = m.group(0)
                # An allowed surface form may be narrative ("Yan et al. (2013)")
                # while YEAR_RE matches only its parenthetical tail ("(2013)").
                # Accept the match if any allowed form covers this position.
                covered = False
                for form in allow:
                    start = 0
                    while (idx := line.find(form, start)) != -1:
                        if idx <= m.start() and m.end() <= idx + len(form):
                            covered = True
                            break
                        start = idx + 1
                    if covered:
                        break
                if covered:
                    continue
                residual.append({"text": token, "file": path.name, "line": i})

    # --- cross-references ------------------------------------------------
    headings = collect_headings()
    broken_secs = sorted({r for r in SECREF_RE.findall(all_body)
                          if r not in headings})

    tables_defined = set(TABLEDEF_RE.findall(all_body))
    broken_tables = sorted({r for r in TABLEREF_RE.findall(all_body)
                            if r not in tables_defined})
    figs_defined = set(FIGDEF_RE.findall(all_body))
    broken_figs = sorted({r for r in FIGREF_RE.findall(all_body)
                          if r not in figs_defined})

    uncited = sorted(set(entries) - cited) if entries else []
    word_count = len(all_body.split())

    result = {
        "numeric_citations": len(cited),
        "reference_entries": reference_entries,
        "residual_author_year": residual,
        "broken_section_refs": broken_secs,
        "broken_table_refs": broken_tables,
        "broken_figure_refs": broken_figs,
        "uncited_entries": uncited,
        "word_count": word_count,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"distinct numeric citations in body : {result['numeric_citations']}")
        print(f"reference-list entries             : {result['reference_entries']}")
        print(f"residual author-year strings       : {len(residual)}")
        for r in residual[:15]:
            print(f"    {r['file']}:{r['line']}  {r['text']}")
        if len(residual) > 15:
            print(f"    … and {len(residual) - 15} more")
        print(f"broken section cross-references    : {broken_secs or 'none'}")
        print(f"broken table references            : {broken_tables or 'none'}")
        print(f"broken figure references           : {broken_figs or 'none'}")
        print(f"uncited reference entries          : {uncited or 'none'}")
        print(f"word count (section files only)    : {word_count}")

    ok = (not residual and not broken_secs and not broken_tables
          and not broken_figs and not uncited
          and result["numeric_citations"] == result["reference_entries"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
