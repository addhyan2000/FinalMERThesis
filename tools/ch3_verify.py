#!/usr/bin/env python3
"""Mechanical drift + citation-pairing verifier for the Chapter 3 reformat.

Replaces the scriptable parts of audit briefs 3a and 3b. Compares each section
file against a git ref (default HEAD) and asserts that nothing but formatting
changed, then checks every name-plus-number citation pairing.

    python3 tools/ch3_verify.py                    # all sections vs HEAD
    python3 tools/ch3_verify.py 09_3.9             # one section (substring match)
    python3 tools/ch3_verify.py --ref a0d4358      # against a different ref
"""
import argparse, re, subprocess, sys
from collections import Counter
from pathlib import Path

CH3 = Path("Thesis_Chapter3_LiteratureReview/Tech-wise")
REFS = CH3 / "11_References.md"
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?%?")
QUOTE_RE = re.compile(r'"([^"\n]{12,})"')
HEAD_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.M)
ROW_RE = re.compile(r"^\|.*\|\s*$", re.M)
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Only NARRATIVE AUTHOR citations, never method names or sentence-final markers:
#   "Yan et al. [1]"  "Yang et al.'s [14]"  "Li, Huang and Zhao [16, 17]"
S = r"[A-ZÄÖÜ][A-Za-zäöüßéíñ'\u2019\-]+"
NAMED_RE = re.compile(
    r"(?:^|[\s(])("
    + S + r")(?:\s+et\s+al\.(?:'s|\u2019s)?"
    + r"|,\s+" + S + r"\s+and\s+" + S
    + r"|\s+(?:and|&)\s+" + S
    + r")\s\[(\d+(?:\s*,\s*\d+)*)\]")
ENTRY_RE = re.compile(r"^\[(\d+)\]\s+([A-ZÄÖÜ][A-Za-zäöüßéíñ'’\-]+)")


def git_show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                          text=True, check=True).stdout


def strip_citations(text, numeric=None):
    """Remove BOTH citation shapes from BOTH sides, so the number comparison is
    like-for-like. Applying only one shape per side made bracketed non-citations
    such as the math interval $[8, 16]$ look like lost numbers."""
    text = re.sub(r"\s*\((?:[^()]{0,90}?)(?:19|20)\d{2}[a-c]?\)", "", text)
    text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)
    return text


def check_section(path, ref):
    after = path.read_text(encoding="utf-8")
    before = git_show(ref, str(path))
    a, b = strip_citations(after), strip_citations(before)
    issues = []

    lost = Counter(NUM_RE.findall(b)) - Counter(NUM_RE.findall(a))
    if lost:
        issues.append(("NUMBER LOST", dict(lost)))

    for q in QUOTE_RE.findall(before):
        if q not in after:
            issues.append(("QUOTATION ALTERED", q[:90]))

    hb = [(h, t) for h, t in HEAD_RE.findall(before)]
    ha = [(h, t) for h, t in HEAD_RE.findall(after)]
    hb_s = {t for _, t in hb}
    for lvl, t in hb:
        if t not in {x for _, x in ha}:
            issues.append(("HEADING CHANGED OR REMOVED", t[:80]))

    rb, ra = len(ROW_RE.findall(before)), len(ROW_RE.findall(after))
    if ra < rb:
        issues.append(("TABLE ROWS LOST", f"{rb} -> {ra}"))

    return issues, len(before.split()), len(after.split())


def check_pairings():
    entries = {}
    for line in REFS.read_text(encoding="utf-8").splitlines():
        m = ENTRY_RE.match(line)
        if m:
            entries[int(m.group(1))] = m.group(2)
    bad = []
    for path in sorted(CH3.glob("[0-9][0-9]_3.*.md")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for phrase, nums in NAMED_RE.findall(line):
                surname = phrase.strip()
                first = int(nums.split(",")[0])
                if first in entries and not entries[first].startswith(surname[:4]):
                    bad.append((path.name, lineno, phrase.strip(), first, entries[first]))
    return bad, len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="")
    ap.add_argument("--ref", default="HEAD")
    args = ap.parse_args()

    files = [p for p in sorted(CH3.glob("[0-9][0-9]_3.*.md")) if args.filter in p.name]
    if not files:
        sys.exit(f"no section file matches {args.filter!r}")

    total = 0
    print(f"=== drift vs {args.ref} ===")
    for p in files:
        issues, wb, wa = check_section(p, args.ref)
        flag = "DRIFT" if issues else "ok"
        print(f"{p.name:42s} {wb:6d} -> {wa:6d} words   {flag}")
        for kind, detail in issues:
            print(f"    {kind}: {detail}")
        total += len(issues)

    print("\n=== name/number pairing ===")
    bad, n = check_pairings()
    print(f"reference entries parsed: {n}")
    if bad:
        for f, l, phrase, num, surname in bad:
            print(f"    MISPAIRED {f}:{l}  '{phrase}' -> [{num}] but entry {num} is {surname}")
    else:
        print("    all pairings consistent")
    total += len(bad)

    print(f"\nTOTAL ISSUES: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
