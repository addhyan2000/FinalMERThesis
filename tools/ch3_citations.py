#!/usr/bin/env python3
"""Chapter 3 citation inventory and substitution.

    python3 tools/ch3_citations.py extract         # build citation_inventory.tsv
    python3 tools/ch3_citations.py apply --dry-run # preview substitutions
    python3 tools/ch3_citations.py apply           # rewrite the section files

Run from the repository root.
"""
import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

CH3 = Path("Thesis_Chapter3_LiteratureReview/Tech-wise")
SECTION_GLOB = "[0-9][0-9]_3.*.md"
INVENTORY = CH3 / "citation_inventory.tsv"
MAP = CH3 / "citation_map.tsv"

# Matches a parenthesised citation-shaped token, greedy enough to capture
# multi-author and multi-year forms:
#   (Yan et al., 2014)   (Li, Huang & Zhao, 2018, 2021)   (Zhao & Pietikainen, 2007)
PAREN_RE = re.compile(r"\([^()]{0,100}?(?:19|20)\d{2}[a-c]?(?:\s*,\s*(?:19|20)\d{2}[a-c]?)*\s*(?:,[^()]{0,40})?\)")
# Matches a narrative citation:  Yan et al. (2014)   Li, Huang and Zhao (2018, 2021)
NARRATIVE_RE = re.compile(
    r"[A-ZÄÖÜ][A-Za-zäöüßéíñ'’\-]+"
    r"(?:(?:\s+et\s+al\.)|(?:,?\s*(?:&|and)\s+[A-ZÄÖÜ][A-Za-zäöüßéíñ'’\-]+)"
    r"(?:\s+(?:&|and)\s+[A-ZÄÖÜ][A-Za-zäöüßéíñ'’\-]+)?)?"
    r"\s+\((?:19|20)\d{2}[a-c]?(?:\s*,\s*(?:19|20)\d{2}[a-c]?)*"
    r"(?:,\s*[^()]{1,30})?\)")


def section_files():
    return sorted(CH3.glob(SECTION_GLOB))


def extract():
    found = OrderedDict()  # surface form -> [count, first_file, first_line]
    for path in section_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for regex in (NARRATIVE_RE, PAREN_RE):
                for m in regex.finditer(line):
                    tok = m.group(0).strip()
                    if tok in found:
                        found[tok][0] += 1
                    else:
                        found[tok] = [1, path.name, lineno]

    with INVENTORY.open("w", encoding="utf-8") as fh:
        fh.write("surface_form\tcount\tfirst_file\tfirst_line\n")
        for tok, (count, fname, lineno) in found.items():
            fh.write(f"{tok}\t{count}\t{fname}\t{lineno}\n")
    print(f"wrote {INVENTORY} — {len(found)} distinct surface forms")
    return found


def load_map():
    if not MAP.exists():
        sys.exit(f"error: {MAP} does not exist — classify the inventory first")
    rows = []
    lines = MAP.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            sys.exit(f"error: malformed map row: {line!r}")
        surface, bucket, number = parts[0], parts[1], parts[2]
        if bucket not in ("in-corpus", "out-of-corpus", "not-a-citation"):
            sys.exit(f"error: unknown bucket {bucket!r} for {surface!r}")
        if bucket == "in-corpus" and not number.strip():
            sys.exit(f"error: in-corpus row without a number: {surface!r}")
        rows.append((surface, bucket, number.strip()))
    return rows


def replacement_for(surface, number):
    """Build the numeric form, preserving any author name in narrative usage.

    (Yan et al., 2014)            -> [1]
    Yan et al. (2014)             -> Yan et al. [1]
    Li et al. (2018, Table 6)     -> Li et al. [6] (Table 6)
    Li, Huang and Zhao (2018, 2021) -> Li, Huang and Zhao [7, 8]
    """
    nums = "[" + ", ".join(n.strip() for n in number.split(",")) + "]"
    if surface.startswith("("):
        return nums
    # Narrative: everything before the opening parenthesis is the author phrase.
    head, _, tail = surface.partition("(")
    author = head.rstrip()
    locator = ""
    inner = tail.rstrip(")")
    # A locator is any non-year fragment after the final comma, e.g. "Table 6".
    fragments = [f.strip() for f in inner.split(",")]
    non_year = [f for f in fragments if not re.fullmatch(r"(?:19|20)\d{2}[a-c]?", f)]
    if non_year:
        locator = " (" + ", ".join(non_year) + ")"
    return f"{author} {nums}{locator}"


def apply(dry_run):
    rows = load_map()
    # Longest surface form first, so "Liong et al. (2019a)" is consumed before
    # any shorter pattern can fire on the same text.
    rows.sort(key=lambda r: len(r[0]), reverse=True)

    total = 0
    for path in section_files():
        text = original = path.read_text(encoding="utf-8")
        for surface, bucket, number in rows:
            if bucket != "in-corpus":
                continue
            if surface not in text:
                continue
            count = text.count(surface)
            text = text.replace(surface, replacement_for(surface, number))
            total += count
        if text != original:
            if dry_run:
                print(f"would rewrite {path.name}")
            else:
                path.write_text(text, encoding="utf-8")
                print(f"rewrote {path.name}")
    print(f"{'would substitute' if dry_run else 'substituted'} {total} occurrences")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CH3.exists():
        sys.exit("error: run from the repository root")
    if args.cmd == "extract":
        extract()
    else:
        apply(args.dry_run)


if __name__ == "__main__":
    main()
