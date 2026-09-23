import re, io, os, collections

FILES = [
    ("Ch1", "contents/introduction.tex"),
    ("Ch2", "contents/chapter2_shortened.tex"),
    ("Ch3", "contents/chapter3_cut.tex"),
    ("Ch4", "contents/chapter4_shortened.tex"),
    ("Ch5", "contents/chapter5_shortened.tex"),
    ("Ch6", "contents/chapter6.tex"),
]


def nc(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in t.split("\n"))


raw, body = {}, {}
for k, f in FILES:
    raw[k] = io.open(f, encoding="utf-8").read()
    body[k] = nc(raw[k])

BAR = "=" * 72

# ---------- 1. labels: definition, duplicates, and what each labels ----------
print(BAR)
print("1. LABELS")
owner = {}          # label -> (chap, kind)
dups = []
for k, _ in FILES:
    lines = body[k].split("\n")
    envstack = []
    for i, l in enumerate(lines):
        for m in re.finditer(r"\\begin\{(figure|table|equation|align)\*?\}", l):
            envstack.append(m.group(1))
        for m in re.finditer(r"\\end\{(figure|table|equation|align)\*?\}", l):
            if envstack:
                envstack.pop()
        for m in re.finditer(r"\\label\{([^}]*)\}", l):
            lab = m.group(1)
            kind = envstack[-1] if envstack else "sec"
            if lab in owner:
                dups.append((lab, owner[lab][0], k))
            owner[lab] = (k, kind)
print("  total labels: %d   duplicates: %d" % (len(owner), len(dups)))
for d in dups:
    print("    *** DUPLICATE %s  (%s and %s)" % d)

# prefix vs actual kind
print("\n  prefix / target-kind mismatches (autoref would print the wrong word):")
bad = 0
for lab, (k, kind) in sorted(owner.items()):
    pre = lab.split(":")[0]
    ok = ((pre == "sec" and kind == "sec") or (pre == "ch" and kind == "sec")
          or (pre == "tab" and kind == "table") or (pre == "fig" and kind == "figure")
          or (pre == "eq" and kind in ("equation", "align")))
    if not ok:
        print("    *** %-40s in %s labels a %-8s (prefix '%s')" % (lab, k, kind, pre))
        bad += 1
if not bad:
    print("    none")

# ---------- 2. references ----------
print(BAR)
print("2. REFERENCES")
allrefs = collections.defaultdict(list)
for k, _ in FILES:
    for m in re.finditer(r"\\(autoref|ref|nameref|eqref)\{([^}]*)\}", body[k]):
        allrefs[m.group(2)].append((k, m.group(1)))
unres = {r: v for r, v in allrefs.items() if r not in owner}
print("  distinct targets referenced: %d   UNRESOLVED: %d" % (len(allrefs), len(unres)))
for r in sorted(unres):
    print("    *** %-42s referenced from %s" % (r, ", ".join(sorted({c for c, _ in unres[r]}))))

print("\n  \\autoref used on a float but with \\ref-style intent, or vice versa:")
issues = 0
for r, uses in sorted(allrefs.items()):
    if r not in owner:
        continue
    kind = owner[r][1]
    for k, cmd in uses:
        if cmd == "eqref" and kind not in ("equation", "align"):
            print("    *** \\eqref{%s} in %s but target is a %s" % (r, k, kind))
            issues += 1
if not issues:
    print("    none")

print("\n  cross-chapter reference map (who points where):")
mat = collections.Counter()
for r, uses in allrefs.items():
    if r in owner:
        for k, _ in uses:
            mat[(k, owner[r][0])] += 1
order = [k for k, _ in FILES]
print("      " + "".join("%6s" % c for c in order))
for a in order:
    print("  %-4s" % a + "".join("%6s" % (mat.get((a, b), "-")) for b in order))

# ---------- 3. duplicated float words ----------
print(BAR)
print("3. FLOAT REFERENCE STYLE")
dw = 0
for k, _ in FILES:
    for m in re.finditer(r"(Table|Figure|Section|Chapter)~?\\autoref\{", body[k]):
        print("    *** %s doubles the word: %s" % (k, m.group(0)))
        dw += 1
    for m in re.finditer(r"(Table|Figure)~?\d+\.\d+", body[k]):
        print("    *** %s hard-codes a float number: %s" % (k, m.group(0)))
        dw += 1
if not dw:
    print("  no doubled float words, no hard-coded float numbers")

print("\n  \\ref vs \\autoref usage (bare \\ref prints only a number):")
for k, _ in FILES:
    n_auto = len(re.findall(r"\\autoref\{", body[k]))
    n_ref = len(re.findall(r"(?<!auto)(?<!eq)\\ref\{", body[k]))
    print("    %s  autoref=%-4d bare ref=%-4d" % (k, n_auto, n_ref))

# ---------- 4. citations ----------
print(BAR)
print("4. CITATIONS")
bib = io.open("bib/thesis.bib", encoding="utf-8").read()
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
used = collections.defaultdict(set)
for k, _ in FILES:
    for m in re.finditer(r"\\cite\{([^}]*)\}", body[k]):
        for x in m.group(1).split(","):
            used[x.strip()].add(k)
invalid = sorted(set(used) - keys)
orphan = sorted(keys - set(used))
print("  bib entries: %d   distinct keys cited: %d" % (len(keys), len(used)))
print("  invalid keys: %s" % (invalid if invalid else "none"))
print("  ORPHAN entries (in .bib, cited nowhere): %s" % (orphan if orphan else "none"))
print("  keys cited in only one chapter: %d"
      % len([x for x in used if len(used[x]) == 1]))

# ---------- 5. floats ----------
print(BAR)
print("5. FLOATS")
tot_t = tot_f = 0
for k, _ in FILES:
    t = len(re.findall(r"\\begin\{table\}", body[k]))
    f = len(re.findall(r"\\begin\{figure\}", body[k]))
    tot_t += t
    tot_f += f
    print("  %s  tables=%-3d figures=%-3d" % (k, t, f))
print("  TOTAL tables=%d figures=%d" % (tot_t, tot_f))

print("\n  floats missing a \\label or a short \\caption[...]:")
prob = 0
for k, _ in FILES:
    for m in re.finditer(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", body[k], re.S):
        blk = m.group(2)
        nm = re.search(r"\\label\{([^}]*)\}", blk)
        if not nm:
            print("    *** %s: %s with NO label" % (k, m.group(1)))
            prob += 1
            continue
        if "\\caption[" not in blk:
            print("    *** %s: %s (%s) has no short caption" % (k, m.group(1), nm.group(1)))
            prob += 1
if not prob:
    print("    none -- every float has a label and a short caption")

print("\n  images referenced:")
missing = 0
for k, _ in FILES:
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body[k]):
        img = m.group(1)
        found = [p for p in ("figures/" + img, "report_figures_thesis/" + img) if os.path.exists(p)]
        print("    %s  %-34s %s" % (k, img, found[0] if found else "*** NOT FOUND"))
        if not found:
            missing += 1

# ---------- 6. hard-coded section numbers ----------
print(BAR)
print("6. HARD-CODED SECTION NUMBERS")
hc = 0
for k, _ in FILES:
    for i, l in enumerate(body[k].split("\n"), 1):
        if re.search(r"\\(caption|label|includegraphics)", l):
            continue
        for m in re.finditer(r"(?<![\w.$\\])([1-6])\.(\d+)(?:\.(\d+))?(?![\w.%])", l):
            ctx = l.strip()
            if re.search(r"\d\s*$", m.group(0)):
                pass
            print("    %s line %d: '%s'  | %s" % (k, i, m.group(0), ctx[:78]))
            hc += 1
if not hc:
    print("  none")

# ---------- 7. duplicated prose ----------
print(BAR)
print("7. DUPLICATED PROSE ACROSS CHAPTERS")


def sentences(t):
    t = re.sub(r"\\begin\{(figure|table|tabular)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\(cite|autoref|ref|label|includegraphics|caption)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)
    t = re.sub(r"[{}\[\]~&\\$]", " ", t)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", t):
        w = [x.lower().strip(".,;:()\"'") for x in s.split()]
        w = [x for x in w if any(c.isalpha() for c in x)]
        if len(w) >= 9:
            out.append(" ".join(w))
    return out


sent = {k: sentences(body[k]) for k, _ in FILES}
seen = {}
hits = []
for k, _ in FILES:
    for s in sent[k]:
        if s in seen and seen[s] != k:
            hits.append((seen[s], k, s))
        else:
            seen.setdefault(s, k)
print("  identical sentences (>=9 words) shared between chapters: %d" % len(hits))
for a, b, s in hits[:25]:
    print("    *** %s <-> %s : %s" % (a, b, s[:105]))

# near-duplicate via shingles
print("\n  near-duplicate sentence pairs (>=85%% token overlap, different chapters):")


def shing(s):
    return set(s.split())


near = 0
alls = [(k, s) for k, _ in FILES for s in sent[k]]
byfirst = collections.defaultdict(list)
for k, s in alls:
    byfirst[len(s.split())].append((k, s))
checked = set()
for n, group in byfirst.items():
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            k1, s1 = group[i]
            k2, s2 = group[j]
            if k1 == k2 or s1 == s2:
                continue
            a, b = shing(s1), shing(s2)
            ov = len(a & b) / float(max(len(a), len(b)))
            if ov >= 0.85:
                key = tuple(sorted([s1, s2]))
                if key in checked:
                    continue
                checked.add(key)
                print("    ~ %s <-> %s (%.0f%%): %s" % (k1, k2, ov * 100, s1[:88]))
                near += 1
if not near:
    print("    none")

# ---------- 8. size ----------
print(BAR)
print("8. SIZE")


def words(t):
    t = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"[{}\[\]~&\\]", " ", t)
    return len([w for w in t.split() if any(c.isalpha() for c in w)])


tw = 0
for k, f in FILES:
    w = words(body[k])
    tw += w
    print("  %-4s %-34s %5d lines  ~%6d words" % (k, os.path.basename(f), raw[k].count("\n") + 1, w))
print("  TOTAL ~%d words" % tw)

# ---------- 9. chapter/section counts ----------
print(BAR)
print("9. STRUCTURE")
for k, _ in FILES:
    print("  %-4s chapters=%d sections=%-3d subsections=%-3d display-math=%d"
          % (k, len(re.findall(r"\\chapter\{", body[k])),
             len(re.findall(r"\\section\{", body[k])),
             len(re.findall(r"\\subsection\{", body[k])),
             len([l for l in body[k].split("\n") if l.strip() == "\\["])))
    st = []
    bad2 = 0
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body[k]):
        if m.group(1) == "begin":
            st.append(m.group(2))
        elif not st or st.pop() != m.group(2):
            bad2 += 1
    if st or bad2:
        print("       *** unclosed=%s mismatch=%d" % (st, bad2))
