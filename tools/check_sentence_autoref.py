import re, sys
FIX = len(sys.argv) > 1 and sys.argv[1] == "fix"
names = {"ch": "Chapter", "sec": "Section", "tab": "Table", "fig": "Figure", "eq": "Equation"}
# sentence start = beginning of a paragraph line (after optional indentation / \item / bold lead-in),
# or after ". " / "? " / "! "
pat = re.compile(r"(^[ \t]*(?:\\item[ \t]+)?(?:\\textbf\{[^{}]*\}[ \t]+)?|(?<=[.?!]) |(?<=[.?!]\}) )\\autoref\{((ch|sec|tab|fig|eq):[^}]*)\}", re.M)
total = 0
for f in ["introduction", "chapter2_shortened", "chapter3_cut", "chapter4_shortened", "chapter5_shortened", "chapter6"]:
    p = "contents/%s.tex" % f
    s = open(p, encoding="utf-8").read()
    out_lines = []
    n = 0
    for line in s.split("\n"):
        if line.lstrip().startswith("%"):
            out_lines.append(line); continue
        def rep(m):
            global total
            nonlocal_n[0] += 1
            pre, lab, kind = m.group(1), m.group(2), m.group(3)
            if kind == "eq":
                return "%sEquation~\\eqref{%s}" % (pre, lab)
            return "%s%s~\\ref{%s}" % (pre, names[kind], lab)
        nonlocal_n = [0]
        new = pat.sub(rep, line)
        n += nonlocal_n[0]
        out_lines.append(new)
    total += n
    print(f, n)
    if FIX and n:
        open(p, "w", encoding="utf-8").write("\n".join(out_lines))
print("total", total)
