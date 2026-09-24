import re, sys
ch = sys.argv[1] if len(sys.argv) > 1 else "contents/chapter5_shortened.tex"
s = open(ch, encoding="utf-8").read()
s = "\n".join(l if not l.lstrip().startswith("%") else "" for l in s.split("\n"))
bib = open("bib/thesis.bib", encoding="utf-8").read()
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
uses = {}
for m in re.finditer(r"\\cite\{([^}]*)\}", s):
    line = s[:m.start()].count("\n") + 1
    for k in m.group(1).split(","):
        uses.setdefault(k.strip(), []).append(line)
print("citation commands:", len(re.findall(r"\\cite\{", s)), " distinct keys:", len(uses))
for k in sorted(uses):
    t = re.search(r"@\w+\{" + re.escape(k) + r",.*?title\s*=\s*\"(.*?)\"", bib, re.S)
    print(("OK      " if k in bibkeys else "MISSING ") + k.ljust(16), "lines", uses[k], "|", (t.group(1)[:75] if t else ""))
