import re, sys
files = sys.argv[1:]
for f in files:
    s = open(f, encoding="utf-8").read()
    s = "\n".join("" if l.lstrip().startswith("%") else l for l in s.split("\n"))
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", s, flags=re.S)
    flat = re.sub(r"\s+", " ", s)
    sents = re.split(r"(?<=[.;])\s+(?=[A-Z\\])", flat)
    n = 0
    print("=" * 30, f)
    for se in sents:
        keys = []
        for m in re.finditer(r"\\cite\{([^}]*)\}", se):
            keys += [k.strip() for k in m.group(1).split(",")]
        if not keys:
            continue
        n += 1
        t = re.sub(r"\\(autoref|ref|label|eqref)\{[^}]*\}", "#", se)
        t = re.sub(r"\\textbf\{([^}]*)\}|\\textit\{([^}]*)\}", lambda m: m.group(1) or m.group(2), t)
        print(f"[{n}] {','.join(dict.fromkeys(keys))} :: {t[:330]}")
