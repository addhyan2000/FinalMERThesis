import re, io, os, glob

SP = os.path.dirname(os.path.abspath(__file__))
CH = [("Ch1", "contents/introduction.tex"), ("Ch2", "contents/chapter2_shortened.tex"),
      ("Ch3", "contents/chapter3_cut.tex"), ("Ch4", "contents/chapter4_shortened.tex"),
      ("Ch5", "contents/chapter5_shortened.tex"), ("Ch6", "contents/chapter6.tex")]


def nrm(s):
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb00", "ff")
    s = s.replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    s = re.sub(r"-\s*\n\s*", "", s)                 # de-hyphenate line breaks
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.lower()
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


src = {os.path.basename(p)[:-4]: nrm(io.open(p, encoding="utf-8").read())
       for p in glob.glob(os.path.join(SP, "txt", "*.txt"))}


def clean_tex(q):
    q = re.sub(r"\\ldots|\\dots|\.\.\.", " @@ ", q)
    q = re.sub(r"\$[^$]*\$", " @@ ", q)             # inline math breaks a quote into segments
    q = re.sub(r"\\(textit|emph|textbf)\{([^}]*)\}", r"\2", q)
    q = re.sub(r"\\[a-zA-Z]+\*?", " ", q)
    q = q.replace("~", " ").replace("{", "").replace("}", "")
    return q


results = []
for ch, f in CH:
    raw = io.open(f, encoding="utf-8").read()
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in raw.split("\n"))
    paras = re.split(r"\n\s*\n", body)
    for para in paras:
        cites = [k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", para) for k in m.split(",")]
        for m in re.finditer(r"``(.+?)''", para, re.S):
            q = m.group(1)
            segs = [s for s in (nrm(x) for x in clean_tex(q).split("@@")) if len(s.split()) >= 3]
            if not segs or sum(len(s.split()) for s in segs) < 4:
                continue
            # attribution = cite keys nearest the quote in the paragraph
            before = para[:m.start()]
            after = para[m.end():]
            near = re.findall(r"\\cite\{([^}]*)\}", before)[-1:] + re.findall(r"\\cite\{([^}]*)\}", after)[:1]
            near = [k.strip() for x in near for k in x.split(",")] or cites
            ok_in = [k for k in near if k in src and all(s in src[k] for s in segs)]
            anywhere = [k for k in src if all(s in src[k] for s in segs)]
            results.append((ch, re.sub(r"\s+", " ", q)[:95], near, ok_in, anywhere, segs))

good = [r for r in results if r[3]]
wrongsrc = [r for r in results if not r[3] and r[4]]
missing = [r for r in results if not r[3] and not r[4]]
print("quotations checked (>=4 words): %d" % len(results))
print("  verbatim in the CITED source : %d" % len(good))
print("  found, but in a DIFFERENT source than the one cited : %d" % len(wrongsrc))
print("  NOT FOUND verbatim in any of the 30 sources : %d" % len(missing))
for label, grp in (("FOUND IN A DIFFERENT SOURCE", wrongsrc), ("NOT FOUND ANYWHERE", missing)):
    if grp:
        print("\n--- %s ---" % label)
        for ch, q, near, ok, anyw, segs in grp:
            print("  [%s] cited=%s  found_in=%s" % (ch, ",".join(near) or "NONE", ",".join(anyw) or "-"))
            print("       \"%s\"" % q)
print("\n--- all verified quotes ---")
for ch, q, near, ok, anyw, segs in good:
    print("  ok [%s] %-12s \"%s\"" % (ch, ok[0], q[:80]))
