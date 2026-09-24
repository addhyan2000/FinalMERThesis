import re, os, sys
ch = sys.argv[1] if len(sys.argv) > 1 else "contents/chapter3_cut.tex"
src = open(ch, encoding="utf-8").read()
src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))
cache = {}
for f in os.listdir("tools/txt"):
    t = open("tools/txt/" + f, encoding="utf-8", errors="ignore").read()
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("\u2212", "-")
    cache[f[:-4]] = re.sub(r"\s+", " ", t)
# split into sentences roughly
body = re.sub(r"\s+", " ", src)
sents = re.split(r"(?<=[.;:])\s+(?=[A-Z\\])", body)
total = 0; flagged = 0; keys_seen = set(); uncached = set()
for s in sents:
    keys = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", s):
        keys += [k.strip() for k in m.group(1).split(",")]
    if not keys:
        continue
    keys_seen.update(keys)
    nums = set(re.findall(r"(?<![\w.])(\d+(?:\.\d+)+|\d{2,}(?:,\d{3})*)(?![\w])", re.sub(r"\\(ref|autoref|label|eqref)\{[^}]*\}", "", s)))
    nums -= {"2019", "2021", "2022", "2018", "2014", "2013", "2016", "2012", "2011", "2017", "101", "16", "10", "100"}
    cached = [k for k in keys if k in cache]
    for k in keys:
        if k not in cache: uncached.add(k)
    if not nums or not cached:
        continue
    total += 1
    miss = []
    for n in nums:
        variants = {n, n.replace(",", ""), n.replace(",", " "), n.replace(".", ",")}
        if not any(any(v in cache[k] for v in variants) for k in cached):
            miss.append(n)
    if miss:
        flagged += 1
        print("MISS", miss, "| keys", cached, "|", s[:230])
print("\nsentences with cited numbers checked:", total, " flagged:", flagged)
print("keys cited:", len(keys_seen), " without cached text (foundational, checked by title only):", sorted(uncached))
