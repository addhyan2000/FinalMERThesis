"""Search each cached source for the evidence behind a specific thesis claim."""
import re, sys
checks = [l.rstrip("\n").split(" | ") for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.startswith("#")]
cache = {}
def text(k):
    if k not in cache:
        t = open(f"tools/txt/{k}.txt", encoding="utf-8", errors="ignore").read()
        t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("−", "-")
        t = re.sub(r"-\s*\n\s*", "", t)          # rejoin hyphenated line breaks
        cache[k] = re.sub(r"\s+", " ", t)
    return cache[k]
for row in checks:
    tag, key, pat = row[0], row[1], row[2]
    t = text(key)
    m = re.search(pat, t, re.I)
    if m:
        a, b = max(0, m.start() - 90), min(len(t), m.end() + 90)
        print(f"OK   {tag:28} {key:12} ...{t[a:b]}...")
    else:
        print(f"MISS {tag:28} {key:12} pattern: {pat}")
