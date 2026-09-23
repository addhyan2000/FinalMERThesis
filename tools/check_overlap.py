import re, io, os, glob, collections

SP = os.path.dirname(os.path.abspath(__file__))
CH = [("Ch1", "contents/introduction.tex"), ("Ch2", "contents/chapter2_shortened.tex"),
      ("Ch3", "contents/chapter3_cut.tex"), ("Ch4", "contents/chapter4_shortened.tex"),
      ("Ch5", "contents/chapter5_shortened.tex"), ("Ch6", "contents/chapter6.tex")]
N = 8


def nrm_src(s):
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb00", "ff")
    s = s.replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    s = re.sub(r"-\s*\n\s*", "", s)
    s = s.lower().replace("\u2019", "'")
    return re.findall(r"[a-z0-9]+", s)


def tex_words(raw):
    t = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in raw.split("\n"))
    t = re.sub(r"``.+?''", " QQQ ", t, flags=re.S)            # drop marked quotations
    t = re.sub(r"\\(cite|autoref|ref|label|includegraphics|caption)\*?(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)             # display math
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)
    t = t.lower()
    return re.findall(r"[a-z0-9]+", t)


src = {}
for p in glob.glob(os.path.join(SP, "txt", "*.txt")):
    src[os.path.basename(p)[:-4]] = nrm_src(io.open(p, encoding="utf-8").read())

index = collections.defaultdict(set)
for k, w in src.items():
    for i in range(len(w) - N + 1):
        index[" ".join(w[i:i + N])].add(k)

report = []
tot_grams = tot_hit = 0
for ch, f in CH:
    w = tex_words(io.open(f, encoding="utf-8").read())
    hits = [i for i in range(len(w) - N + 1) if " ".join(w[i:i + N]) in index]
    tot_grams += max(0, len(w) - N + 1)
    tot_hit += len(hits)
    # merge consecutive hits into maximal runs
    runs = []
    for i in hits:
        if runs and i <= runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for a, b in runs:
        span = w[a:b + N]
        if "qqq" in span:
            continue
        srcs = set()
        for j in range(a, b + 1):
            srcs |= index.get(" ".join(w[j:j + N]), set())
        report.append((ch, len(span), sorted(srcs), " ".join(span)))
    pct = 100.0 * len(hits) / max(1, len(w) - N + 1)
    print("%s  words=%-6d  8-grams also in a source: %-4d (%.2f%%)" % (ch, len(w), len(hits), pct))

print("\nWHOLE MANUSCRIPT: %.2f%% of 8-word sequences occur in any source paper"
      % (100.0 * tot_hit / max(1, tot_grams)))
report.sort(key=lambda r: -r[1])
print("\nMatching runs of >= %d words (unquoted prose), longest first: %d" % (N, len(report)))
for ch, n, s, txt in report:
    print("  [%s] %2d words  src=%-24s %s" % (ch, n, ",".join(s)[:24], txt[:150]))
