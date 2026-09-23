import re, io, os, json, time, subprocess, urllib.parse, difflib

SP = os.path.dirname(os.path.abspath(__file__))
bib = io.open("bib/thesis.bib", encoding="utf-8").read()


def field(body, n):
    m = re.search(n + r"\s*=\s*[\"{](.*?)[\"}]\s*,?\s*$", body, re.S | re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("{", "").replace("}", "").replace("$", "").replace('\\"', "")
    s = s.replace("×", "x")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


out = {}
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
    typ, key, body = m.groups()
    body = body + "\n"
    e = {k: field(body, k) for k in ("title", "author", "year", "journal",
                                     "booktitle", "volume", "pages", "doi")}
    q = urllib.parse.quote(norm(e["title"]) + " " + e["author"].split(",")[0])
    url = ("https://api.crossref.org/works?query.bibliographic=%s&rows=3"
           "&select=DOI,title,author,container-title,volume,page,issued,type" % q)
    r = subprocess.run(["curl", "-s", "-m", "30", "-A",
                        "thesis-ref-check (mailto:addhyanp2k@gmail.com)", url],
                       capture_output=True, text=True, encoding="utf-8")
    try:
        items = json.loads(r.stdout)["message"]["items"]
    except Exception:
        items = []
    best, bs = None, 0.0
    for it in items:
        t = norm((it.get("title") or [""])[0])
        s = difflib.SequenceMatcher(None, norm(e["title"]), t).ratio()
        if s > bs:
            best, bs = it, s
    out[key] = {"bib": e, "score": bs, "cr": best}
    time.sleep(0.4)

json.dump(out, io.open(os.path.join(SP, "crossref.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)


def crauth(it):
    return [a.get("family", "") for a in it.get("author", [])]


def biblast(a):
    ps = [x.strip() for x in a.split(" and ")]
    return [norm(p.split(",")[0] if "," in p else p.split()[-1]) for p in ps]


print("%-16s %-5s %-6s %-6s %-9s %-9s %s" % ("key", "title", "year", "authrs", "volume", "pages", "notes"))
print("-" * 96)
for key, d in out.items():
    e, it, s = d["bib"], d["cr"], d["score"]
    if not it or s < 0.85:
        print("%-16s %-5.2f  *** NO CONFIDENT CROSSREF MATCH (best=%s)"
              % (key, s, (it.get("title") or ["-"])[0][:50] if it else "-"))
        continue
    notes = []
    cy = str((it.get("issued", {}).get("date-parts") or [[None]])[0][0])
    y_ok = cy == e["year"]
    ca = [norm(x) for x in crauth(it)]
    ba = biblast(e["author"])
    a_ok = ca == ba
    if not a_ok:
        if set(ca) == set(ba):
            notes.append("author ORDER differs: cr=%s" % ",".join(ca))
        else:
            notes.append("authors cr=%s" % ",".join(ca))
    cv = it.get("volume", "")
    v_ok = (not e["volume"]) or (norm(cv) == norm(e["volume"]))
    cp = it.get("page", "")
    p_ok = (not e["pages"]) or (norm(cp) == norm(e["pages"].replace("--", "-")))
    if not y_ok:
        notes.append("year cr=%s" % cy)
    if not v_ok:
        notes.append("vol cr=%s" % cv)
    if not p_ok:
        notes.append("pages cr=%s" % cp)
    if e["doi"] and e["doi"].lower() != it.get("DOI", "").lower():
        notes.append("DOI cr=%s" % it.get("DOI"))
    if not e["doi"]:
        notes.append("(bib has no DOI; cr=%s)" % it.get("DOI"))
    print("%-16s %-5.2f %-6s %-6s %-9s %-9s %s"
          % (key, s, "ok" if y_ok else "DIFF", "ok" if a_ok else "DIFF",
             "ok" if v_ok else "DIFF", "ok" if p_ok else "DIFF", "; ".join(notes)))
