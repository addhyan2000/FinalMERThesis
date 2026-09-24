"""Verify every thesis.bib entry against Crossref / Semantic Scholar.

No personal data is sent: generic User-Agent, no mailto.
Output: one line per entry with verdict and any metadata mismatches.
"""
import re, json, time, difflib, urllib.request, urllib.parse, sys

UA = {"User-Agent": "thesis-bib-verifier/1.0"}
bib = open("bib/thesis.bib", encoding="utf-8").read()


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=40))
        except Exception as e:
            if "429" in str(e) or "timed out" in str(e):
                time.sleep(6 * (k + 1)); continue
            return None
    return None


def field(body, n):
    m = re.search(r"\b" + n + r"\s*=\s*\"(.*?)\"\s*,?\s*$", body, re.S | re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = s.replace("×", "x").replace("^3", "3").replace("^2", "2")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def surname(authors):
    a = authors.split(" and ")[0]
    a = a.split(",")[0] if "," in a else a.split()[-1]
    return norm(a)


rows = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
    typ, key, body = m.groups()
    e = {k: field(body + "\n", k) for k in ("title", "author", "year", "journal", "booktitle",
                                             "volume", "number", "pages", "doi", "howpublished", "publisher")}
    src, rec = None, None
    if e["doi"]:
        r = get("https://api.crossref.org/works/" + urllib.parse.quote(e["doi"]))
        if r:
            rec, src = r["message"], "crossref-doi"
    if rec is None and typ != "book":
        q = urllib.parse.quote(norm(e["title"]) + " " + surname(e["author"]))
        r = get("https://api.crossref.org/works?rows=5&query.bibliographic=" + q)
        best, bs = None, 0
        for it in (r or {}).get("message", {}).get("items", []):
            s = difflib.SequenceMatcher(None, norm(e["title"]), norm((it.get("title") or [""])[0])).ratio()
            if s > bs: best, bs = it, s
        if best and bs > 0.9:
            rec, src = best, "crossref-search"
    s2 = None
    if rec is None:
        time.sleep(1.5)
        r = get("https://api.semanticscholar.org/graph/v1/paper/search/match?fields=title,authors,year,venue,externalIds&query="
                + urllib.parse.quote(norm(e["title"])))
        if r and r.get("data"):
            s2, src = r["data"][0], "semanticscholar"
    issues = []
    if rec:
        t = (rec.get("title") or [""])[0]
        ts = difflib.SequenceMatcher(None, norm(e["title"]), norm(t)).ratio()
        if ts < 0.9: issues.append(f"title~{ts:.2f} [{t[:70]}]")
        fam = [norm(a.get("family", a.get("name", ""))) for a in rec.get("author", [])]
        if surname(e["author"]) not in " ".join(fam): issues.append(f"1st-author {surname(e['author'])} not in {fam[:3]}")
        ys = {str(p[0]) for k in ("published-print", "published-online", "issued", "published")
              for p in ((rec.get(k) or {}).get("date-parts") or [[None]]) if p and p[0]}
        if e["year"] and e["year"] not in ys: issues.append(f"year {e['year']} vs {sorted(ys)}")
        if e["volume"] and rec.get("volume") and e["volume"] != rec.get("volume"): issues.append(f"vol {e['volume']} vs {rec.get('volume')}")
        if e["pages"] and rec.get("page"):
            p1 = e["pages"].replace("--", "-").split("-")[0]
            if p1 != rec["page"].split("-")[0]: issues.append(f"pages {e['pages']} vs {rec['page']}")
        if not e["doi"] and rec.get("DOI"): issues.append(f"(DOI available: {rec['DOI']})")
        verdict = "VERIFIED" if not [i for i in issues if not i.startswith("(")] else "CHECK"
    elif s2:
        ts = difflib.SequenceMatcher(None, norm(e["title"]), norm(s2["title"])).ratio()
        au = [norm(a["name"]).split()[-1] for a in s2.get("authors", []) if a.get("name")]
        if ts < 0.9: issues.append(f"title~{ts:.2f} [{s2['title'][:70]}]")
        if surname(e["author"]) not in " ".join(au): issues.append(f"1st-author {surname(e['author'])} not in {au[:3]}")
        if e["year"] and s2.get("year") and abs(int(e["year"]) - int(s2["year"])) > 1:
            issues.append(f"year {e['year']} vs S2 {s2['year']}")
        issues.append(f"venue: {s2.get('venue') or '-'}")
        verdict = "VERIFIED" if not [i for i in issues if not i.startswith("venue")] else "CHECK"
    else:
        verdict = "BOOK" if typ == "book" else "NOT FOUND"
    rows.append((key, verdict, src, issues))
    print(f"{verdict:9} {key:17} {src or '-':16} {'; '.join(issues)}", flush=True)
    time.sleep(0.6)

print("\nSUMMARY:", {v: sum(1 for r in rows if r[1] == v) for v in set(r[1] for r in rows)})
