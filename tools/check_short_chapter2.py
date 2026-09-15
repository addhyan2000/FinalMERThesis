"""Static integration and approximate word counts; does not compile LaTeX."""
from pathlib import Path
from collections import Counter
import json
import re
from assemble_short_chapter2 import HEADER

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = ROOT / "contents"

def read(name):
    return (CONTENTS / name).read_text(encoding="utf-8")

def clean(text):
    return re.sub(r"(?m)(?<!\\)%.*", "", text)

def labels(text):
    return re.findall(r"\\label\{([^}]+)\}", clean(text))

def keys(text):
    return list(dict.fromkeys(k.strip() for group in re.findall(
        r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", clean(text)) for k in group.split(",")))

def refs(text):
    return set(re.findall(r"\\(?:auto|eq|page)?ref\{([^}]+)\}", clean(text)))

def words(text):
    # Approximate prose count, including headings and visible float text.
    # Excludes comments, displayed/inline mathematics, citation/label keys,
    # LaTeX command names, table column declarations and short-caption copies.
    text = clean(text)
    text = re.sub(r"\\\[.*?\\\]|\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{tabular\}[^\n]*", " ", text)
    text = re.sub(r"\\caption\[[^\]]*\]", r"\\caption", text)
    text = re.sub(r"\\(?:label|cite|autoref|ref|begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", text)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))

def sections(text):
    starts = list(re.finditer(r"(?m)^\\section\{([^}]+)\}", clean(text)))
    text = clean(text)
    return [(m.group(1), text[m.start():starts[i+1].start() if i+1 < len(starts) else len(text)])
            for i, m in enumerate(starts)]

def syntax(text):
    text = clean(text)
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in text)
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if kind == "begin":
            stack.append(name)
        else:
            assert stack and stack.pop() == name, (kind, name)
    assert not stack
    depth = 0
    for char in re.sub(r"\\[{}%]", "", text):
        depth += (char == "{") - (char == "}")
        assert depth >= 0
    assert depth == 0
    assert text.count(r"\[") == text.count(r"\]")
    assert len(re.findall(r"(?<!\\)\$", text)) % 2 == 0

def main():
    original = read("chapter2.tex")
    candidate = read("chapter2_shortened.tex")
    halves = [read(f"chapter2_short_half{i}.tex") for i in (1, 2)]
    assert candidate == HEADER + "\n\n".join(s.rstrip() for s in halves) + "\n"
    assert len(sections(candidate)) == 9
    assert candidate.count(r"\subsection{") == original.count(r"\subsection{") == 40
    assert [x for x in labels(candidate) if x in labels(original)] == labels(original)
    assert len(labels(original)) == 50
    assert len(labels(candidate)) == len(set(labels(candidate))) == 52
    assert keys(candidate) == keys(original), "Citation keys or first-use order changed"
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "bib/thesis.bib").read_text(encoding="utf-8")))
    assert set(keys(candidate)) <= bibkeys
    assert not re.search(r"Chapter[~\s]+\d|Section[~\s]+\d\.\d", clean(candidate), re.I)
    for name in ("chapter2_shortened.tex", "chapter2_short_half1.tex", "chapter2_short_half2.tex",
                 "chapter2_short_acronyms.tex", "list_of_acronyms.tex", "list_of_tables.tex", "list_of_figures.tex"):
        syntax(read(name))
    floats = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", clean(candidate), re.S)
    assert len(floats) == 2
    for float_text in floats:
        assert re.search(r"\\caption\[[^\]]+\]\{", float_text)
        assert len(labels(float_text)) == 1 and labels(float_text)[0] in refs(candidate)
    assert r"\listoftables" in clean(read("list_of_tables.tex"))
    assert r"\listoffigures" in clean(read("list_of_figures.tex"))
    acronyms = re.findall(r"\\item\[([^\]]+)\]", read("list_of_acronyms.tex"))
    assert len(acronyms) == len(set(acronyms))
    assert set(re.findall(r"\\item\[([^\]]+)\]", read("chapter2_short_acronyms.tex"))) <= set(acronyms)
    other = "\n".join(read(n) for n in ("introduction.tex", "chapter4.tex", "chapter5.tex", "chapter6.tex"))
    for chapter3 in ("chapter3.tex", "chapter3_shortened.tex"):
        baseline = original + read(chapter3) + other
        integrated = candidate + read(chapter3) + other
        assert refs(integrated) - set(labels(integrated)) <= refs(baseline) - set(labels(baseline))
        assert not [k for k,v in Counter(labels(integrated)).items() if v > 1]
    counts = {"original": words(original), "shortened": words(candidate),
              "halves": [words(h) for h in halves], "sections": []}
    for i, ((_, old), (title, new)) in enumerate(zip(sections(original), sections(candidate)), 1):
        counts["sections"].append({"section": f"2.{i}", "title": title,
                                   "original": words(old), "shortened": words(new)})
    print(json.dumps(counts, indent=2))
    print("PASS: assembly, structure, citations/order, references, floats, acronyms and static syntax.")
    print("Rendering, generated lists and pagination require the external thesis master and TeX engine.")

if __name__ == "__main__":
    main()
