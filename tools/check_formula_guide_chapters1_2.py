"""Static coverage and reference checks for the first formula-guide batch."""
from pathlib import Path
import re
from check_short_chapter2 import clean, labels, keys, syntax

ROOT=Path(__file__).resolve().parents[1]
intro=(ROOT/'contents/introduction.tex').read_text(encoding='utf-8')
chapter=(ROOT/'contents/chapter2_shortened.tex').read_text(encoding='utf-8')
guide=(ROOT/'contents/formula_guide_chapters1_2.tex').read_text(encoding='utf-8')

def math_spans(text):
    return list(re.finditer(r'\\\[(.*?)\\\]|(?<!\\)\$([^$]+)\$',clean(text),re.S))

def canonical(math):
    math=re.sub(r'\\label\{[^}]+\}','',math)
    math=re.sub(r'\\(?:begin|end)\{aligned\}','',math)
    math=re.sub(r'\\(?:qquad|quad)\b','',math)
    math=math.replace(r'\\','')
    return re.sub(r'[\s&,.]','',math)

assert not math_spans(intro), 'Introduction now has formulas; extend guide'
source_displays=re.findall(r'\\\[(.*?)\\\]',clean(chapter),re.S)
assert len(source_displays)==16
marked=re.findall(r'% SOURCE-DISPLAY-(\d+)\s*\\begin\{equation\}(.*?)\\end\{equation\}',guide,re.S)
assert [int(n) for n,_ in marked]==list(range(1,17))
for i,((_,entry),source) in enumerate(zip(marked,source_displays),1):
    assert canonical(entry)==canonical(source), ('Source display changed',i)

# All inline source math belongs to these manually reviewed guide topics.
# Variables and repeated settings are covered alongside their parent formula.
groups=[([1,2,3],'capture'),([4,5],'geometry'),([6,7,8,10,11],'evm'),
        ([9],'pyramid'),([12],'clip'),(list(range(13,18)),'flow'),
        ([18,19,20],'strain'),([21,22],'magnitude'),([23,24],'tensor'),
        ([25,26],'input-normalisation'),(list(range(27,32)),'unit'),
        ([32],'relu'),([33],'convparams'),([34],'pool'),
        (list(range(35,41)),'conv3d'),([41,42],'bn'),
        ([43,44,46,47,48,49],'simamstats'),([45,50],'simamgate'),
        ([51,52],'attention'),([53,54,55],'projection'),([56,57],'ce'),
        (list(range(58,64)),'focal'),([64,65],'sampling'),
        (list(range(66,71)),'smoothing'),(list(range(71,82)),'metrics'),([82],'macro')]
assert len(math_spans(chapter))==82, 'Source math inventory changed; review coverage'
assert sorted(i for ids,_ in groups for i in ids)==list(range(1,83))
assert all('fg:'+topic in labels(guide) for _,topic in groups)
syntax(guide)
assert len(labels(guide))==len(set(labels(guide)))
assert guide.count(r'\begin{document}')==guide.count(r'\end{document}')==1
assert len(re.findall(r'\\begin\{equation\}',guide))==29
assert not re.search(r'TODO|PLACEHOLDER|verified supplied transformer source and equation locator',guide)
entries=re.findall(r'\\bibitem\{([^}]+)\}',guide)
assert len(entries)==len(set(entries))==9
assert keys(guide)==entries, 'Bibliography must match first-use citation order'
bib=(ROOT/'bib/thesis.bib').read_text(encoding='utf-8')
assert set(entries)<=set(re.findall(r'@\w+\{([^,]+),',bib))
assert 'Bai, Mengjiong and Goecke' in bib
papers={
 'yan2014':'CASME II An Improved Spontaneous Micro-Expression.pdf',
 'bai2021':'Micro-expression recognition based on video motion magnification and pre-trained neural network.pdf',
 'liong2019a':'off-apexnet-on-micro-expression-recognition-system.pdf',
 'zhaoS2021':'a-two-stage-3d-cnn-based-learning-method-for-spontaneous-micro-expression-recognition.pdf',
 'shreve2011':'Macro-and micro-expression spotting in long videos using spatio-temporal strain.pdf',
 'yang2021':'SimAM A Simple, Parameter Free Attention Module for Convolutional Neural Networks.pdf',
 'dosovitskiy2021':'an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.pdf',
 'zhang2022':'Short and Long Range Relation Based Spatio-Temporal Transformer for Micro-Expression Recognition.pdf',
 'see2019':'megc-2019-the-second-facial-micro-expressions-grand-challenge.pdf',
}
assert all((ROOT/'docs'/papers[k]).is_file() for k in entries)
print('PASS: Chapter1 has no formulas; all16 Chapter2 display blocks match, allowing layout changes.')
print('PASS: all82 source math spans mapped to reviewed explanation topics; 29 guide equation blocks.')
print('PASS: nine cited papers present locally; embedded bibliography complete and in first-use order.')
print('PASS: balanced LaTeX structure and unique labels. PDF rendering not verified here.')
