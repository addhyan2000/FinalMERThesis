"""Check Chapter 5 assembly, preserved evidence and source-result arithmetic.

Static checks only: the external thesis master/TeX engine must render the files.
"""
from collections import Counter
from itertools import product
from pathlib import Path
import json
import math
import re

from assemble_short_chapter5 import ROOT, CONTENTS, HEADER
from check_short_chapter2 import clean, keys, labels, refs, syntax, words, sections

def read(name):
    return (CONTENTS / name).read_text(encoding='utf-8-sig')

def floats(text):
    return {labels(m.group())[0]: m.group() for m in re.finditer(
        r'\\begin\{(table|figure)\}.*?\\end\{\1\}', clean(text), re.S)}

def table_rows(text):
    body = re.search(r'\\begin\{tabular\}[^\n]*\n(.*?)\\end\{tabular\}', text, re.S).group(1)
    return [line.strip()[:-2].strip() for line in body.splitlines()
            if '&' in line and line.strip().endswith(r'\\')][1:]

def normalise_row(row):
    row = re.sub(r'config\\_(\d+)(?:\\_[A-Za-z0-9]+)+', r'config\\_\1', row)
    return re.sub(r'\s+', '', row)

def config_ids(row):
    return [int(n) for n in re.findall(r'config\\_(\d+)', row)]

def cell_number(cell):
    cell = cell.replace(r'$-$', '-')
    return float(re.search(r'[+-]?\d+(?:\.\d+)?', cell).group())

def close_display(actual, expected):
    assert abs(actual-expected) < 0.000051, (actual, expected)

def result_data():
    data = {}
    for path in (ROOT / 'Ablation_Study/results').glob('config_*/final_results.json'):
        n = int(path.parent.name.split('_')[1])
        obj = json.loads(path.read_text(encoding='utf-8'))
        m = obj['metrics']
        cm = m['confusion_matrix']
        support = [sum(row) for row in cm]
        assert support == [99,32,25], (n, support)
        f1 = [2*cm[i][i]/(support[i]+sum(row[i] for row in cm)) for i in range(3)]
        assert all(math.isclose(a,b) for a,b in zip(f1,m['per_class_f1']))
        correct = sum(cm[i][i] for i in range(3))
        accuracy = correct/156
        assert math.isclose(accuracy, m['micro_f1'])
        assert obj['extra']['loso_folds_run'] == obj['extra']['loso_folds_total'] == 25
        summary = (path.parent/'configuration_summary.txt').read_text(encoding='utf-8')
        duration = float(re.search(r'Total Train Time[^\d]*([\d.]+)',summary).group(1))
        data[n] = {'f1': f1, 'macro':sum(f1)/3, 'accuracy':accuracy, 'correct':correct,
                   'flags':obj['toggles'], 'hours':duration*25/3600}
    assert len(data) == 12
    return data

def verify_tables(fs, data):
    ranking = sorted(data, key=lambda n:data[n]['macro'], reverse=True)
    rank_rows = table_rows(fs['tab:twelve-ranked'])
    assert [config_ids(r)[0] for r in rank_rows] == ranking
    for i,row in enumerate(rank_rows,1):
        n=config_ids(row)[0]
        cells=row.split('&')
        assert int(cells[0]) == i
        for cell,flag in zip(cells[2:6],['use_evm','use_simam','use_cnn','use_transformer']):
            assert (r'\checkmark' in cell) == data[n]['flags'][flag]
        for cell,value in zip(cells[-3:],[data[n]['macro'],data[n]['accuracy'],data[n]['correct']]):
            close_display(cell_number(cell),value)
    for label in ['tab:per-class-f1','tab:simam-per-class']:
        for row in table_rows(fs[label]):
            d=data[config_ids(row)[0]]
            expected=d['f1']+([d['macro']] if label=='tab:per-class-f1' else [])
            for cell,value in zip(row.split('&')[1:],expected):
                close_display(cell_number(cell),value)
    effects=[]
    for label,flag,count in [('tab:transformer-pairs','use_transformer',6),
                             ('tab:evm-pairs-result','use_evm',6),
                             ('tab:simam-pairs-result','use_simam',4),
                             ('tab:cnn-pairs-result','use_cnn',4)]:
        deltas=[]
        for row in table_rows(fs[label]):
            off,on=config_ids(row)
            assert not data[off]['flags'][flag] and data[on]['flags'][flag]
            assert all(data[off]['flags'][f]==data[on]['flags'][f]
                       for f in data[off]['flags'] if f != flag)
            delta=data[on]['macro']-data[off]['macro']
            close_display(cell_number(row.split('&')[-1]),delta)
            deltas.append(delta)
        assert len(deltas)==count
        effects.append((sum(deltas)/count,count,sum(x>0 for x in deltas)))
    for row,(mean,count,positive) in zip(table_rows(fs['tab:component-effects']),effects):
        cells=row.split('&')
        close_display(cell_number(cells[1]),mean)
        assert int(cells[2])==count
        assert [int(n) for n in re.findall(r'\d+',cells[3])]==[positive,count]
    published=[cell_number(r.split('&')[-1]) for r in table_rows(fs['tab:published-uf1'])]
    # Supplied MEGC2019 PDF page4, Table IV, CASME II UF1 column; visually verified.
    assert published==[.8764,.8621,.8382,.8293,.7805,.7068,.7026]
    total=sum(d['hours'] for d in data.values())
    stem=sum(d['hours'] for d in data.values() if d['flags']['use_cnn'])
    assert round(total,2)==50.61 and round(stem,2)==48.87
    assert round(100*stem/total,1)==96.6

def main():
    old=read('chapter5.tex')
    new=read('chapter5_shortened.tex')
    halves=[read(f'chapter5_short_half{i}.tex') for i in (1,2)]
    assert new==HEADER+'\n\n'.join(s.rstrip() for s in halves)+'\n'
    assert labels(new)==labels(old), 'Label content/order changed'
    assert len(labels(new))==len(set(labels(new)))==67
    assert len(sections(new))==len(sections(old))==8
    assert new.count(r'\subsection{')==old.count(r'\subsection{')==39
    assert keys(new)==keys(old) and len(keys(new))==6
    bibkeys=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'bib/thesis.bib').read_text(encoding='utf-8')))
    assert set(keys(new))<=bibkeys
    assert not re.search(r'Chapter[~\s]+\d|Section[~\s]+\d\.\d',clean(new),re.I)
    for name in ['chapter5_shortened.tex','chapter5_short_half1.tex','chapter5_short_half2.tex',
                 'chapter5_short_acronyms.tex','list_of_acronyms.tex','list_of_tables.tex','list_of_figures.tex']:
        syntax(read(name))
    old_floats,new_floats=floats(old),floats(new)
    assert list(old_floats)==list(new_floats) and len(new_floats)==19
    for label,text in new_floats.items():
        assert re.search(r'\\caption\[[^\]]+\]\{',text), label
        assert label in refs(new), ('Float lacks prose reference',label)
        assert label in read('list_of_tables.tex'), ('Float absent from documented list',label)
        if label.startswith('tab:'):
            assert [normalise_row(r) for r in table_rows(text)]==[
                normalise_row(r) for r in table_rows(old_floats[label])], ('Table data changed',label)
        else:
            paths=re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}',text)
            assert len(paths)==1 and Path(paths[0]).parent == Path('figures')
            # Overleaf holds the images in figures/. This local checkout keeps
            # the same source PNGs under report_figures_thesis/ instead.
            source = ROOT/paths[0]
            if not source.is_file():
                source = ROOT/'report_figures_thesis'/Path(paths[0]).name
            assert source.is_file(), ('Missing source image', paths[0])
            oldpath=re.search(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}',old_floats[label]).group(1)
            assert Path(paths[0]).name==Path(oldpath).name
    shared=clean(read('list_of_tables.tex'))
    assert shared.count(r'\listoftables')==shared.count(r'\listoffigures')==1
    acronyms=re.findall(r'\\item\[([^\]]+)\]',read('list_of_acronyms.tex'))
    assert len(acronyms)==len(set(acronyms))
    for companion in ['chapter5_short_acronyms.tex','chapter2_short_acronyms.tex']:
        assert set(re.findall(r'\\item\[([^\]]+)\]',read(companion)))<=set(acronyms)
    # Both original and shortened prior chapters remain compatible.
    for suffixes in product(['','_shortened'],repeat=3):
        others=read('introduction.tex')+read('chapter6.tex')+'\n'.join(
            read(f'chapter{n}{s}.tex') for n,s in zip([2,3,4],suffixes))
        integrated=new+others
        baseline=old+others
        assert not [k for k,v in Counter(labels(integrated)).items() if v>1]
        assert refs(integrated)-set(labels(integrated)) <= refs(baseline)-set(labels(baseline))
    verify_tables(new_floats,result_data())
    summary={'original_words':words(old),'shortened_words':words(new),
             'half_words':[words(h) for h in halves], 'sections':[]}
    for i,((title,a),(_,b)) in enumerate(zip(sections(old),sections(new)),1):
        summary['sections'].append({'section':f'5.{i}','title':title,'before':words(a),'after':words(b)})
    print(json.dumps(summary,indent=2))
    print('PASS: exact assembly; all67 labels, six citation keys/order, ten tables and nine figures retained.')
    print('PASS: table numeric cells unchanged; 12 configurations and20 pair effects match source results.')
    print('PASS: eight chapter-integration combinations, automatic lists, acronym coverage, static syntax.')
    print('Figure references target Overleaf figures/; source images verified against this local checkout.')
    print('LaTeX rendering, generated list pages and pagination require the external thesis master/engine.')

if __name__=='__main__':
    main()
