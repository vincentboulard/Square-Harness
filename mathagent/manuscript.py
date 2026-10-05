"""A map of a manuscript: sections, numbered statements, their proofs and what each proof uses.

Pure and deterministic: no model call, no network. The review workflows use it so
that each model call receives one proof with exactly what it needs (the statements
it refers to and the equations it cites), instead of reading the paper blindly.
Line numbers are 1-based and inclusive, as in manuscript locators [M1:L4-L9].
It reads TeX sources and plain text (Markdown, or the text extracted from a PDF);
the text heuristics can miss structure, and the map then says so in `warnings`.
"""
import re

KNOWN = ('theorem', 'lemma', 'proposition', 'corollary', 'definition', 'assumption', 'hypothesis',
         'remark', 'claim', 'conjecture', 'example', 'notation', 'condition', 'problem', 'question')
PROVABLE = ('theorem', 'lemma', 'proposition', 'corollary', 'claim')
# Environment names used without \newtheorem in many papers and notes.
DEFAULT_ENVS = {name: name for name in KNOWN}
DEFAULT_ENVS.update({'thm': 'theorem', 'lem': 'lemma', 'prop': 'proposition', 'cor': 'corollary',
                     'defn': 'definition', 'defi': 'definition', 'dfn': 'definition', 'rem': 'remark',
                     'rmk': 'remark', 'ass': 'assumption', 'hyp': 'hypothesis', 'conj': 'conjecture',
                     'ex': 'example', 'exa': 'example', 'clm': 'claim'})
TYPE_WORDS = '|'.join(name.capitalize() for name in KNOWN)
NUMBER = r'(?:[A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)*|[A-Z])'
TEXT_STATEMENT = re.compile(r'^\s*(?:#{1,6}\s*)?[*_]{0,2}\s*(' + TYPE_WORDS + r')(?:\s+(' + NUMBER + r'))?'
                            r'(?:[*_]{0,2}\s*(\([^()]{0,90}(?:\([^()]*\)[^()]{0,40})?\))[*_]{0,2}\s*[.:]?|\.|:|[*_]{2})[*_]{0,2}(?=\s|$)')
TEXT_PROOF = re.compile(r'^\s*(?:#{1,6}\s*)?[*_]{0,2}\s*Proof\b[*_]{0,2}\s*(?=[.:(]|of\b|that\b|for\b|sketch\b|$)(.{0,160})')
TEXT_SECTION = re.compile(r'^\s*(\d+|[A-Z])\.\s+([A-Z][^.]{2,90})$')
TEXT_SUBSECTION = re.compile(r'^\s*((?:\d+|[A-Z])(?:\.\d+)+)\.?\s+([A-Z][^.]{1,90})\.(?:\s|$)')
MD_HEADING = re.compile(r'^\s*(#{1,6})\s+(.*\S)')
END_MARK = re.compile(r'[□∎■]|\\qed\b|\bQ\.E\.D\.?')
TEXT_REF = re.compile(r'\b(' + TYPE_WORDS + r')s?\s+(' + NUMBER + r'(?:\s*(?:,|and|&|or)\s*' + NUMBER + r')*)')
TEXT_CITE = re.compile(r'\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)(?:\s*,\s*([^\[\]]{1,80}?))?\]')
TEXT_EQUATION = re.compile(r'\((\d+\.\d+|[A-Z]\.\d+)\)')
PLACE = re.compile(r'\b(?:Theorem|Thm|Lemma|Proposition|Prop|Corollary|Cor|Section|Sec|Chapter|Ch|Definition|'
                   r'Remark|Example|Equation|Eq|Appendix|Formula|Page|p|pp)\b\.?', re.I)

TEX_NEWTHEOREM = re.compile(r'\\newtheorem(\*?)\s*\{([^}]+)\}\s*(?:\[([^\]]+)\])?\s*\{([^}]+)\}\s*(?:\[([^\]]+)\])?')
TEX_DECLARE = re.compile(r'\\declaretheorem\s*(?:\[([^\]]*)\])?\s*\{([^}]+)\}')
TEX_BEGIN = re.compile(r'\\begin\{([A-Za-z*]+)\}')
TEX_END = re.compile(r'\\end\{([A-Za-z*]+)\}')
TEX_SECTION = re.compile(r'\\(section|subsection|subsubsection)(\*?)\s*(?:\[[^\]]*\])?\s*\{')
TEX_LABEL = re.compile(r'\\label\{([^}]+)\}')
TEX_REF = re.compile(r'\\(?:[cC]ref|ref|autoref|eqref|namecref|vref|Cpageref|pageref)\*?\{([^}]+)\}')
TEX_CITE = re.compile(r'\\(?:cite|citep|citet|citealp|parencite|textcite|autocite)\*?\s*(?:\[([^\]]*)\])?\s*(?:\[([^\]]*)\])?\s*\{([^}]+)\}')
TEX_BIBITEM = re.compile(r'\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
EQUATION_ENVS = ('equation', 'equation*', 'align', 'align*', 'gather', 'gather*', 'multline', 'multline*',
                 'eqnarray', 'eqnarray*', 'flalign', 'flalign*')


def _strip_comment(line):
    """Drop a TeX comment, keeping escaped percent signs."""
    match = re.search(r'(?<!\\)%', line)
    return line[:match.start()] if match else line


def _braced(text, start):
    """Text of the brace group opening at or after `start` (best effort, single line)."""
    begin = text.find('{', start)
    if begin < 0:
        return ''
    depth = 0
    for index in range(begin, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[begin + 1:index]
    return text[begin + 1:]


def detect_format(path, lines):
    head = '\n'.join(lines[:400])
    if str(path).lower().endswith('.tex') or re.search(r'\\begin\{(?:document|theorem|lemma|proof)\}|\\section\{', head):
        return 'tex'
    if str(path).lower().endswith('.md'):
        return 'markdown'
    return 'text'


def _unit(kind, name, number, label, title, line, section, appendix):
    return {'id': '', 'type': kind, 'name': name, 'number': number, 'label': label, 'title': title,
            'statement': [line, line], 'proofs': [], 'refs': [], 'statement_refs': [], 'cites': [],
            'equations': [], 'section': section, 'appendix': appendix, 'main': False}


def _finish(result):
    units = result['units']
    for index, unit in enumerate(units, 1):
        unit['id'] = f'U{index}'
    result['units'] = units
    _mark_main(result)
    return result


def _mark_main(result):
    units = result['units']
    flagged = [u for u in units if re.search(r'\bmain\b', (u['label'] + ' ' + u['title']).lower())]
    theorems = [u for u in units if u['type'] == 'theorem']
    first = result['sections'][0]['number'] if result['sections'] else None
    intro = [u for u in theorems if first is not None and u['section'] == first]
    if flagged:
        chosen = flagged
    elif intro:
        chosen = intro
    elif theorems:
        chosen = theorems if len(theorems) <= 4 else theorems[:3]
    else:
        provable = [u for u in units if u['type'] in PROVABLE]
        chosen = provable if len(provable) <= 2 else sorted(provable, key=proof_length, reverse=True)[:2]
    for unit in chosen:
        unit['main'] = True


def proof_length(unit):
    return sum(b - a + 1 for a, b in unit['proofs'])


# -- TeX ------------------------------------------------------------------------------

def _tex_theorems(clean):
    """Environment name → (type, shared counter, within, numbered) from the preamble."""
    envs = {}
    text = '\n'.join(clean)
    for star, env, shared, name, within in TEX_NEWTHEOREM.findall(text):
        kind = name.strip().strip('\\').lower()
        kind = next((k for k in KNOWN if kind.startswith(k)), kind.split()[0] if kind.split() else env)
        envs[env.strip()] = {'type': kind, 'name': name.strip(), 'counter': (shared or env).strip(),
                             'within': (within or '').strip(), 'numbered': not star}
    for options, env in TEX_DECLARE.findall(text):
        opts = dict(re.findall(r'(\w+)\s*=\s*([^,]+)', options or ''))
        name = opts.get('name', env).strip()
        kind = next((k for k in KNOWN if name.lower().startswith(k)), name.lower())
        counter = (opts.get('sibling') or opts.get('numberlike') or env).strip()
        envs[env.strip()] = {'type': kind, 'name': name, 'counter': counter,
                             'within': (opts.get('numberwithin') or opts.get('parent') or '').strip(),
                             'numbered': opts.get('numbered', 'yes').strip() != 'no'}
    declared = bool(envs)
    for env, kind in DEFAULT_ENVS.items():
        envs.setdefault(env, {'type': kind, 'name': kind.capitalize(), 'counter': env, 'within': '',
                              'numbered': False})
    return envs, declared


def _parse_tex(lines):
    clean = [_strip_comment(line) for line in lines]
    envs, declared = _tex_theorems(clean)
    result = {'format': 'tex', 'lines': len(lines), 'sections': [], 'units': [], 'bibliography': {},
              'abstract': None, 'labels': {}, 'equations': {}, 'warnings': []}
    if not declared:
        result['warnings'].append('No \\newtheorem declarations found; statements are named by their labels, without numbers.')
    counters, section, appendix, section_number = {}, None, False, 0
    units, open_unit, open_proof, proof_owner, open_env = [], None, None, None, []
    equation_start, abstract_start, bib_key, bib_start = None, None, None, None
    body_started = not any('\\begin{document}' in line for line in clean)
    for number, line in enumerate(clean, 1):
        if not body_started:
            if '\\begin{document}' in line:
                body_started = True
            continue
        if '\\appendix' in line:
            appendix, section_number = True, 0
        match = TEX_SECTION.search(line)
        if match and open_proof is None:
            title = _braced(line, match.end() - 1)
            if match.group(1) == 'section':
                if not match.group(2):
                    section_number += 1
                    section = chr(64 + section_number) if appendix else str(section_number)
                    for name, value in list(counters.items()):
                        if value.get('within') == 'section':
                            counters[name]['n'] = 0
                else:
                    section = title
                result['sections'].append({'number': section, 'title': title, 'line': number, 'appendix': appendix})
                if re.match(r'\s*(references|bibliography)\s*$', title, re.I):
                    appendix = True
        for label in TEX_LABEL.findall(line):
            result['labels'][label] = number
        for env in TEX_BEGIN.findall(line):
            if env == 'abstract':
                abstract_start = number
            elif env in EQUATION_ENVS:
                equation_start = number
            elif env == 'proof' and open_unit is None:
                tail = line[line.find('\\begin{proof}') + len('\\begin{proof}'):]
                option = tail[tail.find('[') + 1:tail.find(']')] if tail.lstrip().startswith('[') else ''
                labels = [lab for group in TEX_REF.findall(option) for lab in group.split(',')]
                owner = next((u for u in units for lab in labels if u['label'] == lab.strip()), None)
                if owner is None and option:
                    owner = _owner_by_name(units, option)
                if owner is None:
                    owner = _pending_owner(units, proof_owner)
                if owner is None:
                    owner = _unit('proof', f'Proof at line {number}', '', '', option.strip(), number, section, appendix)
                    owner['statement'] = None
                    units.append(owner)
                open_proof, proof_owner = [number, number], owner
                open_env.append('proof')
            elif env in envs and open_unit is None and open_proof is None:
                spec = envs[env]
                shown = ''
                if spec['numbered'] and declared:
                    counter = counters.setdefault(spec['counter'], {'n': 0, 'within': envs.get(spec['counter'], spec)['within']})
                    counter['n'] += 1
                    within = counter['within'] or spec['within']
                    shown = f'{section}.{counter["n"]}' if within == 'section' and section else str(counter['n'])
                tail = line[line.find('\\begin{' + env + '}') + len(env) + 8:]
                title = tail[tail.find('[') + 1:tail.find(']')] if tail.lstrip().startswith('[') else ''
                open_unit = _unit(spec['type'], '', shown, '', title.strip(), number, section, appendix)
                open_unit['_env'] = env
        if open_unit is not None:
            labels = TEX_LABEL.findall(line)
            if labels and not open_unit['label']:
                open_unit['label'] = labels[0]
            for group in TEX_REF.findall(line):
                open_unit['statement_refs'] += [g.strip() for g in group.split(',')]
            if '\\end{' + open_unit['_env'] + '}' in line:
                open_unit['statement'][1] = number
                kind = open_unit['type'].capitalize()
                open_unit['name'] = (f'{kind} {open_unit["number"]}' if open_unit['number'] else
                                     f'{kind} ({open_unit["title"] or open_unit["label"]})' if open_unit['title'] or open_unit['label']
                                     else f'{kind} at line {open_unit["statement"][0]}')
                del open_unit['_env']
                units.append(open_unit)
                open_unit = None
        if open_proof is not None:
            for group in TEX_REF.findall(line):
                proof_owner['refs'] += [g.strip() for g in group.split(',')]
            for note, note2, keys in TEX_CITE.findall(line):
                for key in keys.split(','):
                    proof_owner['cites'].append({'key': key.strip(), 'note': (note2 or note).strip(), 'line': number})
            if '\\end{proof}' in line:
                open_proof[1] = number
                proof_owner['proofs'].append(open_proof)
                open_proof = None
        for env in TEX_END.findall(line):
            if env == 'abstract' and abstract_start:
                result['abstract'] = [abstract_start, number]
            elif env in EQUATION_ENVS and equation_start:
                for label in TEX_LABEL.findall('\n'.join(clean[equation_start - 1:number])):
                    result['equations'][label] = [equation_start, number]
                equation_start = None
        match = TEX_BIBITEM.search(line)
        if match or '\\end{thebibliography}' in line:
            if bib_key:
                result['bibliography'][bib_key] = ' '.join(l.strip() for l in clean[bib_start - 1:number - (0 if not match else 1)])[:400]
            bib_key, bib_start = (match.group(1).strip(), number) if match else (None, None)
    if open_proof is not None:
        open_proof[1] = len(lines)
        proof_owner['proofs'].append(open_proof)
        result['warnings'].append(f'A proof starting at line {open_proof[0]} is not closed.')
    if open_unit is not None:
        result['warnings'].append(f'A statement starting at line {open_unit["statement"][0]} is not closed.')
    by_label = {u['label']: u for u in units if u['label']}
    for unit in units:
        unit['equations'] = [result['equations'][r] for r in dict.fromkeys(unit['refs']) if r in result['equations']]
        unit['refs'] = list(dict.fromkeys(by_label[r]['label'] for r in unit['refs'] if r in by_label and by_label[r] is not unit))
        unit['statement_refs'] = list(dict.fromkeys(by_label[r]['label'] for r in unit['statement_refs'] if r in by_label and by_label[r] is not unit))
    result['units'] = units
    _finish(result)
    ids = {u['label']: u['id'] for u in units if u['label']}
    for unit in units:
        unit['refs'] = [ids[r] for r in unit['refs']]
        unit['statement_refs'] = [ids[r] for r in unit['statement_refs']]
    return result


def _owner_by_name(units, text):
    match = re.search(r'(' + TYPE_WORDS + r')\s*~?\s*(' + NUMBER + r')', text)
    if not match:
        return None
    kind, number = match.group(1).lower(), match.group(2)
    return next((u for u in units if u['type'] == kind and u['number'] == number), None)


def _pending_owner(units, last_owner):
    """The statement a bare 'Proof' belongs to: the latest provable statement without a proof."""
    for unit in reversed(units):
        if unit['type'] in PROVABLE:
            return unit if not unit['proofs'] else None
        if unit['type'] in ('remark', 'example', 'definition', 'notation'):
            continue
        return None
    return None


# -- plain text (Markdown or PDF extraction) ---------------------------------------------

def _parse_text(lines, fmt):
    result = {'format': fmt, 'lines': len(lines), 'sections': [], 'units': [], 'bibliography': {},
              'abstract': None, 'labels': {}, 'equations': {}, 'warnings': []}
    units, section, appendix, top = [], None, False, 0
    boundaries = set()  # lines where a statement, proof or section starts
    events = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        heading = MD_HEADING.match(line) if fmt == 'markdown' else None
        statement = TEXT_STATEMENT.match(line)
        proof = TEXT_PROOF.match(line) if not statement else None
        if statement:
            events.append(('statement', number, statement))
            boundaries.add(number)
            continue
        if proof:
            events.append(('proof', number, proof))
            boundaries.add(number)
            continue
        if re.match(r'^\s*(?:#{1,6}\s*)?(?:\d+\.\s+)?(references|bibliography)\s*$', stripped, re.I) and not any(e[0] == 'references' for e in events):
            events.append(('references', number, None))
            boundaries.add(number)
            continue
        if re.match(r'^\s*(?:#{1,6}\s*)?abstract\b', stripped, re.I) and result['abstract'] is None and number < 200:
            result['abstract'] = [number, number]
        if heading:
            title = heading.group(2).strip('*_ ')
            num = re.match(r'^(\d+|[A-Z])(?:\.\d+)*\.?\s', title)
            events.append(('section', number, (num.group(1) if num and len(heading.group(1)) <= 2 else None, title, len(heading.group(1)))))
            boundaries.add(number)
            continue
        match = TEXT_SECTION.match(line)
        if match and len(stripped) < 100:
            label = match.group(1)
            if (label.isdigit() and int(label) == top + 1) or (not label.isdigit() and (appendix or top >= 1) and len(label) == 1):
                if label.isdigit():
                    top = int(label)
                events.append(('section', number, (label, match.group(2).strip(), 1)))
                boundaries.add(number)
                continue
        match = TEXT_SUBSECTION.match(line)
        if match and match.group(1).split('.')[0] in (str(top), section or ''):
            events.append(('subsection', number, (match.group(1), match.group(2).strip())))
            boundaries.add(number)
    if result['abstract']:
        nxt = min([b for b in boundaries if b > result['abstract'][0]] or [len(lines) + 1])
        result['abstract'][1] = min(nxt - 1, result['abstract'][0] + 40)
    ordered = sorted(boundaries)

    def next_boundary(after):
        for b in ordered:
            if b > after:
                return b
        return len(lines) + 1

    in_references, references_line, owner = False, None, None
    for kind, number, data in events:
        if kind == 'references':
            in_references, references_line = True, number
            continue
        if in_references and kind != 'section':
            continue
        if kind == 'section':
            label, title, level = data
            in_references = False
            if level == 1 or (fmt == 'markdown' and level <= 2):
                if label and not label.isdigit() and len(label) == 1:
                    appendix = True
                if re.match(r'appendix', title, re.I):
                    appendix = True
                    letter = re.match(r'appendix\s+([A-Z])\b', title, re.I)
                    label = letter.group(1) if letter else label
                section = label or title
                result['sections'].append({'number': section, 'title': title, 'line': number, 'appendix': appendix})
            continue
        if kind == 'subsection':
            continue
        if kind == 'statement':
            word, num, title = data.group(1).lower(), data.group(2) or '', (data.group(3) or '').strip('() ')
            if num and any(u['type'] == word and u['number'] == num for u in units):
                continue  # a sentence starting with "Lemma 2.3." refers to it; the statement came first
            name = f'{word.capitalize()} {num}' if num else word.capitalize() + (f' ({title})' if title else '')
            unit = _unit(word, name, num, '', title, number,
                         section, appendix or bool(re.match(r'[A-Z]\.', num)))
            end = next_boundary(number) - 1
            unit['statement'] = [number, max(number, min(end, number + 59))]
            if end > number + 59:
                result['warnings'].append(f'{unit["name"]}: statement end not found; first 60 lines kept.')
            units.append(unit)
            continue
        # a proof
        detail = data.group(1).strip(' .:*_')
        target = None
        if re.match(r'of\b', detail, re.I):
            target = _owner_by_name(units, detail)
            if target is None:
                result['warnings'].append(f'Line {number}: "Proof {detail[:60]}" names no statement found in the map.')
        elif re.match(r'that\b', detail, re.I) and owner is not None:
            target = owner  # a separate step of the previous proof
        if target is None and not re.match(r'of\b', detail, re.I):
            target = _pending_owner(units, owner)
        if target is None:
            target = _unit('proof', f'Proof at line {number}', '', '', detail[:90], number, section, appendix)
            target['statement'] = None
            units.append(target)
        end = next_boundary(number) - 1
        stop = next((n for n in range(number, end + 1) if END_MARK.search(lines[n - 1])), None)
        target['proofs'].append([number, stop or end])
        owner = target
    if references_line:
        result['bibliography'] = _text_bibliography(lines, references_line)
    keys = set(result['bibliography'])
    lookup = {(u['type'], u['number']): u for u in units if u['number']}
    for unit in units:
        for a, b in unit['proofs']:
            text = '\n'.join(lines[a - 1:b])
            unit['refs'] += _text_refs(text, lookup, unit)
            unit['cites'] += _text_cites(lines, a, b, keys)
            unit['equations'] += _text_equations(lines, text, a, b)
        if unit['statement']:
            a, b = unit['statement']
            unit['statement_refs'] = _text_refs('\n'.join(lines[a - 1:b]), lookup, unit)
            unit['equations'] += _text_equations(lines, '\n'.join(lines[a - 1:b]), a, b)
        unit['refs'] = list(dict.fromkeys(unit['refs']))
        unit['statement_refs'] = list(dict.fromkeys(unit['statement_refs']))
        unit['equations'] = [list(r) for r in dict.fromkeys(tuple(r) for r in unit['equations'])][:12]
    if not units:
        result['warnings'].append('No numbered statements or proofs were recognised; the review falls back to fixed-size chunks.')
    result['units'] = units
    _finish(result)
    ids = {id(u): u['id'] for u in units}
    for unit in units:
        unit['refs'] = [ids[r] for r in unit['refs']]
        unit['statement_refs'] = [ids[r] for r in unit['statement_refs']]
    return result


def _text_refs(text, lookup, unit):
    found = []
    for word, numbers in TEXT_REF.findall(text):
        for num in re.findall(NUMBER, numbers):
            other = lookup.get((word.lower(), num))
            if other is not None and other is not unit:
                found.append(id(other))
    return found


def _text_cites(lines, a, b, keys):
    found = []
    for number in range(a, b + 1):
        for group, note in TEXT_CITE.findall(lines[number - 1]):
            refs = [k.strip() for k in re.split(r'[,;]', group)]
            if keys and not all(k in keys for k in refs):
                continue
            if not keys and not (note and PLACE.search(note)):
                continue
            for key in refs:
                found.append({'key': key, 'note': note.strip(), 'line': number})
    return found


def _text_equations(lines, text, a, b):
    """Lines where the equations a passage cites are displayed: '(2.27)' at a line start or end."""
    wanted = set(TEXT_EQUATION.findall(text))
    found = []
    for tag in sorted(wanted):
        pattern = re.compile(r'^\s*\(' + re.escape(tag) + r'\)\s|\s\(' + re.escape(tag) + r'\)\s*$')
        for number, line in enumerate(lines, 1):
            if a <= number <= b:
                continue
            if pattern.search(line):
                found.append([max(1, number - 2), min(len(lines), number + 1)])
                break
    return found


def _text_bibliography(lines, start):
    entries, key, buffer = {}, None, []
    for line in lines[start:]:
        match = re.match(r'^\s*\[(\w{1,12})\]\s+(.*)', line)
        if match:
            if key:
                entries[key] = ' '.join(buffer)[:400]
            key, buffer = match.group(1), [match.group(2).strip()]
        elif key and line.strip():
            buffer.append(line.strip())
    if key:
        entries[key] = ' '.join(buffer)[:400]
    return entries


# -- public interface ---------------------------------------------------------------

def build(content, path=''):
    """Map a manuscript; `content` is the pinned text, `path` only helps guess its format."""
    lines = content.splitlines()
    fmt = detect_format(path, lines)
    return _parse_tex(lines) if fmt == 'tex' else _parse_text(lines, fmt)


def unit(mapping, unit_id):
    return next((u for u in mapping['units'] if u['id'] == unit_id), None)


def resolve(mapping, target):
    """The unit a request names: 'Lemma 3.2', 'lem:compact', 'Theorem 1.4', or 'the main theorem'."""
    target = (target or '').strip()
    if not target:
        return None
    units = mapping['units']
    for u in units:
        if u['label'] and u['label'] == target:
            return u
    match = re.search(r'\b(' + TYPE_WORDS + r'|Thm|Lem|Prop|Cor)\.?\s*~?\s*(' + NUMBER + r')\b', target, re.I)
    if match:
        word = match.group(1).lower()
        word = {'thm': 'theorem', 'lem': 'lemma', 'prop': 'proposition', 'cor': 'corollary'}.get(word, word)
        found = [u for u in units if u['number'] == match.group(2) and u['type'] == word]
        if found:
            return found[0]
        found = [u for u in units if u['number'] == match.group(2)]
        if len(found) == 1:
            return found[0]
    for u in units:
        if u['label'] and re.search(r'(?<![\w:])' + re.escape(u['label']) + r'(?![\w:])', target):
            return u
    if re.search(r'\bmain\b', target, re.I):
        main = [u for u in units if u['main'] and u['proofs']]
        if len(main) == 1:
            return main[0]
    titled = [u for u in units if u['title'] and u['title'].lower() in target.lower() and len(u['title']) > 3]
    if len(titled) == 1:
        return titled[0]
    return None


def checkable(mapping):
    """Units with a proof, in reading order."""
    return [u for u in mapping['units'] if u['proofs']]


def priority(mapping, depth=None, main_ids=None, appendix=True):
    """Units with proofs: the main results first, then what their proofs use (breadth first).

    depth None = then every remaining unit with a proof; appendix False leaves out the appendices
    unless a main proof depends on them.
    """
    by_id = {u['id']: u for u in mapping['units']}
    main = [by_id[i] for i in (main_ids or []) if i in by_id] or [u for u in mapping['units'] if u['main']]
    order, seen = [], set()
    frontier = [u['id'] for u in main]
    level = 0
    while frontier and (depth is None or level <= depth):
        nxt = []
        for uid in frontier:
            if uid in seen:
                continue
            seen.add(uid)
            u = by_id[uid]
            if u['proofs']:
                order.append(uid)
            nxt += [r for r in u['refs'] + u['statement_refs'] if r not in seen]
        frontier, level = nxt, level + 1
    if depth is None:
        for u in mapping['units']:
            if u['id'] not in seen and u['proofs'] and (appendix or not u['appendix']):
                order.append(u['id'])
    return order


def chunks(total, size=120):
    """Fixed-size line ranges, the fallback when no structure was recognised."""
    return [[a, min(total, a + size - 1)] for a in range(1, total + 1, size)]


def outline(mapping, limit=80):
    """A compact list of statements for a model prompt."""
    rows = []
    for u in mapping['units'][:limit]:
        where = f'L{u["statement"][0]}' if u['statement'] else 'no statement'
        proof = ', proof ' + ', '.join(f'L{a}-L{b}' for a, b in u['proofs']) if u['proofs'] else ', no proof here'
        title = f' ({u["title"]})' if u['title'] else ''
        rows.append(f'{u["id"]}: {u["name"]}{title}, section {u["section"]}, {where}{proof}{", appendix" if u["appendix"] else ""}')
    if len(mapping['units']) > limit:
        rows.append(f'… and {len(mapping["units"]) - limit} more statements.')
    return '\n'.join(rows)
