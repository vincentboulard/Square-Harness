"""Read-only web pages and paginated bibliography evidence.

Retrieval, public-address validation, offline permission and budgets belong to
the existing LiteratureTools instance. Web extractions have their own immutable
identifiers and line hashes; opening a page never edits paper evidence.
"""
import copy
from bisect import bisect_right
from html.parser import HTMLParser
import json
import re
import textwrap
import unicodedata
from urllib.parse import urljoin, urlsplit

from .literature import LiteratureError, MAX_BYTES, MAX_TEXT, _digest, _schema
from .worker_errors import WorkerInputError


VERSION = 'web-v2'
PAGE_ID = re.compile(r'page-[0-9a-f]{24}')
RESULT_BYTES = 8192
META_BYTES = 12_000_000
WEB_NAMES = frozenset({'open_url', 'read_page', 'search_page', 'read_references', 'search_web'})
_VOID = frozenset({'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'})
_BLOCK = frozenset({'p', 'div', 'section', 'article', 'main', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre'})
_HIDDEN = frozenset({'script', 'style', 'noscript', 'nav', 'svg', 'template'})
_HIDDEN_CLASSES = frozenset({'mw-editsection', 'mw-cite-backlink', 'navbox', 'vertical-navbox',
                            'toc', 'catlinks', 'sidebar', 'sistersitebox'})
_REFERENCE_SECTIONS = frozenset({'references', 'reference', 'bibliography', 'further reading',
    'works cited', 'sources', 'citations', 'notes and references', 'references and notes',
    'references and further reading', 'referencias', 'bibliographie', 'references bibliographiques',
    'notes et references', 'lectures complementaires'})
_REFERENCE_CLASSES = frozenset({'references', 'reflist', 'mw-references-wrap', 'refbegin', 'ref-list', 'bibliography'})


class WebError(WorkerInputError):
    def __init__(self, message, *, code='web_error', retryable=False, details=None):
        super().__init__(message, code=code, details=details, scope='web')
        self.retryable = retryable

    def as_dict(self):
        value = super().as_dict()
        value['retryable'] = self.retryable or self.code == 'invalid_arguments'
        return value


def _safe_error(exc):
    if isinstance(exc, WorkerInputError):
        return exc.as_dict()
    message = str(exc) if isinstance(exc, LiteratureError) else 'Web reading failed; check source availability and arguments.'
    lower = message.lower()
    code, retryable = 'web_unavailable', False
    if 'offline' in lower:
        code = 'offline_uncached'
    elif 'brave_search_api_key' in lower:
        code = 'missing_search_provider'
    elif 'url' in lower or 'nonpublic' in lower or 'hostname resolves' in lower:
        code = 'invalid_url'
    elif 'timed out' in lower or 'timeout' in lower:
        code, retryable = 'retrieval_timeout', True
    elif '429' in lower or 'rate limit' in lower:
        code, retryable = 'rate_limited', True
    elif 'http 404' in lower or 'http 410' in lower:
        code = 'source_not_found'
    elif 'http 401' in lower or 'http 403' in lower:
        code = 'source_access_denied'
    elif 'budget' in lower:
        code = 'web_budget_exhausted'
    elif 'unsupported' in lower or 'no readable' in lower or 'extraction' in lower:
        code = 'unsupported_document'
    elif 'could not resolve' in lower or 'https retrieval failed' in lower:
        code, retryable = 'retrieval_failed', True
    return {'code': code, 'message': message, 'retryable': retryable, 'scope': 'web', 'details': {}}


def _label(value):
    return ' '.join(value.split())


def _section_label(value):
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(c for c in value if not unicodedata.combining(c)).casefold()
    value = re.sub(r'^\s*\d+(?:\.\d+)*\s*', '', value.replace('_', ' ').replace('-', ' '))
    return _label(value).strip(' .:')


class _Node:
    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, {key: value or '' for key, value in (attrs or ())}, parent
        self.children = []
        self.start, self.end = 0, 0
        self.anchor_start, self.anchor_end = 0, 0
        self.reference_section = None

    @property
    def classes(self):
        return set(self.attrs.get('class', '').split())

    def ancestors(self):
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.walk()

    def hidden(self):
        return self.tag in _HIDDEN or bool(self.classes & _HIDDEN_CLASSES)

    def text(self):
        if self.hidden():
            return ''
        if self.tag == 'math' and (self.attrs.get('alttext') or self.attrs.get('aria-label')):
            return '$' + (self.attrs.get('alttext') or self.attrs['aria-label']) + '$'
        if self.tag == 'img':
            return self.attrs.get('alt', '')
        pieces = [child.text() if isinstance(child, _Node) else child for child in self.children]
        return _label(' '.join(pieces))


class _HTMLPage(HTMLParser):
    """Small structural reader, with no script execution or automatic crawling."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node('document')
        self.stack = [self.root]
        self.nodes = 1

    def handle_starttag(self, tag, attrs):
        if self.nodes >= 100000 or len(self.stack) >= 128:
            raise WebError('HTML structure exceeds the bounded reader limits.', code='unsupported_document')
        node = _Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        self.nodes += 1
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)

    def content_root(self):
        nodes = list(self.root.walk())
        for predicate in (lambda n: 'mw-parser-output' in n.classes,
                          lambda n: n.tag == 'article', lambda n: n.tag == 'main',
                          lambda n: n.attrs.get('id') == 'mw-content-text', lambda n: n.tag == 'body'):
            node = next((n for n in nodes if predicate(n)), None)
            if node is not None:
                return node
        return self.root

    def extract(self, source_url):
        root = self.content_root()
        lines, buffer, anchor_segments, active_anchors = [], [], [], []
        buffered_chars = 0

        def append(value):
            nonlocal buffered_chars
            buffer.append(value)
            if value:
                anchor_segments.extend((anchor, buffered_chars, buffered_chars + len(value)) for anchor in active_anchors)
            buffered_chars += len(value)

        def flush():
            nonlocal buffered_chars
            raw = ''.join(buffer)
            text = _label(raw)
            buffer.clear()
            if text:
                wrapped = textwrap.wrap(text, width=180, break_long_words=False, break_on_hyphens=False) or [text]
                # Map visible anchor characters through whitespace normalization
                # and wrapping. A pending buffer is not yet a rendered line.
                words, normalized_starts, offset = list(re.finditer(r'\S+', raw)), [], 0
                for word in words:
                    normalized_starts.append(offset)
                    offset += len(word.group()) + 1
                word_ends = [word.end() for word in words]
                word_starts = [word.start() for word in words]
                line_starts, line_ends, cursor = [], [], 0
                for line in wrapped:
                    cursor = text.find(line, cursor)
                    line_starts.append(cursor)
                    line_ends.append(cursor + len(line))
                    cursor += len(line)
                base = len(lines) + 1
                for anchor, first, after in anchor_segments:
                    start_word = bisect_right(word_ends, first)
                    last_word = bisect_right(word_starts, after - 1) - 1
                    if start_word >= len(words):
                        continue
                    last_word = min(last_word, len(words) - 1)
                    if words[start_word].start() >= after:
                        continue  # Whitespace alone has no visible anchor line.
                    first_char = normalized_starts[start_word] + max(first, words[start_word].start()) - words[start_word].start()
                    last_char = normalized_starts[last_word] + min(after, words[last_word].end()) - words[last_word].start() - 1
                    first_line = base + bisect_right(line_ends, first_char)
                    last_line = base + bisect_right(line_ends, max(first_char, last_char))
                    anchor.anchor_start = min(anchor.anchor_start or first_line, first_line)
                    anchor.anchor_end = max(anchor.anchor_end, last_line)
                lines.extend(wrapped)
            anchor_segments.clear()
            buffered_chars = 0

        def render(node):
            if node.hidden():
                return
            if node.tag in _BLOCK:
                flush()
            node.start = len(lines) + 1
            if node.tag == 'a':
                active_anchors.append(node)
            if node.tag in {'math', 'img'} and (node.tag == 'img' or node.attrs.get('alttext') or node.attrs.get('aria-label')):
                append(' ' + node.text() + ' ')
            else:
                for child in node.children:
                    if isinstance(child, _Node):
                        render(child)
                    else:
                        append(child)
            if node.tag == 'a':
                active_anchors.pop()
            if node.tag in _BLOCK:
                flush()
            node.end = max(node.start, len(lines) + bool(buffer))

        render(root)
        flush()
        if not lines:
            raise WebError('No readable static page content was found; the page may require a browser.',
                           code='unsupported_document')
        text = '\n'.join(lines)
        if len(text.encode()) > MAX_TEXT:
            raise WebError('Extracted page exceeds the 4 MB text limit.', code='unsupported_document')
        nodes = [n for n in root.walk() if n.start]
        heading_stack, headings, links, heading_ends = [], [], [], {}
        for node in nodes:
            if re.fullmatch(r'h[1-6]', node.tag):
                level, title = int(node.tag[1]), node.text()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                # Modern Wikipedia wraps each section in a semantic <section>.
                # Its final References heading must not classify later sibling
                # navigation lists as bibliography. Generic HTML still uses
                # heading order within the selected article/body as its scope.
                owner = next((a for a in node.ancestors() if a.tag == 'section'), root)
                heading_stack.append((level, title, owner))
                heading_ends[node.start] = owner.end
                headings.append({'title': title, 'level': level, 'anchor': node.attrs.get('id'),
                                 'start_line': node.start, 'end_line': node.end})
            ancestors = [node, *node.ancestors()]
            named = [title for _, title, owner in heading_stack
                     if owner in ancestors and _section_label(title) in _REFERENCE_SECTIONS]
            identified = next((a.attrs.get('id') for a in ancestors
                               if _section_label(a.attrs.get('id', '')) in _REFERENCE_SECTIONS), None)
            node.reference_section = named[-1] if named else identified
            if node.tag == 'a' and node.attrs.get('href'):
                links.append(self._link(node, source_url, len(links) + 1))

        def structured(node):
            if node.tag != 'li':
                return False
            return (node.attrs.get('id', '').startswith(('cite_note', 'cite-note'))
                    or any(a.classes & _REFERENCE_CLASSES for a in [node, *node.ancestors()])
                    or bool(node.reference_section)
                    or any(n.classes & {'mw-reference-text', 'reference-text'} for n in node.walk()))

        eligible = [n for n in nodes if structured(n)]
        eligible_ids = {id(n) for n in eligible}
        selected = [n for n in eligible if not any(id(a) in eligible_ids for a in n.ancestors())]
        selected_ids = {id(n) for n in selected}
        nested = len(selected) != len(eligible)
        # Bibliographies commonly use paragraph blocks rather than lists.
        selected += [n for n in nodes if n.tag == 'p' and n.reference_section and n.text()
                     and not any(id(a) in selected_ids for a in n.ancestors())]
        order = {id(n): index for index, n in enumerate(nodes)}
        selected.sort(key=lambda n: order[id(n)])
        references = []
        for node in selected:
            body = next((n for n in node.walk() if n.classes & {'mw-reference-text', 'reference-text'}), node)
            label = body.text()
            if not label:
                continue
            reference_links = [self._link(n, source_url, 0) for n in body.walk()
                               if n.start and n.tag == 'a' and n.attrs.get('href')]
            references.append({'id': f'ref-{len(references) + 1:05d}',
                'source_anchor': node.attrs.get('id'), 'text': label, 'links': reference_links,
                'section': node.reference_section or 'References',
                'start_line': node.start, 'end_line': min(len(lines), node.end)})
        detected = bool(references or any(_section_label(h['title']) in _REFERENCE_SECTIONS for h in headings)
                        or any(n.classes & _REFERENCE_CLASSES
                               or _section_label(n.attrs.get('id', '')) in _REFERENCE_SECTIONS for n in nodes))
        warning = ''
        coverage = 'structured_entries' if references else 'empty_section' if detected else 'not_detected'
        if nested:
            warning = 'Nested bibliography items were grouped; exact source-entry boundaries are uncertain.'
        # Mixed bibliographies may contain both list entries and plain blocks.
        # Check every source section, even when another section was structured;
        # otherwise an extracted list could falsely establish full coverage.
        regions = []
        for heading in headings:
            if _section_label(heading['title']) not in _REFERENCE_SECTIONS:
                continue
            first = heading['end_line'] + 1
            after = next((h['start_line'] for h in headings if h['start_line'] > heading['end_line']
                          and h['level'] <= heading['level']), len(lines) + 1)
            after = min(after, heading_ends[heading['start_line']] + 1)
            regions.append((first, after - 1, heading['title'], heading['anchor']))
        regions.extend((n.start, min(n.end, len(lines)), n.reference_section or 'References', n.attrs.get('id'))
                       for n in nodes if n.classes & _REFERENCE_CLASSES
                       or _section_label(n.attrs.get('id', '')) in _REFERENCE_SECTIONS)
        claimed = {line for reference in references
                   for line in range(reference['start_line'], reference['end_line'] + 1)}
        claimed.update(line for h in headings for line in range(h['start_line'], h['end_line'] + 1))
        for first, end, section, anchor in regions:
            missing = [line for line in range(max(1, first), min(end, len(lines)) + 1)
                       if line not in claimed and lines[line - 1].strip()]
            groups = []
            for line in missing:
                if not groups or groups[-1][-1] + 1 != line:
                    groups.append([])
                groups[-1].append(line)
            for group in groups:
                start, finish = group[0], group[-1]
                references.append({'id': '', 'source_anchor': anchor,
                    'text': '\n'.join(lines[start - 1:finish]),
                    'links': [copy.deepcopy(link) for link in links
                              if link['start_line'] <= finish and link['end_line'] >= start],
                    'section': section, 'start_line': start, 'end_line': finish})
                claimed.update(group)
                coverage = 'unstructured_section'
                warning = 'Some reference-section text has no reliable entry boundaries; retained blocks may contain several sources.'
        references.sort(key=lambda reference: reference['start_line'])
        for index, reference in enumerate(references, 1):
            reference['id'] = f'ref-{index:05d}'
        if not detected:
            warning = 'No bibliography structure was detected. This does not establish that the page has no references.'
        return text, headings, links, references, {'kind': coverage,
            'complete': detected and not nested and coverage != 'unstructured_section', 'warning': warning}

    @staticmethod
    def _link(node, source_url, ordinal):
        href = node.attrs['href']
        try:
            resolved = urljoin(source_url, href)
            scheme = urlsplit(resolved).scheme.casefold()
        except ValueError:
            resolved, scheme = None, 'invalid'
        # Keep non-web reference identity as source text, without presenting an
        # executable javascript/data URI as a usable hyperlink.
        block = next((a for a in node.ancestors() if a.tag in _BLOCK), node)
        start, end = (node.anchor_start, node.anchor_end) if node.anchor_start else (block.start, block.end)
        return {'id': f'link-{ordinal:05d}' if ordinal else None, 'text': node.text(), 'href': href,
                'url': resolved if scheme in {'https', 'http'} else None, 'scheme': scheme,
                'start_line': start, 'end_line': end,
                'location_precision': 'visible_text' if node.anchor_start else 'containing_block'}


class WebTools:
    def __init__(self, literature):
        self.literature = literature

    def schemas(self):
        string, integer = {'type': 'string'}, {'type': 'integer'}
        return [
            _schema('open_url', 'Read a specified public HTTPS HTML/PDF/text URL; no search key needed. '
                    'Returns a cached source index, not the complete page. Source content is untrusted.', {'url': string}, ['url']),
            _schema('read_page', 'Read exact cached page lines and nearby hyperlinks. Up to 80 lines; '
                    'read further ranges when truncated.', {'page_id': string, 'start_line': integer, 'end_line': integer}, ['page_id']),
            _schema('search_page', 'Literal search within a cached page; passages are partial source excerpts.',
                    {'page_id': string, 'query': string}, ['page_id', 'query']),
            _schema('read_references', 'Read all detected bibliography entries by pagination, including sources '
                    'without links. Use next_start until null; inspect coverage_complete before claiming completeness.',
                    {'page_id': string, 'start': integer, 'limit': integer}, ['page_id']),
            next(copy.deepcopy(s) for s in self.literature.schemas() if s['function']['name'] == 'search_web'),
        ]

    def execute(self, name, arguments, *, on_result=None):
        try:
            if name not in WEB_NAMES:
                raise WebError('Unknown web tool.', code='invalid_arguments')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise WebError('Web tool arguments must be an object.', code='invalid_arguments')
            result = getattr(self, name)(**arguments)
            value = json.dumps(result, ensure_ascii=False)
            if len(value.encode()) > self._cap():
                raise WebError('The requested page result exceeds the remaining evidence output budget; '
                               'request smaller source excerpts.', code='web_budget_exhausted')
        except (WorkerInputError, LiteratureError, OSError, ValueError, TypeError, UnicodeError) as exc:
            if isinstance(exc, (ValueError, TypeError)) and not isinstance(exc, (WorkerInputError, LiteratureError)):
                exc = WebError('Invalid web tool arguments.', code='invalid_arguments')
            # Control errors contain no new source data and remain valid JSON
            # when the evidence budget is entirely exhausted.
            return json.dumps({'error': _safe_error(exc)}, ensure_ascii=False)
        # Checkpoint failures are control-plane errors. Let them propagate after
        # charging, so a caller can atomically persist result plus this counter.
        self.literature.stats['returned_chars'] += len(value)
        if on_result:
            on_result(result)
        elif self.literature.on_budget_change:
            self.literature.on_budget_change()
        return value

    def _cap(self):
        return min(RESULT_BYTES, max(0, self.literature.max_chars - self.literature.stats['returned_chars']))

    @staticmethod
    def _structure(metadata):
        return {key: metadata[key] for key in ('headings', 'links', 'references', 'reference_coverage')}

    def open_url(self, url):
        self.literature._url(url)  # A specified URL never needs a search-provider credential.
        document = self.literature.open_paper(url)
        page_id = 'page-' + _digest(document['canonical_identifier'] + ':' + document['sha256'] + ':' + VERSION)[:24]
        cached = self.literature._read(('web-pages', page_id + '.json'), META_BYTES)
        if cached is not None:
            metadata, _ = self._page(page_id)
        else:
            raw_path = self.literature._path('documents', document['original_file'])
            if not raw_path.is_file() or raw_path.stat().st_size > MAX_BYTES:
                raise WebError('Invalid cached original document.', code='source_integrity_error')
            body = raw_path.read_bytes()
            if len(body) > MAX_BYTES or _digest(body) != document['sha256']:
                raise WebError('Cached original document changed; extraction refused.', code='source_integrity_error')
            if document['original_file'].endswith('.html'):
                parser = _HTMLPage()
                parser.feed(body.decode('utf-8', errors='replace'))
                text, headings, links, references, coverage = parser.extract(document['source_url'])
                source_kind = 'html'
            else:
                _, lines = self.literature._document(document['document_id'])
                text = '\n'.join(lines)
                headings, links, references = [], [], []
                source_kind = 'pdf' if document['original_file'].endswith('.pdf') else 'text'
                coverage = {'kind': 'not_detected', 'complete': False,
                            'warning': 'Structured bibliography extraction is available for HTML pages; read the cached text for this source.'}
            metadata = {'page_id': page_id, 'document_id': document['document_id'],
                'requested_url': document.get('requested_url', url), 'source_url': document['source_url'],
                'final_url': document['source_url'], 'retrieved_at': document['retrieved_at'],
                'source_sha256': document['sha256'], 'text_sha256': _digest(text), 'extractor': VERSION,
                'source_kind': source_kind, 'title': document.get('title', ''), 'total_lines': len(text.splitlines()),
                'headings': headings, 'links': links, 'references': references, 'reference_coverage': coverage}
            metadata['structure_sha256'] = _digest(json.dumps(self._structure(metadata), ensure_ascii=False, sort_keys=True))
            serialized = json.dumps(metadata, ensure_ascii=False)
            if len(serialized.encode()) > META_BYTES:
                raise WebError('Extracted page structure exceeds the cache bound.', code='unsupported_document')
            self.literature._write(('web-pages', page_id + '.txt'), text)
            self.literature._write(('web-pages', page_id + '.json'), serialized)
        result = {key: metadata[key] for key in ('page_id', 'document_id', 'source_url', 'final_url',
            'retrieved_at', 'source_sha256', 'text_sha256', 'source_kind', 'title', 'total_lines')}
        result.update(requested_url=url, cached=bool(cached), headings=copy.deepcopy(metadata['headings'][:20]),
                      headings_total=len(metadata['headings']), links=copy.deepcopy(metadata['links'][:12]),
                      links_total=len(metadata['links']), reference_count=len(metadata['references']),
                      reference_coverage=copy.deepcopy(metadata['reference_coverage']),
                      next_action='Read page lines or paginate read_references. A URL was read, not its linked sources.')
        # The full structural lists remain cached; index previews explicitly
        # advertise their limits rather than changing stored bibliography data.
        def counts():
            result['links_returned'] = len(result['links'])
            result['headings_returned'] = len(result['headings'])
            result['index_complete'] = result['links_returned'] == result['links_total'] and result['headings_returned'] == result['headings_total']
        counts()
        while len(json.dumps(result, ensure_ascii=False).encode()) > self._cap() and (result['links'] or result['headings']):
            if result['links']:
                result['links'].pop()
            else:
                result['headings'].pop()
            counts()
        return result

    def _page(self, page_id):
        if not isinstance(page_id, str) or not PAGE_ID.fullmatch(page_id):
            raise WebError('Unknown page identifier; use open_url first.', code='unknown_page')
        raw = self.literature._read(('web-pages', page_id + '.json'), META_BYTES)
        text = self.literature._read(('web-pages', page_id + '.txt'))
        if raw is None or text is None:
            raise WebError('This page is not cached; use open_url first.', code='unknown_page')
        try:
            metadata = json.loads(raw)
            valid = (metadata['page_id'] == page_id and metadata['text_sha256'] == _digest(text)
                     and metadata['structure_sha256'] == _digest(json.dumps(self._structure(metadata), ensure_ascii=False, sort_keys=True)))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise WebError('Cached page evidence changed; hash verification failed.', code='source_integrity_error')
        return metadata, text.splitlines()

    @staticmethod
    def _source(metadata):
        return {key: metadata[key] for key in ('page_id', 'source_url', 'retrieved_at', 'text_sha256', 'source_kind')}

    def read_page(self, page_id, start_line=1, end_line=80):
        if type(start_line) is not int or type(end_line) is not int or not 1 <= start_line <= end_line:
            raise WebError('Line bounds must be positive integers with start <= end.', code='invalid_arguments')
        metadata, lines = self._page(page_id)
        if start_line > len(lines):
            raise WebError('Requested starting line is beyond this page.', code='invalid_arguments')
        requested_end = min(end_line, len(lines))
        end = min(requested_end, start_line + 79)
        result = None
        while end >= start_line:
            result = {**self._source(metadata), 'total_lines': len(lines), 'start_line': start_line, 'end_line': end,
                'citation': f'{page_id}:L{start_line}-L{end}',
                'passage': '\n'.join(f'{index + 1}: {lines[index]}' for index in range(start_line - 1, end)),
                'links': [copy.deepcopy(link) for link in metadata['links']
                          if link['start_line'] <= end and link['end_line'] >= start_line],
                'headings': [copy.deepcopy(h) for h in metadata['headings'] if start_line <= h['start_line'] <= end],
                'truncated': end < requested_end, 'next_line': end + 1 if end < len(lines) else None,
                'evidence_kind': 'untrusted_cached_web_extraction'}
            if len(json.dumps(result, ensure_ascii=False).encode()) <= self._cap():
                return result
            end -= 1
        raise WebError('Even one requested line and its links exceed the remaining output budget; '
                       'use search_page or increase the evidence budget.', code='web_budget_exhausted')

    def search_page(self, page_id, query):
        query = self._query(query, 1)
        metadata, lines = self._page(page_id)
        hits = []
        links_by_line = {}
        for link in metadata['links']:
            for line in range(max(1, link['start_line']), min(link['end_line'], len(lines)) + 1):
                links_by_line.setdefault(line, []).append(link)
        total = 0
        for index, line in enumerate(lines, 1):
            pos = line.casefold().find(query.casefold())
            if pos < 0:
                continue
            total += 1
            if len(hits) >= 8:
                continue
            start = max(0, pos - 160)
            hits.append({'line': index, 'citation': f'{page_id}:L{index}', 'passage': line[start:start + 500],
                         'partial_line': start > 0 or len(line) > 500,
                         'links': copy.deepcopy(links_by_line.get(index, []))})
        result = {**self._source(metadata), 'query': query, 'matches': hits[:8], 'total_matches': total,
                  'truncated': total > 8}
        while result['matches'] and len(json.dumps(result, ensure_ascii=False).encode()) > self._cap():
            result['matches'].pop()
            result['truncated'] = True
        if total and not result['matches']:
            raise WebError('A matching page excerpt cannot fit the remaining output budget; '
                           'read smaller source passages instead.', code='web_budget_exhausted')
        return result

    def read_references(self, page_id, start=1, limit=8):
        if type(start) is not int or start < 1 or type(limit) is not int or not 1 <= limit <= 20:
            raise WebError('Reference start must be positive and limit must be 1 to 20.', code='invalid_arguments')
        metadata, _ = self._page(page_id)
        references, coverage = metadata['references'], metadata['reference_coverage']
        total = len(references)
        if start > total + 1:
            raise WebError('Reference start is beyond this bibliography.', code='invalid_arguments')
        selected = copy.deepcopy(references[start - 1:start - 1 + limit])
        for reference in selected:
            reference['citation'] = f'{page_id}:L{reference["start_line"]}-L{reference["end_line"]}'
        result = {**self._source(metadata), 'references': selected, 'total': total, 'start': start,
                  'coverage': coverage['kind'], 'warning': coverage['warning']}
        while True:
            returned = len(selected)
            end = start + returned - 1
            result.update(end=end, returned=returned, next_start=end + 1 if end < total else None,
                          pagination_complete=end >= total,
                          complete=end >= total and bool(coverage['complete']), coverage_complete=bool(coverage['complete']),
                          truncated=returned < min(limit, max(0, total - start + 1)))
            if len(json.dumps(result, ensure_ascii=False).encode()) <= self._cap():
                if returned == 0 and start <= total:
                    raise WebError('A bibliography entry cannot fit the remaining output budget; its full source '
                                   'remains cached. Read source passages instead of claiming complete coverage.',
                                   code='web_budget_exhausted', details={'page_id': page_id, 'next_start': start, 'total': total})
                return result
            if selected:
                selected.pop()
                continue
            raise WebError('A bibliography entry cannot fit the remaining output budget; its full source '
                           'remains cached. Read source passages instead of claiming complete coverage.',
                           code='web_budget_exhausted', details={'page_id': page_id, 'next_start': start, 'total': total})

    def search_web(self, query, limit=5):
        # The existing provider cache is available offline, and lack of a key
        # never affects open_url for a supplied source URL.
        query = self._query(query, limit)
        return self.literature.search_web(query, limit)

    def _query(self, query, limit):
        try:
            return self.literature._query(query, limit)
        except LiteratureError as exc:
            raise WebError(str(exc), code='invalid_arguments', retryable=True) from None
