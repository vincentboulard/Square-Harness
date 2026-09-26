"""Local web server for the visual interface (standard library only).

The security model matches the terminal interface:
- loopback only unless --gui-host explicitly selects another address;
- every /api request needs the workspace's access token (cookie or bearer);
- loopback mode also checks the Host header, against DNS rebinding;
- state-changing requests must be same-origin JSON and get no CORS answer;
- a strict Content-Security-Policy stops rendered model or source text from
  loading anything outside this server, so a hostile document cannot leak
  data through an image or script address;
- network, Python and literature permissions stay fixed at launch.
"""
import hmac
import http.cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import sys
import threading
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser

from ..agent import AgentError
from ..ledger import _directory
from . import store
from .hub import Busy, Hub
from .watch import Watcher

VERSION = '0.4.0'
COOKIE = 'square_token'  # suffixed with the port: cookies ignore ports, workspaces must not collide
UUID = r'(?P<id>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})'
STATIC = Path(__file__).with_name('static')
MAX_BODY = 1_000_000
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "font-src 'self' data:; connect-src 'self'; manifest-src 'self'; worker-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.json': 'application/json',
         '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml', '.png': 'image/png',
         '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
         '.txt': 'text/plain; charset=utf-8', '.map': 'application/json'}
FALLBACK = b'''<!doctype html><meta charset="utf-8"><title>Square Harness</title>
<body style="font:16px system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>Interface assets are missing</h1><p>This checkout has no built interface. Build it with
<code>npm install &amp;&amp; npm run build</code> inside <code>frontend/</code>, then reload.</p>'''

ROUTES = []


def route(method, pattern, auth=True, raw=False):
    def register(function):
        function.raw_body = raw
        ROUTES.append((method, re.compile(pattern + r'\Z'), function, auth))
        return function
    return register


class Streamed:
    """Marker: the handler already wrote a streaming response."""


class Unauthorized(Exception):
    """A login attempt used the wrong access token."""


def _int(body, key, low=None, high=None):
    value = body.get(key)
    if value is None:
        return None
    if type(value) is not int or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{key} must be an integer' + (f' from {low} to {high}' if low is not None and high is not None else ''))
    return value


def _seconds(body, key):
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be a positive number of seconds')
    return float(value)


def _flag(body, key):
    value = body.get(key)
    if value is not None and type(value) is not bool:
        raise ValueError(f'{key} must be true or false')
    return value


def _files(body, key='source_files'):
    value = body.get(key) or []
    if not isinstance(value, list) or len(value) > 20 or any(not isinstance(v, str) or not v or len(v) > 1000 for v in value):
        raise ValueError('Pinned files must be a list of at most 20 workspace-relative paths')
    return value


def _text(body, key, limit):
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Provide {key}')
    if len(value) > limit:
        raise ValueError(f'{key} is limited to {limit} characters')
    return value


# -- routes -------------------------------------------------------------------

@route('GET', '/api/status')
def status(h, body):
    hub, args = h.server.hub, h.server.hub.args
    out = {'version': VERSION, 'instance': hub.bus.instance, 'seq': hub.bus.seq,
           'workspace': str(hub.root), 'model': args.model, 'host': args.host,
           'online': bool(args.online) and not hub.online_locked, 'online_locked': hub.online_locked,
           'allow_python': bool(args.allow_python),
           'proof_literature': bool(args.proof_literature), 'lan': not h.server.loopback,
           'defaults': {'ctx': args.ctx, 'predict': args.predict, 'think': not args.no_think,
                        'proof_rounds': args.proof_rounds, 'proof_tokens': args.proof_tokens,
                        'proof_seconds': args.proof_seconds, 'proof_max_predict': args.proof_max_predict,
                        'research_rounds': args.research_rounds, 'research_tokens': args.research_tokens,
                        'research_input_tokens': args.research_input_tokens,
                        'research_seconds': args.research_seconds, 'research_requests': args.research_requests,
                        'research_chars': args.research_chars},
           'ollama': hub.models(), 'root': str(hub.base),
           'workspace_path': '' if hub.root == hub.base else hub.root.relative_to(hub.base).as_posix(),
           'read': store.load_settings(hub.root)['read'], 'latex': shutil.which('latexmk') is not None,
           'recent': store.recent_folders(hub.base)}
    if not h.server.loopback:
        # Only an authenticated device can read the pairing addresses.
        out['pair_urls'] = [url + '/?' + urlencode({'token': h.server.token}) for url in h.server.urls]
    return out


@route('POST', '/api/login', auth=False)
def login(h, body):
    token = body.get('token')
    if not isinstance(token, str) or not hmac.compare_digest(token.strip().encode(), h.server.token.encode()):
        raise Unauthorized('That access token is not valid for this workspace')
    h.cookie = h.server.token
    return {'ok': True}


@route('GET', '/api/events')
def events(h, body):
    bus = h.server.hub.bus
    after, resync = bus.seq, False
    last = h.headers.get('Last-Event-ID', '')
    if ':' in last:
        instance, _, number = last.partition(':')
        if instance == bus.instance and number.isdigit():
            after = int(number)
        else:
            resync = True
    h.send_response(200)
    h.send_header('Content-Type', 'text/event-stream; charset=utf-8')
    h.send_header('Cache-Control', 'no-store')
    h.send_header('X-Accel-Buffering', 'no')
    h.security_headers()
    h.end_headers()
    h.close_connection = True

    def send(event):
        data = json.dumps(event, ensure_ascii=False, separators=(',', ':'))
        h.wfile.write(f'id: {bus.instance}:{event.get("seq", after)}\ndata: {data}\n\n'.encode())

    h.wfile.write(b'retry: 2000\n\n')
    send({'type': 'hello', 'instance': bus.instance, 'seq': after, 'resync': resync})
    h.wfile.flush()
    while not h.server.stopping.is_set():
        batch, missed = bus.wait(after, 15)
        if missed:
            after = bus.seq
            send({'type': 'hello', 'instance': bus.instance, 'seq': after, 'resync': True})
        elif not batch:
            h.wfile.write(b': ping\n\n')
        for event in batch:
            send(event)
            after = event['seq']
        h.wfile.flush()
    return Streamed


@route('GET', '/api/task')
def task(h, body):
    return h.server.hub.snapshot()


@route('POST', '/api/task/pause')
def pause(h, body):
    return {'task': h.server.hub.pause().summary()}


@route('POST', '/api/approvals/(?P<approval>[0-9a-f]{32})')
def approval(h, body, approval):
    decision = _flag(body, 'approve')
    if decision is None:
        raise ValueError('approve must be true or false')
    h.server.hub.decide(approval, decision)
    return {'ok': True}


@route('GET', '/api/files')
def files(h, body):
    return {'files': store.list_files(h.server.hub.root)}


@route('GET', '/api/proofs')
def proofs(h, body):
    return {'jobs': store.list_proofs(h.server.hub.root, h.active('proof'))}


@route('POST', '/api/proofs')
def proof_start(h, body):
    task = h.server.hub.start_proof(
        _text(body, 'goal', 60000), source_files=_files(body),
        rounds=_int(body, 'rounds', 1, 100), tokens=_int(body, 'tokens', 512),
        seconds=_seconds(body, 'seconds'), max_predict=_int(body, 'max_predict', 128),
        think=_flag(body, 'think'), ctx=_int(body, 'ctx', 2048), literature=bool(_flag(body, 'literature')),
        online=_flag(body, 'online'))
    return {'task': task.summary(), 'id': task.target}


@route('GET', '/api/proofs/' + UUID)
def proof(h, body, id):
    return store.proof_detail(h.server.hub.root, id, h.active('proof'))


@route('POST', '/api/proofs/' + UUID + '/resume')
def proof_resume(h, body, id):
    return {'task': h.server.hub.resume_proof(id).summary(), 'id': id}


@route('GET', '/api/proofs/' + UUID + '/report')
def proof_report(h, body, id):
    return store.proof_report(h.server.hub.root, id)


@route('GET', '/api/proofs/' + UUID + '/sources')
def proof_sources(h, body, id):
    return {'sources': store.proof_sources(h.server.hub.root, id)}


@route('GET', '/api/proofs/' + UUID + '/artifacts/(?P<name>[A-Za-z0-9_.-]+)')
def proof_artifact(h, body, id, name):
    return store.proof_artifact(h.server.hub.root, id, name)


@route('GET', '/api/proofs/' + UUID + '/stream/(?P<name>[A-Za-z0-9_.-]+)')
def proof_stream(h, body, id, name):
    return store.proof_stream(h.server.hub.root, id, name, h.offset())


@route('GET', '/api/research')
def researches(h, body):
    return {'jobs': store.list_research(h.server.hub.root, h.active('research'))}


@route('POST', '/api/research')
def research_start(h, body):
    task = h.server.hub.start_research(
        body.get('kind'), _text(body, 'goal', 20000), source_files=_files(body),
        rounds=_int(body, 'rounds', 1, 100), tokens=_int(body, 'tokens', 1024),
        input_tokens=_int(body, 'input_tokens', 2048), seconds=_seconds(body, 'seconds'),
        requests=_int(body, 'requests', 0, 100), chars=_int(body, 'chars', 1000, 1_000_000),
        online=_flag(body, 'online'))
    return {'task': task.summary(), 'id': task.target}


@route('GET', '/api/research/' + UUID)
def research(h, body, id):
    return store.research_detail(h.server.hub.root, id, h.active('research'))


@route('POST', '/api/research/' + UUID + '/resume')
def research_resume(h, body, id):
    return {'task': h.server.hub.resume_research(id).summary(), 'id': id}


@route('GET', '/api/research/' + UUID + '/sources')
def research_sources(h, body, id):
    return {'sources': store.research_sources(h.server.hub.root, id)}


@route('GET', '/api/research/' + UUID + '/artifacts/(?P<name>[A-Za-z0-9_.-]+)')
def research_artifact(h, body, id, name):
    return store.research_artifact(h.server.hub.root, id, name)


@route('GET', '/api/research/' + UUID + '/stream/(?P<name>[A-Za-z0-9_.-]+)')
def research_stream(h, body, id, name):
    return store.research_artifact(h.server.hub.root, id, name, h.offset())


@route('GET', '/api/chats')
def chats(h, body):
    return {'chats': store.list_chats(h.server.hub.root)}


@route('POST', '/api/chats')
def chat_create(h, body):
    hub = h.server.hub
    think, online = _flag(body, 'think'), _flag(body, 'online')
    chat = store.create_chat(hub.root, body.get('mode'), (not hub.args.no_think) if think is None else think,
                             hub.args.model, hub.online(online))
    hub.bus.publish('chat_saved', chat=chat['id'])
    return _chat_view(chat)


@route('GET', '/api/chats/' + UUID)
def chat(h, body, id):
    return _chat_view(store.load_chat(h.server.hub.root, id))


@route('POST', '/api/chats/' + UUID + '/settings')
def chat_settings(h, body, id):
    hub = h.server.hub
    think, online = _flag(body, 'think'), _flag(body, 'online')
    if think is None and online is None:
        raise ValueError('Give think or online as true or false')
    if online:
        hub.online(True)  # refuses when launched with --offline
    return _chat_view(store.update_chat_settings(hub.root, id, think=think, online=online))


@route('POST', '/api/chats/' + UUID + '/messages')
def chat_message(h, body, id):
    return {'task': h.server.hub.chat(id, _text(body, 'content', 60000), files=_files(body, 'files')).summary()}


@route('POST', '/api/chats/' + UUID + '/route')
def chat_route(h, body, id):
    h.server.hub.route(id, _text(body, 'content', 60000), _files(body, 'files'))
    return {'ok': True}


@route('POST', '/api/chats/' + UUID + r'/routes/(?P<index>\d{1,6})/start')
def route_start(h, body, id, index):
    raw = body.get('limits') or {}
    if not isinstance(raw, dict):
        raise ValueError('limits must be an object')
    limits = {key: value for key, value in (
        ('rounds', _int(raw, 'rounds', 1, 100)), ('tokens', _int(raw, 'tokens', 512)),
        ('input_tokens', _int(raw, 'input_tokens', 2048)), ('seconds', _seconds(raw, 'seconds')),
        ('requests', _int(raw, 'requests', 0, 100)), ('chars', _int(raw, 'chars', 1000, 1_000_000))) if value is not None}
    task = h.server.hub.start_route(id, int(index), body.get('mode'), _text(body, 'request', 60000),
                                    _files(body, 'files'), limits)
    return {'task': task.summary()}


@route('POST', '/api/chats/' + UUID + r'/routes/(?P<index>\d{1,6})/dismiss')
def route_dismiss(h, body, id, index):
    return {'item': h.server.hub.dismiss_route(id, int(index))}


@route('POST', '/api/queue/(?P<task>[0-9a-f]{12})/cancel')
def queue_cancel(h, body, task):
    return {'task': h.server.hub.cancel_queued(task).summary()}


@route('GET', '/api/folders')
def folders(h, body):
    hub = h.server.hub
    return {**store.list_folders(hub.base, (h.query.get('path') or [''])[0]),
            'current': hub.root.relative_to(hub.base).as_posix() if hub.root != hub.base else '',
            'recent': store.recent_folders(hub.base)}


@route('POST', '/api/folders')
def folder_create(h, body):
    path = body.get('path') or ''
    return {'path': store.create_folder(h.server.hub.base, path, body.get('name'))}


@route('POST', '/api/workspace')
def workspace_open(h, body):
    path = body.get('path')
    if not isinstance(path, str):
        raise ValueError('Give the folder path relative to the interface root')
    h.server.switch(path)
    return {'workspace': str(h.server.hub.root)}


@route('GET', '/api/settings')
def settings(h, body):
    return store.load_settings(h.server.hub.root)


@route('POST', '/api/settings')
def settings_save(h, body):
    return store.save_settings(h.server.hub.root, body.get('read'))


@route('POST', '/api/files/upload', raw=True)
def upload(h, body):
    name = (h.query.get('name') or [''])[0]
    saved = store.save_upload(h.server.hub.root, name, body)
    h.server.hub.bus.publish('files', path=saved)
    return {'path': saved}


@route('GET', '/api/templates')
def templates(h, body):
    return {'templates': store.list_templates(h.server.hub.base)}


@route('POST', '/api/writeup')
def writeup_start(h, body):
    template = body.get('template')
    save_as = body.get('save_template')
    notes = body.get('notes') or ''
    if not isinstance(notes, str) or not isinstance(body.get('output') or '', str):
        raise ValueError('notes and output must be text')
    task = h.server.hub.start_writeup(
        _text(body, 'goal', 20000), source_files=_files(body), template_files=_files(body, 'template_files'),
        template=template if isinstance(template, str) and template else None,
        save_template=save_as if isinstance(save_as, str) and save_as.strip() else None,
        notes=notes, output=body.get('output') or '',
        rounds=_int(body, 'rounds', 1, 100), tokens=_int(body, 'tokens', 1024),
        input_tokens=_int(body, 'input_tokens', 2048), seconds=_seconds(body, 'seconds'))
    return {'task': task.summary(), 'id': task.target}


@route('GET', '/api/research/' + UUID + '/pdf')
def research_pdf(h, body, id):
    data = store.research_pdf(h.server.hub.root, id)
    # A top-level PDF view: the browser's viewer must not be blocked by the page CSP.
    h.send_bytes(200, data, 'application/pdf', [('Content-Disposition', 'inline; filename="writeup.pdf"'),
                                                ('Cache-Control', 'no-store')], csp=False)
    return Streamed


@route('POST', '/api/chats/' + UUID + '/review')
def chat_review(h, body, id):
    return {'task': h.server.hub.review(id).summary()}


def _chat_view(chat):
    return {'id': chat['id'], 'mode': chat['mode'], 'title': chat['title'], 'settings': chat['settings'],
            'created_at': chat['created_at'], 'updated_at': chat['updated_at'],
            'transcript': chat['transcript'], 'context_messages': len(chat['history'])}


# -- HTTP plumbing ------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'SquareHarness/' + VERSION
    sys_version = ''

    def log_message(self, format, *args):
        pass  # request logs would print job IDs and paths on every poll

    def do_GET(self):
        self._handle('GET')

    def do_POST(self):
        self._handle('POST')

    def do_PUT(self):
        self._handle('PUT')

    def do_DELETE(self):
        self._handle('DELETE')

    def _handle(self, method):
        self.cookie = None
        self.body_read = method == 'GET'
        try:
            url = urlsplit(self.path)
            self.query = parse_qs(url.query)
            if not self._host_allowed():
                return self.error(403, 'Unexpected Host header; open the address printed in the terminal')
            if url.path.startswith('/api/'):
                return self._api(method, url.path)
            if method != 'GET':
                return self.error(405, 'Method not allowed')
            return self._static(url.path)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def active(self, kind):
        task = self.server.hub.active()
        return task.target if task and task.kind == kind else None

    def offset(self):
        value = (self.query.get('from') or ['0'])[0]
        if not value.isdigit():
            raise ValueError('from must be a nonnegative byte offset')
        return int(value)

    def _host_allowed(self):
        if not self.server.loopback:
            return True  # LAN mode: phones use an IP address; the token is required
        host = self.headers.get('Host', '').lower()
        name = host.split(']')[0] + ']' if host.startswith('[') else host.rsplit(':', 1)[0]
        return name in ('localhost', '127.0.0.1', '[::1]')

    def _authorized(self):
        token = self.server.token.encode()
        header = self.headers.get('Authorization', '')
        if header.startswith('Bearer ') and hmac.compare_digest(header[7:].strip().encode(), token):
            return True
        try:
            morsel = http.cookies.SimpleCookie(self.headers.get('Cookie', '')).get(self.server.cookie_name)
        except http.cookies.CookieError:
            return False
        return morsel is not None and hmac.compare_digest(morsel.value.encode(), token)

    def _same_origin(self):
        site = self.headers.get('Sec-Fetch-Site')
        if site is not None and site not in ('same-origin', 'none'):
            return False
        origin = self.headers.get('Origin')
        return origin is None or urlsplit(origin).netloc.lower() == self.headers.get('Host', '').lower()

    def _body(self):
        length = self.headers.get('Content-Length', '0')
        if not length.isdigit() or int(length) > MAX_BODY:
            raise ValueError('Request body is missing or too large')
        self.body_read = int(length) == 0
        if int(length) == 0:
            return {}
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            raise TypeError('Requests must be JSON')
        data = self.rfile.read(int(length))
        self.body_read = True
        value = json.loads(data.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('Request body must be a JSON object')
        return value

    def _raw_body(self):
        length = self.headers.get('Content-Length', '')
        if not length.isdigit() or int(length) > store.UPLOAD_BYTES:
            raise ValueError('Files are limited to 20 MB')
        data = self.rfile.read(int(length))
        self.body_read = True
        return data

    def _api(self, method, path):
        allowed = [(fn, match, auth) for m, pattern, fn, auth in ROUTES
                   if m == method and (match := pattern.match(path))]
        if not allowed:
            known = any(pattern.match(path) for _, pattern, _, _ in ROUTES)
            return self.error(405 if known else 404, 'Method not allowed' if known else 'Unknown endpoint')
        function, match, auth = allowed[0]
        if auth and not self._authorized():
            return self.error(401, 'Open the address printed in the terminal, or enter its access token')
        body = {}
        if method != 'GET':
            if not self._same_origin():
                return self.error(403, 'Cross-origin request refused')
            try:
                body = self._raw_body() if getattr(function, 'raw_body', False) else self._body()
            except TypeError as exc:
                return self.error(415, str(exc))
            except (ValueError, UnicodeError, RecursionError) as exc:
                return self.error(400, 'Invalid JSON body' if isinstance(exc, (json.JSONDecodeError, RecursionError)) else str(exc))
        try:
            result = function(self, body, **match.groupdict())
        except Busy as exc:
            return self.error(409, str(exc))
        except store.NotFound as exc:
            return self.error(404, str(exc))
        except Unauthorized as exc:
            return self.error(401, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (ValueError, AgentError, OSError) as exc:
            return self.error(400, str(exc))
        except Exception as exc:  # report, never crash the handler thread silently
            print(f'Interface error on {method} {path}: {type(exc).__name__}: {exc}', file=sys.stderr)
            return self.error(500, 'Internal error; see the terminal for details')
        if result is not Streamed:
            self.json(200, result)

    def _static(self, path):
        token = (self.query.get('token') or [''])[0]
        if token:
            if hmac.compare_digest(token.encode(), self.server.token.encode()):
                self.cookie = self.server.token
            # Never leave the token in the address bar, history or screenshots.
            return self.send_bytes(303, b'', 'text/plain', [('Location', '/')])
        relative = (path or '/').lstrip('/') or 'index.html'
        parts = relative.split('/')
        if any(not part or part.startswith('.') or part == '..' for part in parts):
            return self.error(404, 'Not found', html=True)
        base = STATIC.resolve()
        target = (base / relative).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            if relative == 'index.html':
                return self.send_bytes(200, FALLBACK, TYPES['.html'], [('Cache-Control', 'no-store')])
            return self.error(404, 'Not found', html=True)
        cache = 'public, max-age=31536000, immutable' if parts[0] == 'assets' else 'no-cache'
        self.send_bytes(200, target.read_bytes(), TYPES.get(target.suffix.lower(), 'application/octet-stream'),
                        [('Cache-Control', cache)])

    def security_headers(self, csp=True):
        if csp:
            self.send_header('Content-Security-Policy', CSP)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
        if self.cookie:
            # SameSite=Strict and HttpOnly: other sites can neither send nor read it.
            self.send_header('Set-Cookie', f'{self.server.cookie_name}={self.cookie}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000')

    def send_bytes(self, status, body, content_type, headers=(), csp=True):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.security_headers(csp)
        for key, value in headers:
            self.send_header(key, value)
        if not getattr(self, 'body_read', True):
            # Unread body bytes must never be parsed as the next keep-alive request.
            self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_bytes(status, body, 'application/json; charset=utf-8', [('Cache-Control', 'no-store')])

    def error(self, status, message, html=False):
        if html:
            return self.send_bytes(status, message.encode(), 'text/plain; charset=utf-8')
        self.json(status, {'error': message})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, hub, token, loopback):
        if ':' in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, Handler)
        self.hub, self.token, self.loopback = hub, token, loopback
        self.cookie_name = f'{COOKIE}_{self.server_address[1]}'
        self.stopping = threading.Event()
        self.urls = []

    def shutdown(self):
        self.stopping.set()
        super().shutdown()

    def switch(self, relative):
        """Open another folder below the root and follow its saved jobs instead."""
        target = self.hub.set_workspace(relative)
        old = getattr(self, 'watcher', None)
        if old is not None:
            old.stop()
            self.watcher = Watcher(target, self.hub.bus, old.interval)
            self.watcher.start()
        self.hub.bus.publish('workspace', path=target.relative_to(self.hub.base).as_posix())


# -- launch ------------------------------------------------------------------

def workspace_token(root):
    """A per-workspace access token kept with the saved jobs it protects (mode 0600)."""
    base = Path(root) / '.mathagent'
    _directory(base, create=True)
    path = base / 'gui-token'
    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) and not info.st_mode & 0o077:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, encoding='ascii') as stream:
                value = stream.read(200).strip()
            if re.fullmatch(r'[A-Za-z0-9_-]{32,128}', value):
                return value
        path.unlink()  # unreadable, malformed or readable by others: replace it
    except FileNotFoundError:
        pass
    value = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='ascii') as stream:
        stream.write(value + '\n')
    return value


def _loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _lan_addresses():
    found = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))  # TEST-NET-1: a UDP connect sends no packet
            found.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in found if not ipaddress.ip_address(a).is_loopback)


def bind(host, port, hub, token):
    error = None
    for candidate in range(port, port + 10) if port else [0]:
        try:
            return Server((host, candidate), hub, token, _loopback(host))
        except OSError as exc:
            error = exc
    raise OSError(f'Cannot listen on {host} ports {port}–{port + 9}: {error}. Choose another --gui-port.')


def serve(args):
    root = Path(args.workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Workspace must be a directory')
    if not _loopback(args.gui_host):
        try:
            ipaddress.ip_address(args.gui_host)
        except ValueError:
            raise ValueError('--gui-host must be an IP address such as 127.0.0.1 or 0.0.0.0') from None
    base = Path(getattr(args, 'gui_root', None) or root).expanduser().resolve(strict=True)
    if not base.is_dir() or not root.is_relative_to(base):
        raise ValueError('--gui-root must be a folder that contains the --workspace folder')
    args.workspace, args.gui_root = root, base
    token = workspace_token(base)
    hub = Hub(args)
    server = bind(args.gui_host, args.gui_port, hub, token)
    port = server.server_address[1]
    local = f'http://127.0.0.1:{port}' if server.loopback or args.gui_host in ('0.0.0.0', '::') else f'http://{args.gui_host}:{port}'
    if not server.loopback:
        hosts = _lan_addresses() if args.gui_host in ('0.0.0.0', '::') else [args.gui_host]
        server.urls = [f'http://{h}:{port}' for h in hosts]
    watcher = Watcher(root, hub.bus)
    watcher.start()
    server.watcher = watcher
    query = '/?' + urlencode({'token': token})
    lines = [f'╭─ SQUARE HARNESS · v{VERSION} · visual interface',
             f'│ {args.model} · {args.host}',
             f'│ Workspace: {root}' + (f' (folders below {base} can be opened)' if base != root else ''),
             f'│ Research: {"online" if args.online else "offline (local/cache only)"} · proof literature: '
             f'{"enabled for new jobs" if args.proof_literature else "disabled"}',
             f'│ Open: {local}{query}']
    for url in server.urls:
        lines.append(f'│ Phone / other device: {url}{query}')
    if server.urls:
        lines.append('│ Network mode: traffic is plain HTTP on your local network. Use it only on a trusted')
        lines.append('│ network or through a VPN/SSH tunnel; anyone with the token can control this harness.')
    lines.append('╰─ Ctrl+C stops the interface; running work pauses at its saved checkpoint')
    print('\n'.join(lines), flush=True)
    if args.allow_python:
        print('Python enabled: each snippet requires approval in the interface and runs with your account permissions.')
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, [local + query]).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print('\nStopping. Running work is pausing at its checkpoint…', flush=True)
    finally:
        server.stopping.set()
        hub.shutdown()
        server.watcher.stop()
        server.server_close()
    return 0
