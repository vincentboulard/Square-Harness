"""Run model-written numerical experiments in a separate, resource-limited process.

Isolation is layered and reported, never assumed:

* ``bwrap`` (Linux, bubblewrap installed): no network, read-only system, the home
  folder hidden except the Python installation, writes only in the run folder.
* ``seatbelt`` (macOS ``sandbox-exec``): no network, writes only in the run folder.
* ``netns`` (Linux ``unshare -rn``): no network; the file system is NOT protected.
* ``none``: resource limits only.

Every method also applies CPU, memory, file-size and wall-clock limits, a clean
environment and a Python-level network guard. The method actually used is saved
with each run so a reader can judge what the code could have touched.
"""
from functools import lru_cache
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

PERMISSIONS = ('off', 'ask', 'auto')
ISOLATED = ('bwrap', 'seatbelt', 'netns')
MAX_CODE = 60_000
MAX_STREAM = 20_000
MAX_RESULTS = 1_000_000
MAX_FIGURES = 12
MAX_FIGURE_BYTES = 5_000_000
FIGURE_SUFFIXES = ('.png', '.svg')
PACKAGES = ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib', 'python-flint')
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)

PRELUDE = r'''# Square Harness experiment launcher (fixed code, not model-written).
import os, re, runpy, sys, traceback
sys.path.insert(0, os.getcwd())
sys.path.append(%(package_root)r)
import socket
def _no_network(*args, **kwargs):
    raise OSError('Network access is disabled in Square Harness experiments')
socket.socket.connect = socket.socket.connect_ex = _no_network
socket.create_connection = socket.getaddrinfo = _no_network
_code = 0
try:
    runpy.run_path(%(script)r, run_name='__main__')
except SystemExit as exc:
    _code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    traceback.print_exc()
    _code = 1
finally:
    plt = sys.modules.get('matplotlib.pyplot')
    if plt is not None:
        try:
            os.makedirs('figures', exist_ok=True)
            for number in plt.get_fignums()[:12]:
                figure = plt.figure(number)
                name = re.sub(r'[^A-Za-z0-9_-]+', '-', figure.get_label() or '').strip('-')[:60] or 'figure-%%d' %% number
                if not os.path.exists(os.path.join('figures', name + '.png')):
                    figure.savefig(os.path.join('figures', name + '.png'), dpi=120, bbox_inches='tight')
                    figure.savefig(os.path.join('figures', name + '.svg'), bbox_inches='tight')
        except Exception:
            traceback.print_exc()
sys.stdout.flush()
os._exit(_code)
'''

SEATBELT = '''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "%s") (literal "/dev/null") (literal "/dev/zero"))
'''


def _probe(command):
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _visible_paths():
    """Directories the Python interpreter needs inside a bubblewrap sandbox."""
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix, PACKAGE_ROOT,
             str(Path(sys.executable).resolve().parent)}
    paths.update(p for p in sys.path if p)
    return sorted(p for p in paths if Path(p).exists())


def _bwrap(workdir):
    command = ['bwrap', '--die-with-parent', '--unshare-net', '--unshare-pid', '--new-session',
               '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp']
    home = Path.home()
    if home.exists() and str(home) != '/':
        command += ['--tmpfs', str(home)]
        for path in _visible_paths():
            if Path(path).resolve().is_relative_to(home):
                command += ['--ro-bind', path, path]
    return command + ['--bind', str(workdir), str(workdir), '--chdir', str(workdir), '--']


@lru_cache(maxsize=1)
def available_isolation():
    """The strongest isolation that actually works on this computer."""
    python = [sys.executable, '-I', '-c', 'pass']
    if sys.platform.startswith('linux'):
        if shutil.which('bwrap') and _probe(_bwrap(Path('/tmp')) + python):
            return 'bwrap'
        if shutil.which('unshare') and _probe(['unshare', '-rn'] + python):
            return 'netns'
    elif sys.platform == 'darwin' and shutil.which('sandbox-exec'):
        if _probe(['sandbox-exec', '-p', SEATBELT % '/tmp'] + python):
            return 'seatbelt'
    return 'none'


def environment():
    """Interpreter and numerical-library versions, saved with every run."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'isolation': available_isolation(), 'packages': versions}


def permitted(mode, approve, preview):
    """Return (allowed, how). 'auto' runs only under real isolation, else asks."""
    if mode not in PERMISSIONS:
        raise ValueError('Experiment permission must be off, ask or auto')
    if mode == 'off':
        return False, 'Experiments are switched off; the code was saved but not run.'
    isolation = available_isolation()
    if mode == 'auto' and isolation in ISOLATED:
        return True, f'Run automatically under {isolation} isolation.'
    note = '' if mode == 'ask' else f'Automatic runs need network isolation, which is unavailable ({isolation}); asking instead.\n'
    header = (f'RUN NUMERICAL EXPERIMENT (isolation: {isolation}; '
              + ('no network' if isolation in ISOLATED else 'NOT sandboxed: runs with your account permissions') + ')\n')
    if approve(header + note + preview):
        return True, 'Approved by the user' + (f' ({isolation} isolation).' if isolation != 'none' else ' without sandbox.')
    return False, 'The user declined this run; the code was saved but not run.'


def _limits(seconds, memory_mb, file_mb):
    import resource

    def apply():
        os.setsid()
        cpu = max(1, int(seconds) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        size = int(file_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        if sys.platform.startswith('linux'):
            memory = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _read_capped(path, limit):
    with open(path, 'rb') as handle:
        data = handle.read(limit + 1)
    text = data[:limit].decode('utf-8', errors='replace')
    return text + ('\n[output truncated]' if len(data) > limit else '')


def execute(workdir, script, *, seconds=120, memory_mb=2048, file_mb=64, threads=1, isolation=None, cache=None):
    """Run ``script`` (a file inside ``workdir``) through the fixed launcher.

    ``cache`` is an optional writable folder kept between runs (font caches).
    """
    workdir = Path(workdir).resolve(strict=True)
    cache = Path(cache).resolve() if cache else workdir / 'tmp'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (workdir / script).is_file() or '/' in script:
        raise ValueError('The script must be a file directly inside the run folder')
    for name, value, low, high in (('seconds', seconds, 1, 86400), ('memory_mb', memory_mb, 128, 262144),
                                   ('file_mb', file_mb, 1, 4096), ('threads', threads, 1, 256)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
    isolation = isolation or available_isolation()
    (workdir / 'tmp').mkdir(exist_ok=True)
    (workdir / '_launcher.py').write_text(PRELUDE % {'package_root': PACKAGE_ROOT, 'script': script}, encoding='utf-8')
    python = [sys.executable, '-I', '-X', 'utf8', '_launcher.py']
    if isolation == 'bwrap':
        command = _bwrap(workdir)
        command[-3:-3] = ['--bind', str(cache), str(cache)]
        command += python
    elif isolation == 'seatbelt':
        profile = SEATBELT % str(workdir).replace('"', '')
        profile += '(allow file-write* (subpath "%s"))\n' % str(cache).replace('"', '')
        command = ['sandbox-exec', '-p', profile] + python
    elif isolation == 'netns':
        command = ['unshare', '-rn'] + python
    else:
        command = python
    count = str(int(threads))
    env = {'PATH': os.defpath, 'HOME': str(workdir), 'TMPDIR': str(workdir / 'tmp'), 'LANG': 'C.UTF-8',
           'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(cache), 'PYTHONHASHSEED': '0',
           'OMP_NUM_THREADS': count, 'OPENBLAS_NUM_THREADS': count, 'MKL_NUM_THREADS': count,
           'VECLIB_MAXIMUM_THREADS': count, 'NUMEXPR_NUM_THREADS': count}
    if isolation == 'bwrap':
        memory_mb = max(memory_mb, 512)
    started = time.monotonic()
    timed_out = False
    with open(workdir / 'stdout.txt', 'wb') as out, open(workdir / 'stderr.txt', 'wb') as err:
        process = subprocess.Popen(command, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=err, preexec_fn=_limits(seconds, memory_mb, file_mb))
        try:
            process.wait(timeout=seconds + 5)
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
    elapsed = time.monotonic() - started
    code = process.returncode
    stderr = _read_capped(workdir / 'stderr.txt', MAX_STREAM)
    reason = None
    if timed_out:
        reason = f'wall-clock limit of {seconds} s reached'
    elif code == -signal.SIGXCPU or code == 128 + signal.SIGXCPU:
        reason = f'CPU-time limit of {seconds} s reached'
    elif code in (-signal.SIGKILL, 128 + signal.SIGKILL):
        reason = 'killed (memory or CPU limit)'
    elif 'MemoryError' in stderr:
        reason = f'memory limit of {memory_mb} MB reached'
    elif code == -signal.SIGXFSZ or 'File too large' in stderr:
        reason = f'file-size limit of {file_mb} MB reached'
    results, results_error = None, None
    if (workdir / 'results.json').is_file():
        try:
            if (workdir / 'results.json').stat().st_size > MAX_RESULTS:
                raise ValueError('results.json exceeds 1 MB')
            results = json.loads((workdir / 'results.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            results_error = str(exc)
    figures = []
    folder = workdir / 'figures'
    if folder.is_dir() and not folder.is_symlink():
        for path in sorted(folder.iterdir()):
            if (path.suffix.lower() in FIGURE_SUFFIXES and path.is_file() and not path.is_symlink()
                    and path.stat().st_size <= MAX_FIGURE_BYTES and re.fullmatch(r'[A-Za-z0-9_-]+\.(png|svg)', path.name)):
                figures.append(path.name)
        figures = figures[:2 * MAX_FIGURES]
    return {'returncode': code, 'ok': code == 0 and not timed_out, 'timed_out': timed_out,
            'limit_reason': reason, 'seconds': round(elapsed, 3), 'isolation': isolation,
            'stdout': _read_capped(workdir / 'stdout.txt', MAX_STREAM), 'stderr': stderr,
            'results': results, 'results_error': results_error, 'figures': figures,
            'limits': {'seconds': seconds, 'memory_mb': memory_mb, 'file_mb': file_mb, 'threads': threads}}


def run_python(code, workdir, **limits):
    """Write model-written ``code`` to experiment.py in a fresh folder and run it."""
    if not isinstance(code, str) or not code.strip() or len(code) > MAX_CODE:
        raise ValueError(f'Provide Python source, at most {MAX_CODE} characters')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=False, mode=0o700)
    (workdir / 'experiment.py').write_text(code, encoding='utf-8')
    return execute(workdir, 'experiment.py', **limits)
