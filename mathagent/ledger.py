"""Durable, versioned storage for bounded proof attempts.

The JSON state is the controller's source of truth. Mathematical arguments live
in immutable artifacts; Markdown reports are replaceable views of that state.
Only local filesystem operations and the Python standard library are used.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterator
import uuid
import warnings


class LedgerError(ValueError):
    """A proof ledger is invalid, unsafe to access, or in use."""


class _VersionError(LedgerError):
    """A future schema must never be overwritten using an older backup."""


MAX_FILE_BYTES = 4 * 1024 * 1024
_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_ARTIFACT = re.compile(r"[0-9]{4,}-[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(?:md|jsonl)\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace(root) -> Path:
    try:
        result = Path(root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LedgerError(f"Cannot open proof workspace: {exc}") from exc
    if not result.is_dir():
        raise LedgerError(f"Proof workspace is not a directory: {result}")
    return result


def _directory(path: Path, *, create: bool = False) -> None:
    if path.is_symlink():
        raise LedgerError(f"Refusing symlink in proof storage: {path}")
    if create:
        try:
            path.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise LedgerError(f"Cannot create proof directory {path}: {exc}") from exc
    if not path.is_dir() or path.is_symlink():
        raise LedgerError(f"Missing or unsafe proof directory: {path}")


def _regular(path: Path, *, optional: bool = False) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if optional:
            return
        raise LedgerError(f"Missing proof file: {path}") from None
    if not stat.S_ISREG(info.st_mode):
        raise LedgerError(f"Refusing non-regular proof file: {path}")


def _read(path: Path) -> str:
    _regular(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise LedgerError(f"Refusing non-regular proof file: {path}")
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise LedgerError(f"Proof file exceeds {MAX_FILE_BYTES} bytes: {path}")
        return data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise LedgerError(f"Cannot read proof file {path}: {exc}") from exc


def _bytes(text: str) -> bytes:
    if not isinstance(text, str):
        raise LedgerError("Proof content must be text.")
    try:
        data = text.encode("utf-8")
    except UnicodeError as exc:
        raise LedgerError(f"Proof content is not valid UTF-8 text: {exc}") from exc
    if len(data) > MAX_FILE_BYTES:
        raise LedgerError(f"Proof content exceeds {MAX_FILE_BYTES} bytes.")
    return data


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one view or state file only after flushing a complete new file."""
    _regular(path, optional=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _regular(path, optional=True)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except OSError as exc:
        raise LedgerError(f"Cannot atomically write proof file {path.name}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _number(value, name: str, *, positive: bool = False, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"Invalid numeric ledger field: {name}")
    if integer and not isinstance(value, int):
        raise LedgerError(f"Ledger field must be an integer: {name}")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (positive and value == 0):
        raise LedgerError(f"Invalid budget or counter in ledger: {name}")


def _validate(state: dict, proof_id: str) -> None:
    if not isinstance(state, dict) or type(state.get("version")) is not int:
        raise LedgerError("Missing proof ledger version (expected 1).")
    if state["version"] != 1:
        raise _VersionError("Unsupported proof ledger version (expected 1).")
    if state.get("id") != proof_id:
        raise LedgerError("Proof ledger ID does not match its directory.")
    for key in ("goal", "status", "created_at", "updated_at", "next_task"):
        if not isinstance(state.get(key), str) or (key != "next_task" and not state[key].strip()):
            raise LedgerError(f"Missing or invalid ledger field: {key}")
    for key in ("revision", "rounds_started", "tokens_charged", "stagnant_rounds"):
        _number(state.get(key), key, integer=True)
    _number(state.get("seconds_used"), "seconds_used")
    for key in ("calls", "rounds", "claims", "sources"):
        if not isinstance(state.get(key), list):
            raise LedgerError(f"Ledger field must be a list: {key}")
    for source in state["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"]:
            raise LedgerError("Each pinned source must have a nonempty relative path.")
        path = Path(source["path"])
        if path.is_absolute() or any(part == ".." or part.startswith(".") for part in path.parts):
            raise LedgerError("Pinned source paths must stay within the nonhidden workspace.")
        if not isinstance(source.get("content"), str):
            raise LedgerError("Each pinned source must contain its original text snapshot.")
        if "sha256" in source:
            digest = hashlib.sha256(_bytes(source["content"])).hexdigest()
            if source["sha256"] != digest:
                raise LedgerError("Pinned source digest does not match its saved content.")
    for key in ("calls", "rounds", "claims"):
        if any(not isinstance(item, dict) for item in state[key]):
            raise LedgerError(f"Ledger entries must be objects: {key}")
    if not isinstance(state.get("settings"), dict):
        raise LedgerError("Ledger settings must be an object.")
    for key in ("max_rounds", "max_tokens", "max_seconds"):
        if key not in state["settings"]:
            raise LedgerError(f"Missing required proof budget: settings.{key}")
    for key in ("max_rounds", "token_budget", "max_tokens", "time_budget", "max_seconds", "predict", "ctx", "max_predict"):
        if key in state["settings"]:
            _number(state["settings"][key], f"settings.{key}", positive=True,
                    integer=key in {"max_rounds", "token_budget", "max_tokens", "predict", "ctx", "max_predict"})
    if "truncation_streak" in state:
        _number(state["truncation_streak"], "truncation_streak", integer=True)
    for key in ("pending", "final_audit"):
        if key not in state or (state[key] is not None and not isinstance(state[key], dict)):
            raise LedgerError(f"Ledger field must be an object or null: {key}")


def _decode(path: Path, proof_id: str) -> dict:
    try:
        state = json.loads(_read(path), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as exc:
        raise LedgerError(f"Invalid proof state in {path.name}: {exc}") from exc
    _validate(state, proof_id)
    return state


class ProofStore:
    """A proof job's atomic state, immutable arguments, and human-readable report."""

    def __init__(self, workspace: Path, directory: Path, state: dict, *, recovered: bool = False):
        self.workspace = workspace
        self.directory = directory
        self.state = state
        self._revision = state["revision"]
        self._recovered = recovered

    @classmethod
    def create(cls, workspace_root, goal: str, settings: dict, sources: list | None = None) -> "ProofStore":
        workspace = _workspace(workspace_root)
        proof_id = str(uuid.uuid4())
        stamp = _now()
        # JSON round-trip detaches caller-owned data and rejects non-JSON values.
        try:
            settings = json.loads(json.dumps(settings, allow_nan=False))
            sources = json.loads(json.dumps(sources if sources is not None else [], allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"Proof settings and sources must be JSON serializable: {exc}") from exc
        state = {
            "version": 1, "id": proof_id, "goal": goal, "settings": settings,
            "sources": sources, "created_at": stamp, "updated_at": stamp, "revision": 0,
            "status": "ready", "rounds_started": 0, "tokens_charged": 0,
            "seconds_used": 0, "calls": [], "rounds": [], "claims": [],
            "next_task": "Explore a promising proof strategy and establish one useful intermediate claim.",
            "stagnant_rounds": 0, "pending": None, "final_audit": None,
        }
        _validate(state, proof_id)
        base = workspace / ".mathagent"
        _directory(base, create=True)
        _directory(base / "proofs", create=True)
        directory = base / "proofs" / proof_id
        try:
            directory.mkdir(mode=0o700)
        except OSError as exc:
            raise LedgerError(f"Cannot create proof job: {exc}") from exc
        _directory(directory / "artifacts", create=True)
        store = cls(workspace, directory, state)
        store.save()
        return store

    @classmethod
    def load(cls, workspace_root, proof_id: str) -> "ProofStore":
        if not isinstance(proof_id, str) or not _ID.fullmatch(proof_id):
            raise LedgerError("Invalid proof ID; use the UUID shown by /proofs.")
        workspace = _workspace(workspace_root)
        directory = workspace / ".mathagent" / "proofs" / proof_id
        for parent in (workspace / ".mathagent", directory.parent, directory, directory / "artifacts"):
            _directory(parent)
        # An unsafe path is not ordinary corruption and must never trigger fallback.
        for name in ("state.json", "state.backup.json", "run.lock", "state.lock", "report.md", "ledger.md", "proof.md"):
            _regular(directory / name, optional=True)
        state, recovered = cls._read_state(directory, proof_id)
        if recovered:
            notice = ("Recovered the last valid backup after state.json was missing or corrupt. "
                      "The most recent state update may be lost; immutable artifacts were retained.")
            state["recovery_notice"] = notice
            warnings.warn(f"Proof {proof_id}: {notice}", RuntimeWarning, stacklevel=2)
        return cls(workspace, directory, state, recovered=recovered)

    @staticmethod
    def _read_state(directory: Path, proof_id: str) -> tuple[dict, bool]:
        try:
            return _decode(directory / "state.json", proof_id), False
        except _VersionError:
            raise
        except LedgerError as original:
            try:
                return _decode(directory / "state.backup.json", proof_id), True
            except LedgerError as backup:
                raise LedgerError(f"Cannot load proof {proof_id}: {original}; backup unavailable or invalid: {backup}") from original

    def _check_paths(self) -> None:
        for path in (self.workspace / ".mathagent", self.directory.parent, self.directory,
                     self.directory / "artifacts"):
            _directory(path)
        for name in ("state.json", "state.backup.json", "run.lock", "state.lock", "report.md", "ledger.md", "proof.md"):
            _regular(self.directory / name, optional=True)

    @contextmanager
    def _file_lock(self, name: str) -> Iterator[None]:
        self._check_paths()
        path = self.directory / name
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise LedgerError(f"Cannot open proof lock: {exc}") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise LedgerError(f"Unsafe proof lock: {path}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LedgerError("This proof is already in use by another operation; try again after it finishes.") from exc
            yield
        finally:
            os.close(fd)

    @contextmanager
    def lock(self) -> Iterator["ProofStore"]:
        """Hold the nonblocking run lock throughout a controller invocation."""
        with self._file_lock("run.lock"):
            yield self

    def save(self) -> None:
        """Commit state atomically, rejecting another writer's newer revision."""
        with self._file_lock("state.lock"):
            main = self.directory / "state.json"
            backup = self.directory / "state.backup.json"
            prior = None
            if main.exists() or backup.exists():
                prior, recovered = self._read_state(self.directory, self.state["id"])
                if prior["revision"] != self._revision:
                    raise LedgerError("Proof state changed in another process; reload it before saving.")
                if recovered and not self._recovered:
                    raise LedgerError("Proof state was corrupted after loading; reload it to recover the backup explicitly.")
            elif self._revision != 0:
                raise LedgerError("Proof state disappeared after loading; refusing to recreate it silently.")
            candidate = dict(self.state)
            candidate["revision"] = self._revision + 1
            candidate["updated_at"] = _now()
            _validate(candidate, self.directory.name)
            try:
                data = _bytes(json.dumps(candidate, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                previous = _bytes(json.dumps(prior, indent=2, ensure_ascii=False, allow_nan=False) + "\n") if prior else None
            except (TypeError, ValueError) as exc:
                raise LedgerError(f"Cannot serialize proof state: {exc}") from exc
            if previous is not None:
                _atomic_write(backup, previous)
            _atomic_write(main, data)
            self.state.update(candidate)
            self._revision = candidate["revision"]
            self._recovered = False

    def write_artifact(self, label: str, text: str) -> str:
        """Write an immutable, complete artifact and return its safe basename."""
        return self._new_artifact(label, _bytes(text), "md")

    def start_stream(self, label: str) -> str:
        """Create a durable empty append-only event journal before inference."""
        return self._new_artifact(label, b"", "jsonl")

    def _new_artifact(self, label: str, data: bytes, extension: str) -> str:
        self._check_paths()
        if not isinstance(label, str) or not label.strip() or any(v in label for v in ("/", "\\", "..")):
            raise LedgerError("Artifact label must be a simple name without path components.")
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-._")
        if label.endswith(".md"):
            label = label[:-3]
        elif label.endswith(".jsonl"):
            label = label[:-6]
        label = label[:80]
        if not label:
            raise LedgerError("Artifact label must contain a letter or digit.")
        directory = self.directory / "artifacts"
        numbers = [int(item.name.split("-", 1)[0]) for item in directory.iterdir() if _ARTIFACT.fullmatch(item.name)]
        number = max(numbers, default=0) + 1
        fd, temporary = tempfile.mkstemp(prefix=".artifact.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            while True:
                name = f"{number:04d}-{label}.{extension}"
                try:
                    # link() publishes without replacing any existing name.
                    os.link(temporary, directory / name)
                    break
                except FileExistsError:
                    number += 1
            _sync_directory(directory)
            return name
        except OSError as exc:
            raise LedgerError(f"Cannot write proof artifact: {exc}") from exc
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def append_stream(self, filename: str, event: dict) -> None:
        """Append and flush an event; a killed process may leave a partial last line."""
        self._check_paths()
        if (not isinstance(filename, str) or not _ARTIFACT.fullmatch(filename)
                or ".." in filename or not filename.endswith(".jsonl")):
            raise LedgerError("Invalid proof stream filename.")
        if not isinstance(event, dict):
            raise LedgerError("A proof stream event must be an object.")
        try:
            data = _bytes(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"Cannot serialize proof stream event: {exc}") from exc
        path = self.directory / "artifacts" / filename
        _regular(path)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise LedgerError(f"Unsafe proof stream: {path}")
                if info.st_size + len(data) > MAX_FILE_BYTES:
                    raise LedgerError(f"Proof stream exceeds {MAX_FILE_BYTES} bytes.")
                remaining = memoryview(data)
                while remaining:
                    written = os.write(fd, remaining)
                    if written == 0:
                        raise LedgerError("Failed to append a proof stream event.")
                    remaining = remaining[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise LedgerError(f"Cannot append proof stream: {exc}") from exc

    def read_artifact(self, filename: str) -> str:
        self._check_paths()
        if not isinstance(filename, str) or not _ARTIFACT.fullmatch(filename) or ".." in filename:
            raise LedgerError("Invalid proof artifact filename.")
        return _read(self.directory / "artifacts" / filename)

    def write_report(self, text: str) -> None:
        self._check_paths()
        _atomic_write(self.directory / "report.md", _bytes(text))

    def write_ledger(self, text: str) -> None:
        self._check_paths()
        _atomic_write(self.directory / "ledger.md", _bytes(text))

    def write_proof(self, text: str) -> None:
        """Export the exact audited candidate separately from its audit trail."""
        self._check_paths()
        _atomic_write(self.directory / "proof.md", _bytes(text))

    @classmethod
    def list(cls, workspace_root) -> list[dict]:
        workspace = _workspace(workspace_root)
        base = workspace / ".mathagent"
        if not base.exists() and not base.is_symlink():
            return []
        _directory(base)
        parent = base / "proofs"
        if not parent.exists() and not parent.is_symlink():
            return []
        _directory(parent)
        jobs = []
        for path in sorted(parent.iterdir()):
            if not _ID.fullmatch(path.name):
                continue
            try:
                store = cls.load(workspace, path.name)
            except LedgerError as exc:
                warnings.warn(f"Skipping unreadable proof {path.name}: {exc}", RuntimeWarning, stacklevel=2)
                continue
            jobs.append({key: store.state[key] for key in ("id", "status", "goal", "rounds_started", "updated_at")})
        return sorted(jobs, key=lambda job: job["updated_at"], reverse=True)
