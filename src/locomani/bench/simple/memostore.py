"""TODO 53: the memo's states on disk, so the next process (the loop runs one setting per process) starts from them.

One file per stored item (a state after a primitive, or one perception output) under `<root>/<signature>/<ab>/<name>`. The
signature is the bed's (adapter files, switches, render mode: the loop's, plus a hash of the code that makes a state, memo_signature
in run.py), so a changed environment makes every old state unreachable; the name is the memo's chain key.

  * Atomic: a file is written to a temporary name in its folder and renamed, so a reader sees all of a file or none of it; up to
    three slots (processes) read and write the same store.
  * A file that cannot be read back (truncated, bit-rotten, another format) is a miss, never an error: every file carries its
    length and a CRC.
  * Bounded: `max_bytes` (default 20 GB) is enforced by `prune`, oldest first (a hit touches the file, so what is used stays).
    One process prunes at a time (a lock file); a process that finds the lock taken skips.
  * `stats()` is the line each row carries: hits, misses, writes, bytes read and written, files dropped, corrupt reads.
"""

from __future__ import annotations

import fcntl
import os
import struct
import time
import zlib
from pathlib import Path

MAGIC = b"LMS1"
HEAD = struct.Struct("<4sQI")  # magic, payload length, crc32 of the payload
GB = 1 << 30
TMP = ".tmp."


class DiskStore:
    def __init__(self, root: str | Path, signature: str, max_bytes: int = 20 * GB, prune_every: int = 64):
        self.root = Path(root)
        self.dir = self.root / signature
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.prune_every = int(max_bytes), prune_every
        self.hits = self.misses = self.writes = self.corrupt = self.pruned = 0
        self.bytes_read = self.bytes_written = 0
        self._since_prune = 0
        self.prune()

    # ---- files

    def path(self, name: str) -> Path:
        return self.dir / name[:2] / name

    def get(self, name: str) -> bytes | None:
        """The payload stored under `name`, or None (absent, or unreadable: counted, and the bad file removed)."""
        f = self.path(name)
        try:
            raw = f.read_bytes()
        except OSError:
            self.misses += 1
            return None
        try:
            magic, n, crc = HEAD.unpack_from(raw)
            body = raw[HEAD.size :]
            if magic != MAGIC or len(body) != n or zlib.crc32(body) != crc:
                raise ValueError("bad file")
        except (struct.error, ValueError):
            self.corrupt += 1
            self.misses += 1
            try:
                f.unlink()
            except OSError:
                pass
            return None
        self.hits += 1
        self.bytes_read += len(raw)
        try:
            os.utime(f)  # in use: the last to be pruned
        except OSError:
            pass
        return body

    def has(self, name: str) -> bool:
        return self.path(name).is_file()

    def put(self, name: str, payload: bytes) -> bool:
        """Store `payload` under `name` atomically; False when it could not be written (the disk is full, the folder is gone)."""
        f = self.path(name)
        tmp = f.with_name(f"{TMP}{os.getpid()}.{time.time_ns()}.{name[:12]}")
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                fh.write(HEAD.pack(MAGIC, len(payload), zlib.crc32(payload)))
                fh.write(payload)
                fh.flush()
            os.replace(tmp, f)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False
        self.writes += 1
        self.bytes_written += HEAD.size + len(payload)
        self._since_prune += 1
        if self._since_prune >= self.prune_every:
            self.prune()
        return True

    # ---- the size cap

    def _files(self):
        """(mtime, size, path) of every item of every signature under the root; temporary files are returned apart."""
        items, temps = [], []
        for sig in self.root.iterdir():
            if not sig.is_dir():
                continue
            for sub in sig.iterdir():
                if not sub.is_dir():
                    continue
                for p in sub.iterdir():
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    (temps if p.name.startswith(TMP) else items).append((st.st_mtime, st.st_size, p))
        return items, temps

    def size(self) -> int:
        return sum(s for _, s, _ in self._files()[0])

    def prune(self) -> int:
        """Delete the oldest items until the store (every signature under the root) is at 90 % of the cap; -> bytes freed. A
        process that cannot take the lock does nothing (another one is pruning)."""
        self._since_prune = 0
        lock = self.root / ".prune.lock"
        try:
            fh = open(lock, "w")
        except OSError:
            return 0
        try:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return 0
            items, temps = self._files()
            now = time.time()
            for mt, _, p in temps:  # a writer that died
                if now - mt > 3600:
                    try:
                        p.unlink()
                    except OSError:
                        pass
            total = sum(s for _, s, _ in items)
            if total <= self.max_bytes:
                return 0
            goal, freed = int(0.9 * self.max_bytes), 0
            for _, s, p in sorted(items, key=lambda t: t[0]):
                if total - freed <= goal:
                    break
                try:
                    p.unlink()
                except OSError:
                    continue
                freed += s
                self.pruned += 1
            return freed
        finally:
            fh.close()

    def stats(self) -> dict:
        return dict(
            hits=self.hits,
            misses=self.misses,
            writes=self.writes,
            corrupt=self.corrupt,
            pruned=self.pruned,
            bytes_read=self.bytes_read,
            bytes_written=self.bytes_written,
        )
