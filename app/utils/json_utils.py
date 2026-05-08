"""
Canonical JSON utilities for Locopilot Monitoring System.

This module provides:

* :class:`NumpyEncoder` — the **single** JSON encoder used everywhere we
  serialise activity payloads. It handles every numpy scalar type the
  Pipeline-1 detectors emit (``np.bool_``, ``np.float32``, ``np.int64``)
  plus :class:`numpy.ndarray`.

* :func:`atomic_write_json` — crash-safe + cross-process-safe writer for
  artefacts whose corruption would break a run (e.g. ``activities.json``).
  It acquires an exclusive ``portalocker`` file lock, writes to a sibling
  temp file with ``fsync``, then ``os.replace``s into place. If anything
  goes wrong before ``os.replace`` the temp file is removed and the
  original target file is left untouched.

Historically there were three independent ``NumpyEncoder`` classes in the
codebase, each missing a different numpy scalar type, and three
non-atomic ``open(..., 'w')`` writers racing on the same file. This
module consolidates that into one definition.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import portalocker


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles every numpy type emitted by Pipeline-1.

    The implementation relies on the fact that every numpy scalar exposes
    a ``.item()`` method that returns the equivalent native Python type.
    This covers ``np.bool_``, ``np.integer`` and ``np.floating`` in one
    branch — there is no need to enumerate them.
    """

    def default(self, o: Any) -> Any:  # noqa: D401 - JSONEncoder override
        # ``hasattr(o, "item")`` covers np.bool_, np.float32, np.int64,
        # etc. Native Python scalars (bool/int/float) never reach
        # ``default`` so this is safe.
        if hasattr(o, "item"):
            try:
                return o.item()
            except (TypeError, ValueError):
                pass
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def atomic_write_json(path: str, payload: Any, *, indent: int = 2) -> None:
    """Crash-safe and cross-process-safe JSON writer.

    The write protocol is:

    1. Acquire ``portalocker.Lock`` on ``<path>.lock`` (exclusive, 30s timeout).
    2. ``mkstemp`` a sibling file in the same directory so ``os.replace``
       stays on the same filesystem (atomic on POSIX + Windows).
    3. ``json.dump`` using :class:`NumpyEncoder`, then ``flush`` + ``fsync``.
    4. ``os.replace(tmp, path)`` — atomically swaps the file into place.
    5. On any exception before step 4, the temp file is removed and the
       original ``path`` is left intact.

    Parameters
    ----------
    path:
        Target file path. The parent directory is created if it does not
        already exist (callers used to be on the hook for this; we now
        guarantee it so the lock file open below cannot fail with
        ``FileNotFoundError``).
    payload:
        Any JSON-serialisable Python object (numpy scalars + ndarrays
        accepted via :class:`NumpyEncoder`).
    indent:
        Pretty-print indent, defaults to 2 to match the previous writers.
    """
    directory = os.path.dirname(path) or "."
    # Ensure the parent directory exists *before* portalocker tries to
    # create the lock file there. Otherwise the very first write to a
    # fresh run directory raises FileNotFoundError on the lock open.
    os.makedirs(directory, exist_ok=True)
    lock_path = path + ".lock"

    with portalocker.Lock(lock_path, flags=portalocker.LOCK_EX, timeout=30):
        fd, tmp = tempfile.mkstemp(prefix=".activities.", suffix=".tmp", dir=directory)
        try:
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(
                        payload,
                        f,
                        indent=indent,
                        ensure_ascii=False,
                        cls=NumpyEncoder,
                    )
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except Exception:
                # Best-effort cleanup of the temp file. We re-raise the
                # original exception so the caller observes the failure.
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                raise
        finally:
            # Best-effort cleanup of the lock file once the protected
            # section is over. On POSIX the unlink is fine even while
            # other processes are blocked on the lock — they hold an
            # fcntl flock on the open fd, not the directory entry.
            #
            # On Windows the lock is exclusive on the file itself, so
            # this unlink may fail because portalocker still has the
            # handle open. That is fine: the lock file lives on, gets
            # reused by the next writer, and never accumulates further.
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
            except OSError:
                # Windows: file is still locked. Leave it — it will be
                # reused, not duplicated, by the next acquirer.
                pass
