"""Where results are written and where the archived results live.

ARCHIVE  the results shipped with the paper (results/ in the repository),
         produced on the first author's laptop GPU; never written to by the
         verification scripts.
OUTPUT   where a run writes: the directory named by the environment variable
         CAMPS_RESULTS_DIR if set, otherwise ARCHIVE. On a verification
         system (e.g. TACC) set CAMPS_RESULTS_DIR to a fresh directory so the
         archived files are preserved and can be compared against.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "results")


def results_dir() -> str:
    d = os.environ.get("CAMPS_RESULTS_DIR") or ARCHIVE
    os.makedirs(d, exist_ok=True)
    return d


def archive_dir() -> str:
    return ARCHIVE
