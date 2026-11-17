"""Compare installed package versions with the committed lock (configs/requirements.lock.txt).

An unpinned ``pip install vllm`` on a fresh Kaggle session once moved the engine from 0.33.0 to
0.34.0 without any message. This check makes that visible: the Step 7 cell runs it before any
benchmark, and every run manifest stores the result.

Command line:  python -m benchmarks.envcheck [lock_file]
Exit code 1 when vllm differs from the lock (a different engine is a different experiment).
"""

from __future__ import annotations

import pathlib
import sys
from typing import Optional

DEFAULT_LOCK = pathlib.Path("configs/requirements.lock.txt")
MUST_MATCH = ("vllm",)


def norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_lock(text: str) -> dict:
    """``name==version`` lines; comments and blank lines ignored."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            out[norm(name)] = version.strip()
    return out


def compare(installed: dict, lock: dict) -> list:
    """Mismatches as dicts; only packages present in the lock are checked."""
    inst = {norm(k): v for k, v in installed.items()}
    problems = []
    for name, want in sorted(lock.items()):
        have = inst.get(name)
        if have != want:
            problems.append({"package": name, "locked": want, "installed": have,
                             "blocking": name in MUST_MATCH})
    return problems


def check(lock_path: Optional[pathlib.Path] = None) -> Optional[list]:
    """None when the lock cannot be read; otherwise the list of mismatches."""
    from benchmarks import runner
    path = pathlib.Path(lock_path or DEFAULT_LOCK)
    try:
        lock = parse_lock(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return compare(runner.installed_versions(tuple(lock)), lock)


def main() -> None:
    problems = check(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    if problems is None:
        sys.exit("cannot read the lock file")
    if not problems:
        print("environment matches the lock")
        return
    for p in problems:
        print(f"{'BLOCKING ' if p['blocking'] else 'differs  '}{p['package']}: "
              f"locked {p['locked']}, installed {p['installed']}")
    sys.exit(1 if any(p["blocking"] for p in problems) else 0)


if __name__ == "__main__":
    main()
