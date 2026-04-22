#!/usr/bin/env python3
"""Refresh generated release PR artifacts after a version bump."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def sync_bun_lock() -> None:
    ui_package = json.loads((ROOT / "packages/fers-ui/package.json").read_text(encoding="utf-8"))
    expected_version = ui_package["version"]
    bun_lock_path = ROOT / "bun.lock"
    bun_lock = bun_lock_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r'("packages/fers-ui":\s*\{.*?^\s*"version":\s*")([^"]+)(",)',
        re.MULTILINE | re.DOTALL,
    )
    updated_lock, replacements = pattern.subn(rf"\g<1>{expected_version}\g<3>", bun_lock, count=1)
    if replacements != 1:
        raise RuntimeError("Failed to locate the fers-ui workspace version entry in bun.lock")

    bun_lock_path.write_text(updated_lock, encoding="utf-8")


def main() -> int:
    run("bun", "install")
    sync_bun_lock()
    run("bun", "run", "licenses:js")
    run("bun", "run", "licenses:rust")
    run("python3", "scripts/verify_versions.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
