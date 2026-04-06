from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> int:
    out_path = pathlib.Path("test_run_output_unit.txt")
    cmd = [sys.executable, "-m", "pytest", "-v", "-m", "unit", "tests", "app/tests", *sys.argv[1:]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_path.write_text(proc.stdout + "\n\n==== STDERR ====\n" + proc.stderr, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.stderr:
        print("\n==== STDERR ====\n" + proc.stderr, file=sys.stderr)
    print(f"\nExit code: {proc.returncode}")
    print(f"Wrote: {out_path.resolve()}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
