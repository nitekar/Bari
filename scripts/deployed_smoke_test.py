from __future__ import annotations

import datetime as _dt
import sys
import urllib.error
import urllib.request
from typing import Iterable


def _now_utc() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _check(url: str, *, expect_status: Iterable[int] = (200,)) -> tuple[bool, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "bari-smoke-test/1.0"},
        method="GET",
    )

    try:
        # urllib follows redirects by default.
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            status = getattr(resp, "status", 200)
            final_url = getattr(resp, "url", url)
            ok = status in set(expect_status)
            return ok, f"{status} (final_url={final_url})"
    except urllib.error.HTTPError as exc:
        final_url = getattr(exc, "url", url)
        ok = exc.code in set(expect_status)
        return ok, f"{exc.code} {exc.reason} (final_url={final_url})"
    except Exception as exc:  # noqa: BLE001
        return False, f"ERROR: {type(exc).__name__}: {exc}"


def main() -> int:
    railway_base = "https://web-production-c7c1.up.railway.app"
    vercel_web = "https://bari-tawny.vercel.app/"

    checks: list[tuple[str, str, Iterable[int]]] = [
        ("Railway health", f"{railway_base}/health", (200,)),
        ("Railway docs", f"{railway_base}/docs", (200,)),
        ("Railway openapi", f"{railway_base}/openapi.json", (200,)),
        ("Vercel web root", vercel_web, (200,)),
    ]

    lines: list[str] = []
    lines.append("Bari deployed smoke test (public endpoints)")
    lines.append(f"Timestamp (UTC): {_now_utc()}")
    lines.append("")

    any_failed = False
    for name, url, expect in checks:
        ok, detail = _check(url, expect_status=expect)
        status = "PASS" if ok else "FAIL"
        if not ok:
            any_failed = True
        lines.append(f"- {status}: {name}: {url} -> {detail}")

    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
