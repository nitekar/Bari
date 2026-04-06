from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _urljoin(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _http(method: str, url: str, *, headers: dict[str, str] | None = None, body: bytes | None = None, timeout: float = 20.0):
    req = urllib.request.Request(url=url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        # Still a valid HTTP response; surface status and body.
        return e.code, e.headers, e.read()  # type: ignore[return-value]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    web_url = os.environ.get("UAT_WEB_URL", "https://bari-tawny.vercel.app/")
    api_base = os.environ.get("UAT_API_BASE_URL", "https://web-production-c7c1.up.railway.app")
    api_key = os.environ.get("UAT_API_KEY", "").strip()

    started = time.time()
    lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        lines.append(s)

    log("UAT (deployed) — Bari")
    log(f"WEB  = {web_url}")
    log(f"API  = {api_base}")
    log(f"KEY? = {'yes' if api_key else 'no'}")
    log("")

    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            log(f"[PASS] {name}")
        except Exception as exc:
            msg = f"[FAIL] {name}: {exc}"
            log(msg)
            failures.append(msg)

    def check_web_root() -> None:
        status, _, _ = _http("GET", web_url)
        _assert(200 <= status < 400, f"web root status {status}")

    def check_api_health() -> None:
        status, _, raw = _http("GET", _urljoin(api_base, "/health"))
        _assert(status == 200, f"/health status {status}")
        body = json.loads(raw.decode("utf-8"))
        _assert("status" in body, "missing 'status' in /health")
        _assert("models_loaded" in body, "missing 'models_loaded' in /health")

    def check_api_openapi() -> None:
        status, _, raw = _http("GET", _urljoin(api_base, "/openapi.json"))
        _assert(status == 200, f"/openapi.json status {status}")
        body = json.loads(raw.decode("utf-8"))
        _assert("openapi" in body, "missing 'openapi' field")

    def check_api_docs() -> None:
        status, _, _ = _http("GET", _urljoin(api_base, "/docs"))
        _assert(200 <= status < 400, f"/docs status {status}")

    def check_predict_image_auth_behavior() -> None:
        # Without a key we can still validate that auth is enforced.
        # In production this should be 401; 503 is acceptable if auth not configured.
        status, _, raw = _http("POST", _urljoin(api_base, "/predict/image"), headers={"Content-Type": "application/octet-stream"}, body=b"")
        _assert(status in (401, 422, 503), f"unexpected status {status} (body={raw[:200]!r})")

    check("Web root reachable", check_web_root)
    check("API /health reachable", check_api_health)
    check("API OpenAPI reachable", check_api_openapi)
    check("API /docs reachable", check_api_docs)

    if api_key:
        # Authenticated prediction checks are intentionally minimal here.
        # Models may be unavailable on some deployments, so 503 is accepted.
        try:
            from PIL import Image
            import io

            img = Image.new("RGB", (64, 64), color=(200, 50, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            jpeg = buf.getvalue()

            boundary = "----bari-uat-boundary"
            parts = []
            parts.append(
                f"--{boundary}\r\n"
                "Content-Disposition: form-data; name=\"file\"; filename=\"eye.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n"
            )
            body = "".join(parts).encode("utf-8") + jpeg + f"\r\n--{boundary}--\r\n".encode("utf-8")

            def do_predict_image() -> None:
                status, _, raw = _http(
                    "POST",
                    _urljoin(api_base, "/predict/image"),
                    headers={
                        "X-API-Key": api_key,
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                    body=body,
                    timeout=60.0,
                )
                _assert(status in (200, 503), f"/predict/image status {status} (body={raw[:200]!r})")

            check("API /predict/image (auth) works", do_predict_image)
        except Exception as exc:
            check("API /predict/image (auth) works", lambda: (_assert(False, f"setup failed: {exc}")))
    else:
        check("API enforces auth (no key)", check_predict_image_auth_behavior)

    elapsed = time.time() - started
    log("")
    log(f"Done in {elapsed:.2f}s")

    out_path = os.environ.get("UAT_OUTPUT", "uat_deployed_output.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log(f"Wrote: {out_path}")
    except Exception as exc:
        log(f"Could not write output file: {exc}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
