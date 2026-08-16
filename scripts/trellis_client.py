"""Dependency-free client for the TRELLIS.2 inference API.

Standard library only, so it runs from any Python 3.9+ on the host, in WSL,
or inside the container without installing anything.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ApiError(RuntimeError):
    pass


def _request(url: str, *, data: bytes | None = None, headers: dict | None = None,
             method: str = "GET", timeout: float = 60.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ApiError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"Cannot reach {url}: {exc.reason}") from exc


def _multipart(image_path: Path, params: dict) -> tuple[bytes, str]:
    """Encode one file plus a `params` JSON field as multipart/form-data."""
    boundary = f"----trellis{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="params"\r\n\r\n')
    parts.append(json.dumps(params).encode() + b"\r\n")

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
    parts.append(image_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class TrellisClient:
    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base = base_url.rstrip("/")

    def health(self) -> dict:
        _, body = _request(f"{self.base}/healthz", timeout=15)
        return json.loads(body)

    def wait_until_ready(self, timeout: float = 1800.0, quiet: bool = False) -> dict:
        """Block until the model has finished loading (cold start is minutes)."""
        deadline = time.time() + timeout
        announced = False
        while time.time() < deadline:
            try:
                h = self.health()
            except ApiError:
                h = {"status": "unreachable"}
            if h.get("status") == "healthy":
                return h
            if h.get("status") == "unhealthy":
                raise ApiError(f"Service unhealthy: {h.get('load_error')}")
            if not quiet and not announced:
                print("Waiting for the model to finish loading...")
                announced = True
            time.sleep(5)
        raise ApiError(f"Model was not ready within {timeout:.0f}s")

    def submit(self, image_path: Path, params: dict) -> str:
        body, content_type = _multipart(image_path, params)
        _, resp = _request(
            f"{self.base}/v1/generate",
            data=body,
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
            method="POST",
            timeout=300,
        )
        return json.loads(resp)["job_id"]

    def job(self, job_id: str) -> dict:
        _, body = _request(f"{self.base}/v1/jobs/{job_id}", timeout=30)
        return json.loads(body)

    def wait(self, job_id: str, timeout: float = 3600.0, poll: float = 3.0,
             on_tick=None) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.job(job_id)
            if job["status"] in ("completed", "failed"):
                return job
            if on_tick:
                on_tick(job)
            time.sleep(poll)
        raise ApiError(f"Job {job_id} did not finish within {timeout:.0f}s")

    def download(self, job_id: str, artifact: str, dest: Path) -> Path:
        _, body = _request(f"{self.base}/v1/jobs/{job_id}/{artifact}", timeout=600)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return dest


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def human_size(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
