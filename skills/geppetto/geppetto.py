#!/usr/bin/env python3
"""Geppetto in three commands. Standard library only.

    geppetto.py up                         start the server, open the web UI
    geppetto.py process FILE [FILE...]     image -> 3D model next to the image
    geppetto.py process all [DIR]          every image in DIR (default: here)

Models are written beside each image with the same name and the right
extension (photo.webp -> photo.stl). Existing models are never replaced
unless --overwrite is given.

`process` starts the server itself if it is not already running.

Environment:
    GEPPETTO_URL    server address (default http://localhost:8080)
    GEPPETTO_REPO   Geppetto checkout, used only if the container has never
                    been created and `docker compose up` is needed
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path

URL = os.environ.get("GEPPETTO_URL", "http://localhost:8080").rstrip("/")
CONTAINER = "geppetto-api"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FORMATS = ("stl", "glb", "obj", "ply")
LOAD_TIMEOUT = 900      # model load is ~90 s; allow for a cold disk
JOB_TIMEOUT = 3600


class Fail(Exception):
    pass


# --- HTTP -------------------------------------------------------------------

def http(path: str, *, data: bytes | None = None, headers: dict | None = None,
         timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(URL + path, data=data, headers=headers or {},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise Fail(f"HTTP {exc.code} from {path}: "
                   f"{exc.read().decode('utf-8', 'replace')[:500]}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise Fail(f"cannot reach {URL}{path}: {getattr(exc, 'reason', exc)}") from exc


def health() -> dict | None:
    """The /healthz document, or None when nothing is answering."""
    try:
        return json.loads(http("/healthz", timeout=10))
    except (Fail, ValueError):
        return None


# --- starting the server ----------------------------------------------------

def docker_cmd() -> tuple[list[str], bool]:
    """How to invoke docker here, and whether that goes through WSL."""
    if shutil.which("docker"):
        return ["docker"], False
    if shutil.which("wsl"):
        return ["wsl", "-e", "docker"], True
    raise Fail("docker not found (looked for `docker`, then `wsl -e docker`). "
               f"Start the Geppetto server yourself, or set GEPPETTO_URL.")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)


def find_repo() -> Path | None:
    candidates = [os.environ.get("GEPPETTO_REPO"), Path(__file__).resolve().parents[2]]
    for c in candidates:
        if c and (Path(c) / "docker-compose.yml").is_file():
            return Path(c)
    return None


def start_container() -> None:
    docker, via_wsl = docker_cmd()
    listed = run(docker + ["ps", "-a", "--filter", f"name=^{CONTAINER}$",
                           "--format", "{{.Names}}"])
    if listed.returncode != 0:
        raise Fail(f"docker is not usable: {listed.stderr.strip() or listed.stdout.strip()}")

    if CONTAINER in listed.stdout.split():
        res = run(docker + ["start", CONTAINER])
    else:
        repo = find_repo()
        if repo is None:
            raise Fail(f"no `{CONTAINER}` container exists yet and no Geppetto checkout "
                       "was found. Set GEPPETTO_REPO to the repo and retry "
                       "(first-time setup: see HOWTO.md).")
        if via_wsl:
            res = run(["wsl", "--cd", str(repo), "-e", "docker", "compose", "up", "-d", "api"])
        else:
            res = run(["docker", "compose", "up", "-d", "api"], cwd=repo)
    if res.returncode != 0:
        raise Fail(f"could not start the server: {res.stderr.strip() or res.stdout.strip()}")


def is_local() -> bool:
    host = urllib.parse.urlsplit(URL).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def ensure_up() -> dict:
    """Return a healthy /healthz document, starting the server if need be."""
    h = health()
    if h and h.get("model_loaded"):
        return h
    if h is None:
        if not is_local():
            raise Fail(f"{URL} is not answering, and it is not this machine, "
                       "so it cannot be started from here.")
        print(f"Server not running - starting {CONTAINER} ...", flush=True)
        start_container()

    print("Loading the model (about 90 s) ", end="", flush=True)
    deadline = time.time() + LOAD_TIMEOUT
    while time.time() < deadline:
        h = health()
        if h and h.get("model_loaded"):
            print(" ready.", flush=True)
            return h
        if h and (h.get("load_error") or h.get("status") == "unhealthy"):
            print()
            raise Fail(f"the model failed to load: {h.get('load_error')}")
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    raise Fail(f"the model was not ready after {LOAD_TIMEOUT} s. "
               f"Check `docker logs {CONTAINER}`.")


# --- converting -------------------------------------------------------------

def multipart(image: Path, params: dict) -> tuple[bytes, str]:
    boundary = f"----geppetto{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="params"\r\n\r\n',
        json.dumps(params).encode(), b"\r\n",
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        image.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


def convert(image: Path, formats: list[str], resolution: int) -> list[Path]:
    """Generate one model and write it beside the image. Returns files written."""
    params = {"resolution": resolution,
              "export_formats": [f for f in formats if f != "glb"]}
    body, ctype = multipart(image, params)
    job_id = json.loads(http("/v1/generate", data=body, timeout=300,
                             headers={"Content-Type": ctype}))["job_id"]

    deadline = time.time() + JOB_TIMEOUT
    while True:
        job = json.loads(http(f"/v1/jobs/{job_id}", timeout=30))
        if job["status"] in ("completed", "failed"):
            break
        if time.time() > deadline:
            raise Fail(f"job {job_id} did not finish within {JOB_TIMEOUT} s")
        time.sleep(3)
    if job["status"] != "completed":
        raise Fail(f"generation failed: {job.get('error')}")

    written = []
    for fmt in formats:
        dest = image.with_suffix(f".{fmt}")
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(http(f"/v1/jobs/{job_id}/asset.{fmt}", timeout=600))
        os.replace(part, dest)
        written.append(dest)
    return written


def cmd_process(args) -> int:
    formats = list(dict.fromkeys(args.format or ["stl"]))
    batch = args.targets[0] == "all"

    if batch:
        if len(args.targets) > 2:
            raise Fail("usage: process all [DIR]")
        folder = Path(args.targets[1]) if len(args.targets) == 2 else Path.cwd()
        if not folder.is_dir():
            raise Fail(f"not a directory: {folder}")
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise Fail(f"no images in {folder}")
    else:
        images = [Path(t) for t in args.targets]
        missing = [str(p) for p in images if not p.is_file()]
        if missing:
            raise Fail("no such file: " + ", ".join(missing))

    # Decide what may be written BEFORE spending minutes of GPU on it.
    todo, skipped, claimed = [], [], {}
    for image in images:
        dests = [image.with_suffix(f".{f}") for f in formats]
        clash = next((claimed[d] for d in dests if d in claimed), None)
        if clash:
            skipped.append((image, f"same output name as {clash.name}"))
            continue
        existing = [d for d in dests if d.exists()]
        if existing and not args.overwrite:
            skipped.append((image, f"{', '.join(d.name for d in existing)} already exists"))
            continue
        for d in dests:
            claimed[d] = image
        todo.append(image)

    if skipped and not batch:
        for image, why in skipped:
            print(f"refusing {image}: {why}", file=sys.stderr)
        print("Nothing was generated. Pass --overwrite to replace existing models.",
              file=sys.stderr)
        return 2
    for image, why in skipped:
        print(f"skip  {image.name}: {why}")
    if not todo:
        print("Nothing to do. Pass --overwrite to replace existing models.")
        return 0

    h = ensure_up()
    print(f"{len(todo)} image(s) on {h.get('gpu_name', 'GPU')}, "
          f"{args.resolution}^3 -> {', '.join(formats)}. A few minutes each.", flush=True)

    failed = 0
    for i, image in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {image.name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            written = convert(image, formats, args.resolution)
        except Fail as exc:
            failed += 1
            print(f"FAILED after {time.time() - t0:.0f}s\n    {exc}", flush=True)
            continue
        print(f"{time.time() - t0:.0f}s", flush=True)
        for w in written:
            print(f"    wrote {w.resolve()}  ({w.stat().st_size / 1e6:.1f} MB)", flush=True)

    print(f"\n{len(todo) - failed} converted, {failed} failed, {len(skipped)} skipped.")
    return 1 if failed else 0


def cmd_up(args) -> int:
    h = ensure_up()
    app = f"{URL}/app"
    print(f"Geppetto is up on {h.get('gpu_name', 'GPU')}: {app}")
    if not args.no_browser:
        webbrowser.open(app)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="geppetto", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="start the server and open the web UI")
    up.add_argument("--no-browser", action="store_true", help="do not open a browser")
    up.set_defaults(func=cmd_up)

    pr = sub.add_parser("process", help="convert image(s) to 3D models")
    pr.add_argument("targets", nargs="+", metavar="FILE | all [DIR]",
                    help="image file(s), or `all` plus an optional directory")
    pr.add_argument("--overwrite", action="store_true",
                    help="replace models that already exist")
    pr.add_argument("--format", action="append", choices=FORMATS,
                    help="output format, repeatable (default: stl)")
    pr.add_argument("--resolution", type=int, choices=[512, 1024, 1536], default=1024,
                    help="detail level (default: 1024)")
    pr.set_defaults(func=cmd_process)

    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):     # piped output on Windows is cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        return args.func(args)
    except Fail as exc:
        print(f"geppetto: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
