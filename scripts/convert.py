#!/usr/bin/env python3
"""Batch 2D -> 3D conversion against the TRELLIS.2 inference API.

Examples
--------
    # One image, fast preview quality
    python scripts/convert.py inputs/chair.png --resolution 512

    # A whole folder at full detail, writing GLBs next to a summary table
    python scripts/convert.py inputs/ --resolution 1536 --out outputs/batch-01

    # Pre-cut RGBA cutouts (skips background removal entirely)
    python scripts/convert.py inputs/cutouts/ --no-preprocess
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trellis_client import ApiError, TrellisClient, collect_images, human_size  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert images to 3D via the TRELLIS.2 API.")
    ap.add_argument("target", type=Path, help="Image file or a directory of images.")
    ap.add_argument("--url", default="http://localhost:8080", help="API base URL.")
    ap.add_argument("--out", type=Path, default=Path("outputs/converted"),
                    help="Where to write the GLBs.")
    ap.add_argument("--resolution", type=int, choices=[512, 1024, 1536], default=1024)
    ap.add_argument("--seed", type=int, help="Fixed seed (default: random per image).")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="Input is already a cut-out RGBA; skip background removal.")
    ap.add_argument("--preview", action="store_true",
                    help="Also render a turntable MP4 (uses nvdiffrast/nvdiffrec: non-commercial).")
    ap.add_argument("--export-backend", choices=["open", "nvdiffrast"],
                    help="Texture bake: open (default on the server) or nvdiffrast (non-commercial).")
    ap.add_argument("--matting", choices=["birefnet", "rmbg2"],
                    help="Background removal: birefnet (MIT, server default) or rmbg2 (non-commercial).")
    ap.add_argument("--texture-size", type=int, choices=[1024, 2048, 4096], default=2048)
    ap.add_argument("--decimation", type=int, default=500_000, help="Target face count.")
    ap.add_argument("--format", dest="formats", action="append",
                    choices=["stl", "obj", "ply"], default=[],
                    help="Also write this format next to the GLB (repeatable). "
                         "STL is geometry only — use it for 3D printing.")
    ap.add_argument("--steps", type=int, help="Override sampling steps for all three stages.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not re-convert images whose GLB already exists.")
    args = ap.parse_args()

    if not args.target.exists():
        print(f"No such path: {args.target}", file=sys.stderr)
        return 2

    images = collect_images(args.target)
    if not images:
        print(f"No images found under {args.target}", file=sys.stderr)
        return 2

    params: dict = {
        "resolution": args.resolution,
        "preprocess_image": not args.no_preprocess,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation,
        "preview_video": args.preview,
        "export_formats": args.formats,
    }
    if args.export_backend:
        params["export_backend"] = args.export_backend
    if args.matting:
        params["matting"] = args.matting
    if args.seed is not None:
        params["seed"] = args.seed
        params["randomize_seed"] = False
    if args.steps is not None:
        for stage in ("sparse_structure", "shape", "texture"):
            params[stage] = {"steps": args.steps}

    client = TrellisClient(args.url)
    try:
        health = client.wait_until_ready()
    except ApiError as exc:
        print(f"API not ready: {exc}", file=sys.stderr)
        return 1

    print(f"Connected to {args.url} — {health.get('gpu_name')} "
          f"({health.get('gpu_free_gb')} GB free)")
    print(f"{len(images)} image(s), {args.resolution}^3, "
          f"texture {args.texture_size}, decimate to {args.decimation:,}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    for i, image in enumerate(images, 1):
        glb_dest = args.out / f"{image.stem}.glb"
        if args.skip_existing and glb_dest.exists():
            print(f"[{i}/{len(images)}] {image.name} — skipped (exists)")
            continue

        print(f"[{i}/{len(images)}] {image.name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            job_id = client.submit(image, params)
            job = client.wait(job_id)
        except ApiError as exc:
            print(f"FAILED\n    {exc}")
            results.append({"image": image.name, "status": "error", "error": str(exc)})
            continue

        elapsed = time.time() - t0
        if job["status"] != "completed":
            print(f"FAILED after {elapsed:.0f}s\n    {job.get('error')}")
            results.append({"image": image.name, "status": "failed",
                            "error": job.get("error"), "seconds": round(elapsed, 1)})
            continue

        client.download(job_id, "asset.glb", glb_dest)
        for fmt in args.formats:
            if job["artifacts"].get(fmt):
                client.download(job_id, f"asset.{fmt}", args.out / f"{image.stem}.{fmt}")
        if args.preview and job["artifacts"].get("preview_video"):
            client.download(job_id, "preview.mp4", args.out / f"{image.stem}.mp4")

        print(f"{elapsed:5.0f}s  {human_size(job.get('glb_bytes')):>7}  "
              f"seed={job.get('effective_seed')}  peak {job.get('peak_vram_gb')} GB")
        for w in job.get("warnings", []):
            print(f"    warning: {w}")

        results.append({
            "image": image.name, "status": "completed", "job_id": job_id,
            "seconds": round(elapsed, 1), "timings": job.get("timings"),
            "glb_bytes": job.get("glb_bytes"), "faces": job.get("face_count"),
            "seed": job.get("effective_seed"), "peak_vram_gb": job.get("peak_vram_gb"),
            "output": str(glb_dest),
        })

    summary = args.out / "conversion-summary.json"
    summary.write_text(json.dumps({"params": params, "results": results}, indent=2))

    ok = sum(1 for r in results if r["status"] == "completed")
    bad = len(results) - ok
    if ok:
        avg = sum(r["seconds"] for r in results if r["status"] == "completed") / ok
        print(f"\n{ok} converted (avg {avg:.0f}s), {bad} failed -> {args.out}")
    else:
        print(f"\n0 converted, {bad} failed -> {summary}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
