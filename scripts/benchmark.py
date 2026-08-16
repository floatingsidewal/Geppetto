#!/usr/bin/env python3
"""Measure real generation times on this workstation.

Replaces the extrapolated-from-H100 estimates in the README with numbers from
the actual hardware. Writes both JSON and a markdown table.

    python scripts/benchmark.py --image inputs/T.png --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trellis_client import ApiError, TrellisClient, human_size  # noqa: E402

# Published upstream reference numbers (NVIDIA H100), for context in the table.
H100_SECONDS = {512: 3.0, 1024: 17.0, 1536: 60.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark TRELLIS.2 across resolutions.")
    ap.add_argument("--image", type=Path, required=True, help="Test image.")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--repeats", type=int, default=3, help="Runs per resolution.")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[512, 1024, 1536],
                    choices=[512, 1024, 1536])
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    args = ap.parse_args()

    if not args.image.exists():
        print(f"No such image: {args.image}", file=sys.stderr)
        return 2

    client = TrellisClient(args.url)
    try:
        health = client.wait_until_ready()
    except ApiError as exc:
        print(f"API not ready: {exc}", file=sys.stderr)
        return 1

    gpu = health.get("gpu_name", "unknown GPU")
    print(f"Benchmarking on {gpu}\n"
          f"image={args.image.name}  repeats={args.repeats}  "
          f"resolutions={args.resolutions}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for res in args.resolutions:
        for n in range(1, args.repeats + 1):
            # Fixed seed per repeat index keeps runs comparable across resolutions.
            params = {"resolution": res, "seed": 1000 + n, "randomize_seed": False}
            print(f"  {res}^3 run {n}/{args.repeats} ... ", end="", flush=True)
            t0 = time.time()
            try:
                job = client.wait(client.submit(args.image, params))
            except ApiError as exc:
                print(f"error: {exc}")
                runs.append({"resolution": res, "run": n, "status": "error", "error": str(exc)})
                continue
            wall = time.time() - t0

            if job["status"] != "completed":
                print(f"failed: {job.get('error')}")
                runs.append({"resolution": res, "run": n, "status": "failed",
                             "error": job.get("error")})
                continue

            t = job.get("timings", {})
            print(f"{wall:6.1f}s  (gen {t.get('generate', 0):.1f}s, "
                  f"export {t.get('export', 0):.1f}s)  "
                  f"peak {job.get('peak_vram_gb')} GB  "
                  f"{human_size(job.get('glb_bytes'))}")
            runs.append({
                "resolution": res, "run": n, "status": "completed",
                "wall_seconds": round(wall, 2),
                "generate_seconds": t.get("generate"),
                "export_seconds": t.get("export"),
                "peak_vram_gb": job.get("peak_vram_gb"),
                "glb_bytes": job.get("glb_bytes"),
                "faces": job.get("face_count"),
            })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "gpu": gpu,
        "image": str(args.image),
        "repeats": args.repeats,
        "runs": runs,
    }
    (args.out / f"benchmark-{stamp}.json").write_text(json.dumps(payload, indent=2))

    # --- Markdown summary ---
    lines = [
        f"# TRELLIS.2 benchmark — {gpu}",
        "",
        f"`{stamp}` · image `{args.image.name}` · {args.repeats} run(s) per resolution",
        "",
        "| Resolution | Median wall | Generate | Export | Peak VRAM | GLB size | H100 ref | Ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in args.resolutions:
        good = [r for r in runs if r["resolution"] == res and r["status"] == "completed"]
        if not good:
            lines.append(f"| {res}³ | failed | - | - | - | - | "
                         f"{H100_SECONDS[res]:.0f}s | - |")
            continue
        med = statistics.median(r["wall_seconds"] for r in good)
        gen = statistics.median(r["generate_seconds"] or 0 for r in good)
        exp = statistics.median(r["export_seconds"] or 0 for r in good)
        vram = max(r["peak_vram_gb"] or 0 for r in good)
        size = statistics.median(r["glb_bytes"] or 0 for r in good)
        ref = H100_SECONDS[res]
        lines.append(
            f"| {res}³ | **{med:.1f}s** | {gen:.1f}s | {exp:.1f}s | {vram:.1f} GB | "
            f"{human_size(int(size))} | {ref:.0f}s | {med / ref:.1f}× |"
        )
    lines += ["", "H100 reference times are upstream's published figures, for scale only."]

    md = args.out / f"benchmark-{stamp}.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[4:]))
    print(f"\nWritten: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
