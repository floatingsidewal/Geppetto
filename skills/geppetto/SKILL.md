---
name: geppetto
description: Turn a picture into a 3D model (GLB for viewing, STL for printing) using a Geppetto inference server. Use when the user says "turn this into a 3D model", "make this printable", "image to 3D", "generate a mesh from this photo", "make an STL of this", or points at an image and asks for something to print.
---

# Geppetto — picture to 3D model

Converts a single image into a textured 3D asset via a Geppetto server
(TRELLIS.2-4B behind a REST API). One picture in, one mesh out.

## Server

Default: `http://localhost:8080`. Override with `GEPPETTO_URL`.

**Always check health first** — the model takes ~90s to load after a restart,
and submitting before it is ready fails:

```bash
curl -fsS "${GEPPETTO_URL:-http://localhost:8080}/healthz"
```

`model_loaded: true` means go. `false` with a null `load_error` means still
loading — wait and re-check rather than reporting failure. A non-null
`load_error`, or no response at all, means the server needs attention: report
that plainly instead of retrying forever.

## Converting

The repo's client handles submit-poll-download in one step. Prefer it:

```bash
python3 scripts/convert.py <image> --format stl --out <output-dir>
```

Add `--url "$GEPPETTO_URL"` when the server is not on localhost.

Without the repo checked out, use the API directly:

```bash
# submit -> returns {"job_id": "..."}
curl -X POST "$GEPPETTO_URL/v1/generate" \
  -F image=@photo.png \
  -F 'params={"resolution":1024,"export_formats":["stl"]}'

# poll until status is "completed" or "failed"
curl -fsS "$GEPPETTO_URL/v1/jobs/<job_id>"

# download
curl -o model.glb "$GEPPETTO_URL/v1/jobs/<job_id>/asset.glb"
curl -o model.stl "$GEPPETTO_URL/v1/jobs/<job_id>/asset.stl"
```

## Choosing formats

Ask what the model is *for* when it is not obvious — it changes the answer:

- **Printing / slicing** → STL. Geometry only, no colour. This is what a slicer
  wants.
- **Viewing, sharing, game engines** → GLB. The only format that carries the
  generated PBR texture.
- **Colour geometry without a texture map** → OBJ or PLY (vertex colours).

Requesting several costs seconds, not another generation — they are converted
from the same mesh. When in doubt, produce GLB + STL.

## Choosing detail

| resolution | use | time (RTX 6000 Ada) | peak VRAM |
|---|---|---|---|
| 512 | quick draft, checking a composition | ~60s | ~4 GB |
| 1024 | the default; right for almost everything | 110–190s | 4.5–6.2 GB |
| 1536 | maximum detail, pair with `texture_size: 4096` | ~300s | 11.3 GB |

Generation is minutes, not seconds. Say so before starting a long one rather
than leaving the user watching a silent prompt.

## Input images

Any PNG/JPEG/WebP. A single, clearly-lit subject works best. Background removal
runs automatically; pass `"preprocess_image": false` for an already-cut-out RGBA
image.

Only the first image of a multi-image request is used — there is no batching in
one job. For several images, loop, or point `convert.py` at a directory.

## Reporting results

State where the files went and what they are. Include the honest caveats:

- **Meshes are not watertight.** For anything structural or load-bearing, say a
  repair pass in the slicer may be needed. Do not promise print-readiness.
- STL has no colour. If the user seemed to want the texture, point them at the
  GLB as well.
- The seed is recorded in the job record. Mention it when a user likes a result
  — it is how they reproduce or vary it.

## When it fails

- **HTTP 400** — the upload was not decodable. Check the file is really an image.
- **HTTP 503** — the model is not loaded; check `/healthz`.
- **Job status `failed`** — read the job's `error` field and report it verbatim.
  Do not silently retry: generation is expensive, and a real failure repeats.
