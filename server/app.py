"""Geppetto — FastAPI front end for TRELLIS.2 image -> 3D inference.

Submit an image, get a job id, poll it, download the GLB. All GPU work is
serialized behind a single worker thread (see worker.py).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from schemas import GenerateParams, Health, Job, JobCreated, JobStatus
from worker import MODEL_ID, OUTPUT_DIR, engine

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("geppetto.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.shutdown()


app = FastAPI(
    title="Geppetto",
    version="1.0.0",
    description=(
        "A picture goes in, a 3D model comes out. Single image -> textured GLB "
        "(plus STL/OBJ/PLY), powered by microsoft/TRELLIS.2-4B."
    ),
    lifespan=lifespan,
)


# The single-page app is designed to work two ways: served from this API at
# /app (same origin), and opened straight off disk as a file:// page, which the
# browser labels origin "null". Supporting the second means permissive CORS.
#
# That is a deliberate trade for a GPU box on a trusted LAN: this service has no
# auth, so anything that can reach the port can already drive it — CORS is not
# what is protecting it. If Geppetto is ever exposed beyond a trusted network,
# put it behind a reverse proxy with auth and set GEPPETTO_CORS_ORIGINS to the
# specific origins you serve the UI from.
_cors = os.environ.get("GEPPETTO_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# An upload is read fully into memory before it is decoded, and this service has
# no authentication in front of it, so an unbounded body is a trivial way to
# exhaust RAM. Input images are photographs — tens of megabytes is already
# generous. Raise it if you genuinely feed it very large scans.
MAX_UPLOAD_BYTES = int(os.environ.get("GEPPETTO_MAX_UPLOAD_MB", "64")) * 1024 * 1024


def _gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"torch_version": torch.__version__}
        free, total = torch.cuda.mem_get_info()
        return {
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(total / 1024**3, 1),
            "gpu_free_gb": round(free / 1024**3, 1),
        }
    except Exception:  # noqa: BLE001
        return {}


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "geppetto",
        "model": MODEL_ID,
        "app": "/app",
        "docs": "/docs",
        "health": "/healthz",
        "submit": "POST /v1/generate",
    }


@app.get("/app", include_in_schema=False)
def web_app() -> HTMLResponse:
    """The single-page UI. Same origin as the API, so no CORS involved.

    The identical file also runs from disk — copy server/static/index.html
    anywhere and open it; it will ask which server to talk to.
    """
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "UI not bundled in this image.")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    stats = engine.stats()
    if engine.load_error:
        status = "unhealthy"
    elif engine.loaded.is_set():
        status = "healthy"
    else:
        status = "starting"
    return Health(
        status=status,
        model=MODEL_ID,
        model_loaded=engine.loaded.is_set(),
        load_error=engine.load_error,
        **_gpu_info(),
        **stats,
    )


@app.post("/v1/generate", response_model=JobCreated, status_code=202)
async def generate(
    request: Request,
    image: UploadFile = File(..., description="Input image (PNG/JPEG/WebP; RGBA respected)."),
    params: str | None = Form(
        None,
        description='JSON object of generation parameters, e.g. {"resolution":512,"preview_video":true}',
    ),
) -> JobCreated:
    if engine.load_error:
        raise HTTPException(503, f"Model failed to load: {engine.load_error}")

    try:
        parsed = GenerateParams.model_validate(json.loads(params)) if params else GenerateParams()
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"`params` is not valid JSON: {exc}") from exc
    except Exception as exc:  # pydantic ValidationError
        raise HTTPException(422, f"Invalid parameters: {exc}") from exc

    # Check the declared length first so an oversized body is rejected before it
    # is buffered, then check what actually arrived — Content-Length is a claim,
    # not a guarantee, and chunked uploads do not send one at all.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Upload exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Upload is {len(raw) // 1024 // 1024} MB; the limit is "
                 f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )

    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "Could not decode the uploaded file as an image.") from exc
    except Exception as exc:  # noqa: BLE001
        # A broken/missing format plugin raises whatever it likes (a bad Pillow
        # build made WebP throw AttributeError on import). That is still a bad
        # upload from the caller's side, not a server fault — say which file
        # and why, with a 400, instead of an opaque 500.
        raise HTTPException(
            400, f"Could not decode {image.filename or 'the upload'}: {type(exc).__name__}: {exc}"
        ) from exc

    # Normalize to RGBA PNG on disk: the pipeline's preprocessor expects to be
    # able to read an alpha channel when preprocess_image is disabled.
    img = img.convert("RGBA")

    job_id = uuid.uuid4().hex[:12]
    workdir = OUTPUT_DIR / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    img.save(workdir / "input.png")

    job = engine.submit(job_id, parsed, image.filename)
    log.info(
        "Queued job %s (%s, %s^3, seed=%s)",
        job_id,
        image.filename,
        parsed.resolution,
        "random" if parsed.randomize_seed else parsed.seed,
    )
    return JobCreated(
        job_id=job_id,
        status=job.status,
        queue_position=job.queue_position or 0,
        poll=f"/v1/jobs/{job_id}",
    )


@app.get("/v1/jobs", response_model=list[Job])
def list_jobs(limit: int = 100) -> list[Job]:
    return engine.list_jobs(limit=limit)


@app.get("/v1/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    job = engine.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return job


@app.delete("/v1/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    if not engine.forget(job_id):
        raise HTTPException(404, "No such job.")
    return JSONResponse({"deleted": job_id})


def _artifact(job_id: str, filename: str, media_type: str) -> FileResponse:
    job = engine.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        detail = job.error or f"Job is {job.status.value}; {filename} is not available."
        raise HTTPException(404, detail)
    return FileResponse(path, media_type=media_type, filename=f"{job_id}-{filename}")


@app.get("/v1/jobs/{job_id}/asset.glb")
def get_glb(job_id: str) -> FileResponse:
    return _artifact(job_id, "asset.glb", "model/gltf-binary")


_MESH_MEDIA_TYPES = {
    "stl": "model/stl",
    "obj": "model/obj",
    "ply": "application/octet-stream",
}


@app.get("/v1/jobs/{job_id}/asset.{ext}")
def get_mesh(job_id: str, ext: str) -> FileResponse:
    """STL/OBJ/PLY, when the job asked for them via `export_formats`."""
    media_type = _MESH_MEDIA_TYPES.get(ext.lower())
    if media_type is None:
        raise HTTPException(404, f"Unsupported mesh format: {ext}")
    return _artifact(job_id, f"asset.{ext.lower()}", media_type)


@app.get("/v1/jobs/{job_id}/preview.mp4")
def get_preview(job_id: str) -> FileResponse:
    return _artifact(job_id, "preview.mp4", "video/mp4")


@app.get("/v1/jobs/{job_id}/input.png")
def get_input(job_id: str) -> FileResponse:
    return _artifact(job_id, "input.png", "image/png")
