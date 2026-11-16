"""Model lifecycle and the single-threaded job runner.

Diffusion inference is not safely concurrent on one device, so exactly one
worker thread owns the GPU and drains a FIFO queue. The model is loaded once,
in the background, so `/healthz` answers immediately during the multi-minute
first load.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import licence_guard
import matting
from schemas import GenerateParams, Job, JobArtifacts, JobStatus

log = logging.getLogger("geppetto.worker")

# Before anything imports trellis2 (and through it o_voxel): o_voxel imports
# nvdiffrast at module level, so without this the model load alone would load
# a non-commercial component. See licence_guard.py.
licence_guard.install()

MODEL_ID = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
OUTPUT_DIR = Path(os.environ.get("TRELLIS2_OUTPUT_DIR", "/outputs"))
ENVMAP_PATH = os.environ.get("TRELLIS2_ENVMAP", "/opt/TRELLIS.2/assets/hdri/forest.exr")

# nvdiffrast cannot index more than 2^24 vertices; upstream applies this exact
# cap before export and so must we. Kept on the open backend too, so both
# backends export the same mesh (and preview renders still use nvdiffrast).
NVDIFFRAST_VERTEX_LIMIT = 16_777_216

EXPORT_BACKENDS = ("open", "nvdiffrast")


def default_export_backend() -> str:
    value = os.environ.get("GEPPETTO_EXPORT_BACKEND", "open").strip().lower()
    if value not in EXPORT_BACKENDS:
        log.warning("GEPPETTO_EXPORT_BACKEND=%r is not one of %s; using open", value, EXPORT_BACKENDS)
        return "open"
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Engine:
    """Owns the pipeline, the queue, and the worker thread."""

    def __init__(self) -> None:
        self.pipeline = None
        self.envmap = None
        self.load_error: str | None = None
        self.loaded = threading.Event()
        self.started_at = time.time()

        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running_job: str | None = None
        self._completed = 0
        self._failed = 0
        self._stop = threading.Event()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._restore_jobs()

    # ---------------------------------------------------------------- startup

    def start(self) -> None:
        threading.Thread(target=self._load_model, name="model-loader", daemon=True).start()
        threading.Thread(target=self._worker_loop, name="job-worker", daemon=True).start()

    def shutdown(self) -> None:
        self._stop.set()

    def _load_model(self) -> None:
        t0 = time.time()
        try:
            log.info("Loading %s (this takes several minutes on a cold cache)", MODEL_ID)
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            # pipeline.json names briaai/RMBG-2.0 and from_pretrained() would
            # load it now. Swap in a matting stub that loads on first use.
            matting.install()
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
            pipeline.cuda()
            self.pipeline = pipeline
            log.info("Pipeline ready in %.1fs", time.time() - t0)

            # The env map is only needed for optional preview renders; a failure
            # here must not take the service down.
            try:
                import cv2
                import torch
                from trellis2.renderers import EnvMap

                raw = cv2.imread(ENVMAP_PATH, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise FileNotFoundError(ENVMAP_PATH)
                self.envmap = EnvMap(
                    torch.tensor(
                        cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Env map unavailable, preview renders disabled: %s", exc)

            self.loaded.set()
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("Model load failed: %s\n%s", self.load_error, traceback.format_exc())

    # ------------------------------------------------------------------- jobs

    def submit(self, job_id: str, params: GenerateParams, source_filename: str | None) -> Job:
        job = Job(
            id=job_id,
            status=JobStatus.queued,
            params=params,
            source_filename=source_filename,
            created_at=_now(),
            queue_position=self._queue.qsize() + 1,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def forget(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        import shutil

        shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
        return True

    def stats(self) -> dict:
        rembg = getattr(self.pipeline, "rembg_model", None)
        matting_loaded = list(getattr(rembg, "loaded", []))
        noncommercial = licence_guard.noncommercial_loaded()
        noncommercial += [
            matting.MATTING_MODELS[m] for m in matting_loaded if m in matting.NONCOMMERCIAL_MATTING
        ]
        return {
            "queue_depth": self._queue.qsize(),
            "running_job": self._running_job,
            "jobs_completed": self._completed,
            "jobs_failed": self._failed,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "export_backend_default": default_export_backend(),
            "matting_default": matting.default_matting(),
            "matting_loaded": matting_loaded,
            "noncommercial_loaded": noncommercial,
        }

    def _persist(self, job: Job) -> None:
        """Write job state next to its artifacts so it survives a restart."""
        try:
            d = OUTPUT_DIR / job.id
            d.mkdir(parents=True, exist_ok=True)
            (d / "job.json").write_text(job.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist job %s: %s", job.id, exc)

    def _restore_jobs(self) -> None:
        for meta in sorted(OUTPUT_DIR.glob("*/job.json")):
            try:
                job = Job.model_validate_json(meta.read_text())
                # Anything mid-flight when we died is not coming back.
                if job.status in (JobStatus.queued, JobStatus.running):
                    job.status = JobStatus.failed
                    job.error = "Service restarted while this job was in flight."
                self._jobs[job.id] = job
            except Exception:  # noqa: BLE001
                continue
        if self._jobs:
            log.info("Restored %d job records from %s", len(self._jobs), OUTPUT_DIR)

    # ----------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            job = self.get(job_id)
            if job is None:
                continue

            # Jobs may be queued before the model finishes loading; hold here.
            while not self.loaded.is_set() and self.load_error is None:
                time.sleep(2.0)
            if self.load_error is not None:
                job.status = JobStatus.failed
                job.error = f"Model failed to load: {self.load_error}"
                job.finished_at = _now()
                self._failed += 1
                self._persist(job)
                continue

            self._running_job = job_id
            try:
                self._run(job)
                job.status = JobStatus.completed
                self._completed += 1
            except Exception as exc:  # noqa: BLE001
                job.status = JobStatus.failed
                job.error = f"{type(exc).__name__}: {exc}"
                self._failed += 1
                log.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            finally:
                job.finished_at = _now()
                job.queue_position = None
                self._running_job = None
                self._persist(job)

    def _run(self, job: Job) -> None:
        import torch
        from PIL import Image

        p = job.params
        workdir = OUTPUT_DIR / job.id
        workdir.mkdir(parents=True, exist_ok=True)

        job.status = JobStatus.running
        job.started_at = _now()
        job.effective_seed = random.randint(0, 2_147_483_647) if p.randomize_seed else p.seed
        self._persist(job)

        image = Image.open(workdir / "input.png")
        job.artifacts.input_image = f"/v1/jobs/{job.id}/input.png"

        torch.cuda.reset_peak_memory_stats()

        backend = p.export_backend or default_export_backend()
        job.export_backend = backend
        rembg = self.pipeline.rembg_model
        if isinstance(rembg, matting.LazyMatting):
            rembg.choice = p.matting
            rembg.last_used = None

        # --- Generation ---
        t0 = time.time()
        outputs = self.pipeline.run(
            image,
            seed=job.effective_seed,
            preprocess_image=p.preprocess_image,
            pipeline_type=p.pipeline_type,
            max_num_tokens=p.max_num_tokens,
            **p.sampler_kwargs(),
        )
        mesh = outputs[0]
        job.timings["generate"] = round(time.time() - t0, 2)
        used = getattr(rembg, "last_used", None)
        if used:
            job.matting_model = matting.MATTING_MODELS[used]
            if used in matting.NONCOMMERCIAL_MATTING:
                job.noncommercial_components.append(f"{job.matting_model} (background removal)")

        # --- Export ---
        t0 = time.time()
        if backend == "open":
            from open_export import to_glb
        else:
            from o_voxel.postprocess import to_glb

            job.noncommercial_components.append("nvdiffrast (texture bake)")

        mesh.simplify(NVDIFFRAST_VERTEX_LIMIT)
        glb = to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=p.decimation_target,
            texture_size=p.texture_size,
            remesh=p.remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb_path = workdir / "asset.glb"
        glb.export(str(glb_path), extension_webp=True)
        job.timings["export"] = round(time.time() - t0, 2)
        job.artifacts.glb = f"/v1/jobs/{job.id}/asset.glb"
        job.glb_bytes = glb_path.stat().st_size

        # Extra formats are converted from the in-memory scene. STL/OBJ/PLY
        # cannot carry the PBR material, so the texture is lost by definition —
        # say so rather than letting it surprise someone downstream.
        for fmt in p.export_formats:
            try:
                out_path = workdir / f"asset.{fmt}"
                glb.export(str(out_path))
                setattr(job.artifacts, fmt, f"/v1/jobs/{job.id}/asset.{fmt}")
            except Exception as exc:  # noqa: BLE001
                job.warnings.append(f"{fmt.upper()} export failed: {exc}")
        if "stl" in p.export_formats:
            job.warnings.append("STL carries geometry only — the texture is in the GLB.")
        try:
            job.face_count = int(mesh.faces.shape[0])
        except Exception:  # noqa: BLE001
            pass

        # --- Optional preview render ---
        if p.preview_video:
            if self.envmap is None:
                job.warnings.append("Preview requested but the env map failed to load.")
            else:
                # Recorded before the attempt: a failed render still ran it.
                job.noncommercial_components.append("nvdiffrast + nvdiffrec (preview render)")
                try:
                    t0 = time.time()
                    import imageio
                    from trellis2.utils import render_utils

                    frames = render_utils.make_pbr_vis_frames(
                        render_utils.render_video(mesh, envmap=self.envmap)
                    )
                    preview = workdir / "preview.mp4"
                    imageio.mimsave(str(preview), frames, fps=15)
                    job.timings["preview"] = round(time.time() - t0, 2)
                    job.artifacts.preview_video = f"/v1/jobs/{job.id}/preview.mp4"
                except Exception as exc:  # noqa: BLE001
                    # A renderer failure must not lose a perfectly good mesh.
                    job.warnings.append(f"Preview render failed: {type(exc).__name__}: {exc}")

        job.peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        job.timings["total"] = round(sum(v for k, v in job.timings.items() if k != "total"), 2)

        del mesh, outputs
        torch.cuda.empty_cache()


engine = Engine()
