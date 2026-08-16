"""Request/response models for the TRELLIS.2 inference service.

Defaults are taken verbatim from upstream `app.py`'s Gradio controls and
`pipeline.json`, so an API call with no parameters reproduces exactly what the
official demo does at its default settings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Maps the API's friendly resolution to upstream's `pipeline_type` string.
PIPELINE_TYPES: dict[int, str] = {
    512: "512",
    1024: "1024_cascade",
    1536: "1536_cascade",
}


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class SamplerParams(BaseModel):
    """One stage of the three-stage flow sampler."""

    steps: int = Field(12, ge=1, le=50)
    guidance_strength: float = Field(7.5, ge=1.0, le=10.0)
    guidance_rescale: float = Field(0.7, ge=0.0, le=1.0)
    rescale_t: float = Field(5.0, ge=1.0, le=6.0)


class GenerateParams(BaseModel):
    """Everything the caller can tune for one image -> GLB conversion."""

    model_config = {"extra": "forbid"}

    # --- Generation ---
    resolution: Literal[512, 1024, 1536] = Field(
        1024, description="Voxel resolution. 512 is fast; 1536 is the detail ceiling."
    )
    seed: int = Field(0, ge=0, le=2_147_483_647)
    randomize_seed: bool = Field(
        True, description="Ignore `seed` and draw a fresh one (the actual seed is reported back)."
    )
    preprocess_image: bool = Field(
        True,
        description=(
            "Run the built-in BiRefNet/RMBG background removal. Set false when the "
            "input is already a cut-out RGBA — this also avoids needing the gated "
            "briaai/RMBG-2.0 weights."
        ),
    )
    max_num_tokens: int = Field(49_152, ge=4_096, le=196_608)

    # --- Per-stage samplers (upstream defaults differ per stage) ---
    sparse_structure: SamplerParams = SamplerParams(
        steps=12, guidance_strength=7.5, guidance_rescale=0.7, rescale_t=5.0
    )
    shape: SamplerParams = SamplerParams(
        steps=12, guidance_strength=7.5, guidance_rescale=0.5, rescale_t=3.0
    )
    texture: SamplerParams = SamplerParams(
        steps=12, guidance_strength=1.0, guidance_rescale=0.0, rescale_t=3.0
    )

    # --- GLB export ---
    decimation_target: int = Field(
        500_000, ge=10_000, le=4_000_000, description="Target face count for the exported mesh."
    )
    texture_size: Literal[1024, 2048, 4096] = 2048
    remesh: bool = True

    # --- Extras ---
    preview_video: bool = Field(
        False, description="Also render a PBR turntable MP4 (adds significant time)."
    )

    def sampler_kwargs(self) -> dict[str, dict[str, Any]]:
        return {
            "sparse_structure_sampler_params": self.sparse_structure.model_dump(),
            "shape_slat_sampler_params": self.shape.model_dump(),
            "tex_slat_sampler_params": self.texture.model_dump(),
        }

    @property
    def pipeline_type(self) -> str:
        return PIPELINE_TYPES[self.resolution]


class JobArtifacts(BaseModel):
    glb: str | None = None
    preview_video: str | None = None
    input_image: str | None = None


class Job(BaseModel):
    id: str
    status: JobStatus
    params: GenerateParams
    effective_seed: int | None = None
    source_filename: str | None = None

    created_at: str
    started_at: str | None = None
    finished_at: str | None = None

    queue_position: int | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    peak_vram_gb: float | None = None
    face_count: int | None = None
    glb_bytes: int | None = None

    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int
    poll: str


class Health(BaseModel):
    status: Literal["starting", "healthy", "unhealthy"]
    model: str
    model_loaded: bool
    load_error: str | None = None
    gpu_name: str | None = None
    gpu_total_gb: float | None = None
    gpu_free_gb: float | None = None
    torch_version: str | None = None
    queue_depth: int
    running_job: str | None = None
    jobs_completed: int
    jobs_failed: int
    uptime_seconds: float
