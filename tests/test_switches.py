"""Licence switches: nothing non-commercial loads unless a job asks for it.

Each check runs in a fresh interpreter, because "was this module ever loaded"
is process-wide state. Run inside the image: `bash scripts/test.sh`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("trellis2")


def _run(code: str, tmp_path, **env) -> str:
    full_env = {**os.environ, "TRELLIS2_OUTPUT_DIR": str(tmp_path), **env}
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=full_env, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()[-1]


def test_model_code_imports_without_loading_nvdiffrast(tmp_path):
    # The import chain the model load takes: pipeline -> VAE -> o_voxel ->
    # o_voxel.postprocess -> `import nvdiffrast.torch`.
    last = _run("""
        import sys
        import worker, licence_guard
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import trellis2.models.sc_vaes.fdg_vae
        import o_voxel.postprocess, open_export
        assert "o_voxel.postprocess" in sys.modules
        print(licence_guard.noncommercial_loaded())
    """, tmp_path)
    assert last == "[]"


def test_nvdiffrast_loads_on_first_use(tmp_path):
    last = _run("""
        import worker, licence_guard
        import o_voxel.postprocess as pp
        pp.dr.RasterizeCudaContext  # attribute access is what loads it
        print(licence_guard.noncommercial_loaded())
    """, tmp_path)
    assert last == "['nvdiffrast']"


def test_matting_is_lazy_and_ignores_pipeline_json(tmp_path):
    last = _run("""
        import worker, matting
        matting.install()
        from trellis2.pipelines import rembg
        assert rembg.BiRefNet is matting.LazyMatting
        m = rembg.BiRefNet(model_name="briaai/RMBG-2.0")  # what pipeline.json asks for
        m.to("cuda"); m.cpu()
        print(m.loaded, matting.default_matting())
    """, tmp_path)
    assert last == "[] birefnet"


@pytest.mark.parametrize("value,expected", [("", "birefnet"), ("rmbg2", "rmbg2"), ("bogus", "birefnet")])
def test_matting_default_from_env(tmp_path, value, expected):
    last = _run("import matting; print(matting.default_matting())", tmp_path, GEPPETTO_MATTING=value or "birefnet")
    assert last == expected


@pytest.mark.parametrize("value,expected", [("open", "open"), ("NVDIFFRAST", "nvdiffrast"), ("x", "open")])
def test_export_backend_default_from_env(tmp_path, value, expected):
    last = _run("import worker; print(worker.default_export_backend())", tmp_path, GEPPETTO_EXPORT_BACKEND=value)
    assert last == expected


def test_params_are_backward_compatible():
    from pydantic import ValidationError

    from schemas import GenerateParams, Job

    p = GenerateParams()
    assert p.export_backend is None and p.matting is None
    assert GenerateParams.model_validate({"export_backend": "nvdiffrast", "matting": "rmbg2"}).matting == "rmbg2"
    with pytest.raises(ValidationError):
        GenerateParams.model_validate({"export_backend": "kaolin"})
    # A job.json written before these fields existed still loads.
    old = {"id": "abc", "status": "completed", "params": {}, "created_at": "2026-01-01T00:00:00+00:00"}
    job = Job.model_validate(old)
    assert job.noncommercial_components == [] and job.export_backend is None
