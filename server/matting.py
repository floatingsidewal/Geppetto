"""Background removal ("matting"), loaded only when a job needs it.

Upstream's `pipeline.json` names `briaai/RMBG-2.0` (CC BY-NC 4.0) and
`Trellis2ImageTo3DPipeline.from_pretrained()` builds it eagerly, so the weights
load at server start even when every request sends a pre-cut RGBA image.

`install()` replaces `trellis2.pipelines.rembg.BiRefNet` (the class
`from_pretrained()` looks up by name) with `LazyMatting`, which:

* ignores the model name in `pipeline.json` and picks one from `MATTING_MODELS`
  per call: the job's `matting` choice, else `GEPPETTO_MATTING` (default
  `birefnet`, i.e. ZhengPeng7/BiRefNet, MIT);
* loads nothing until the first image that actually needs its background
  removed (upstream skips removal for inputs that already have transparency,
  and never calls it with `preprocess_image: false`);
* keeps each loaded model, so switching back and forth costs one load each.

RMBG-2.0 shares BiRefNet's architecture, so both run through upstream's own
`BiRefNet` wrapper class with a different weights repo.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("geppetto.matting")

# Choice -> Hugging Face weights repo. Licences: docs/LICENSES-MODELS.md.
MATTING_MODELS: dict[str, str] = {
    "birefnet": "ZhengPeng7/BiRefNet",  # MIT
    "rmbg2": "briaai/RMBG-2.0",  # CC BY-NC 4.0: non-commercial
}
NONCOMMERCIAL_MATTING = {"rmbg2"}


def default_matting() -> str:
    value = os.environ.get("GEPPETTO_MATTING", "birefnet").strip().lower()
    if value not in MATTING_MODELS:
        log.warning("GEPPETTO_MATTING=%r is not one of %s; using birefnet", value, list(MATTING_MODELS))
        return "birefnet"
    return value


class LazyMatting:
    """Drop-in for upstream's `BiRefNet` class that defers the weights load."""

    def __init__(self, model_name: str | None = None) -> None:
        # The name from pipeline.json is deliberately ignored: it is RMBG-2.0.
        self._upstream_name = model_name
        self._models: dict[str, object] = {}
        self._device = "cpu"
        self._lock = threading.Lock()
        # Set by the worker before each job; None means the server default.
        self.choice: str | None = None
        # The choice that ran since the worker last reset it (per job).
        self.last_used: str | None = None

    # --- the interface upstream calls ------------------------------------
    def to(self, device) -> None:
        self._device = device
        for model in self._models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to("cuda")

    def cpu(self) -> None:
        self.to("cpu")

    def __call__(self, image):
        choice = self.choice or default_matting()
        model = self._get(choice)
        model.to(self._device)
        self.last_used = choice
        return model(image)

    # --- internals --------------------------------------------------------
    def _get(self, choice: str):
        with self._lock:
            if choice not in self._models:
                from trellis2.pipelines.rembg.BiRefNet import BiRefNet as UpstreamBiRefNet

                repo = MATTING_MODELS[choice]
                log.info("Loading matting model %s (%s) on first use", choice, repo)
                self._models[choice] = UpstreamBiRefNet(model_name=repo)
            return self._models[choice]

    @property
    def loaded(self) -> list[str]:
        return sorted(self._models)


def install() -> None:
    """Make `from_pretrained()` build a LazyMatting instead of RMBG-2.0.

    Must run after trellis2 is importable and before `from_pretrained()`.
    """
    from trellis2.pipelines import rembg

    rembg.BiRefNet = LazyMatting
