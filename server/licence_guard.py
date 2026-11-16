"""Keep non-commercial components out of the process unless a job asks for one.

Two components in the TRELLIS.2 stack are licensed for non-commercial use only
(see docs/LICENSES-MODELS.md):

* **nvdiffrast** (NVIDIA Source Code License). `o_voxel/__init__.py` imports
  `o_voxel.postprocess`, which does `import nvdiffrast.torch as dr` at module
  level, and the pipeline's VAE imports `o_voxel`. So merely loading the model
  would load nvdiffrast, whether or not anything ever calls it.
* **briaai/RMBG-2.0** (CC BY-NC 4.0). `pipeline.json` names it, and upstream's
  `from_pretrained()` builds it eagerly. That is handled in `matting.py`.

`install()` puts a placeholder for `nvdiffrast.torch` in sys.modules before
anything imports trellis2. An `import nvdiffrast.torch` then binds the
placeholder; nvdiffrast's own code (and its CUDA extension) only runs on the
first real attribute access, which happens only on the `nvdiffrast` export
backend or a `preview_video` render. `noncommercial_loaded()` reports what
actually ran, so a job can record it and `/healthz` can show it.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types

log = logging.getLogger("geppetto.licence")

# Modules deferred until first use. Finding the submodule's spec imports the
# parent package normally; nvdiffrast/__init__.py only reads its version string.
# Everything that does work (ops, the CUDA/GL plugin) sits under .torch.
_DEFERRED = ("nvdiffrast.torch",)

# Non-commercial modules we watch for, keyed by the name we report.
_WATCHED = {
    "nvdiffrast": "nvdiffrast.torch",
    "nvdiffrec": "nvdiffrec_render",
}

_installed = False


class _Deferred(types.ModuleType):
    """Placeholder for a module whose code has not run yet.

    It carries the real module's spec, file and path, so code that scans
    sys.modules (`inspect.getmodule()`, which torch calls whenever a custom op
    is registered) does not wake it. Any other attribute that is not there
    loads the real module, copies its namespace in, and returns the attribute,
    so references taken earlier (`import nvdiffrast.torch as dr`) keep working.
    `importlib.util.LazyLoader` was tried first: it loads on *any* attribute
    access, including `__file__`, so torch's scan loaded nvdiffrast anyway.
    """

    def __init__(self, spec) -> None:
        super().__init__(spec.name)
        self.__spec__ = spec
        self.__loader__ = spec.loader
        self.__file__ = spec.origin
        self.__package__ = spec.parent if spec.submodule_search_locations is None else spec.name
        if spec.submodule_search_locations is not None:
            self.__path__ = list(spec.submodule_search_locations)

    def __getattr__(self, attr: str):
        # Only called for attributes that are not set. Dunder probes
        # (`__wrapped__`, `__all__`, ...) must not trigger a load.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        name = self.__name__
        log.info("Loading %s (first use: %s)", name, attr)
        if sys.modules.get(name) is self:
            del sys.modules[name]
        real = importlib.import_module(name)
        self.__dict__.update(real.__dict__)
        return getattr(real, attr)


def install() -> bool:
    """Defer nvdiffrast's import. Safe to call more than once.

    Returns False when nvdiffrast is not installed (nothing to guard) or was
    already imported for real before this ran.
    """
    global _installed
    if _installed:
        return True
    for name in _DEFERRED:
        if name in sys.modules:
            if not isinstance(sys.modules[name], _Deferred):
                log.warning("%s was imported before the licence guard was installed", name)
                return False
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None or spec.loader is None:
            log.info("%s is not installed; nothing to defer", name)
            return False
        # Not set as an attribute on the parent package: `import a.b as c`
        # falls back to sys.modules, and the real import sets it later.
        sys.modules[name] = _Deferred(spec)
    _installed = True
    log.info("nvdiffrast deferred: it loads only if a job uses it")
    return True


def is_loaded(module_name: str) -> bool:
    """True when the module's own code has actually run in this process."""
    module = sys.modules.get(module_name)
    return module is not None and not isinstance(module, _Deferred)


def noncommercial_loaded() -> list[str]:
    """Non-commercial code components whose code has run in this process."""
    return sorted(name for name, mod in _WATCHED.items() if is_loaded(mod))
