#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# smoke-test.sh — prove an image works on this machine's GPU before it ships.
#
#   bash scripts/smoke-test.sh geppetto:nvidia-cuda
#   bash scripts/smoke-test.sh geppetto:nvidia-cuda --full   # also generate a mesh
#
# Two levels:
#   default  imports, CUDA visibility, and the attention backend the image was
#            built for. Seconds. No weights needed.
#   --full   additionally runs a real generation end to end. Needs the
#            trellis2-models volume populated (scripts/fetch-models.sh).
#
# This exists because several failures in this stack are silent: a CUDA
# extension that never installed, or an attention backend that is absent but
# only fails at first inference. Import-and-assert turns those into red builds.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
pass() { printf '\033[1;32m[pass]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

IMAGE="${1:-geppetto:nvidia-cuda}"
FULL=false
[[ "${2:-}" == "--full" ]] && FULL=true

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "No such image: $IMAGE"

log "Level 1: extensions, CUDA, attention backend  ($IMAGE)"
docker run --rm --gpus all "$IMAGE" python - <<'PY'
import os, sys, importlib

failures = []

import torch
print(f"torch {torch.__version__}")
if not torch.cuda.is_available():
    failures.append("torch.cuda.is_available() is False — no GPU inside the container")
else:
    print(f"gpu   {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"arch  sm_{cap[0]}{cap[1]}")
    # A mismatch here means the image was built for another card and every
    # kernel will either JIT or fail. Worth catching before publish.
    built_for = os.environ.get("GEPPETTO_CUDA_ARCH", "")
    if built_for:
        want = "sm_" + built_for.replace(".", "")
        got = f"sm_{cap[0]}{cap[1]}"
        if want != got:
            failures.append(f"image built for {want} but this GPU is {got}")

# The six upstream CUDA extensions. Each is a separate build layer, and any of
# them can silently fail to install.
for mod in ("o_voxel", "cumesh", "flex_gemm", "nvdiffrast", "utils3d", "trellis2"):
    try:
        importlib.import_module(mod)
        print(f"ok    {mod}")
    except Exception as exc:
        failures.append(f"{mod} failed to import: {type(exc).__name__}: {exc}")

# Sparse attention accepts ONLY xformers/flash_attn/flash_attn_3 — there is no
# sdpa fallback there. If the backend this image declares is not importable,
# the image cannot generate anything.
backend = os.environ.get("ATTN_BACKEND", "flash_attn")
mod = {"flash_attn": "flash_attn", "flash_attn_3": "flash_attn_3", "xformers": "xformers"}.get(backend)
if mod is None:
    failures.append(f"ATTN_BACKEND={backend} is not valid for sparse attention")
else:
    try:
        m = importlib.import_module(mod)
        print(f"ok    attention backend {backend} ({getattr(m, '__version__', 'unknown')})")
    except Exception as exc:
        failures.append(f"declared ATTN_BACKEND={backend} but {mod} will not import: {exc}")

from PIL import features
if not features.check("webp"):
    failures.append("Pillow has no WebP support — WebP uploads will fail")
else:
    print("ok    pillow webp")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nlevel 1 ok")
PY
pass "extensions, CUDA and attention backend verified"

if [[ "$FULL" != true ]]; then
  echo
  echo "Skipped the generation test. Re-run with --full to convert a real image."
  exit 0
fi

log "Level 2: end-to-end generation"
docker volume inspect trellis2-models >/dev/null 2>&1 \
  || die "The trellis2-models volume is empty — run scripts/fetch-models.sh first."

[[ -f inputs/smoke.png ]] || {
  log "Fetching a test image"
  mkdir -p inputs
  curl -fsSL "https://raw.githubusercontent.com/microsoft/TRELLIS.2/75fbf0183001ed9876c8dbb35de6b68552ee08bd/assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp" \
    -o inputs/smoke.webp
  docker run --rm -v "$PWD/inputs:/i" "$IMAGE" \
    python -c "from PIL import Image; Image.open('/i/smoke.webp').save('/i/smoke.png')"
}

CID=$(docker run -d --rm --gpus all \
  -v trellis2-models:/models -v "$PWD/outputs:/outputs" \
  -p 18080:8000 "$IMAGE")
trap 'docker stop "$CID" >/dev/null 2>&1 || true' EXIT

log "Waiting for the model to load"
for _ in $(seq 1 80); do
  h=$(curl -fsS localhost:18080/healthz 2>/dev/null || true)
  printf '%s' "$h" | grep -q '"model_loaded": *true' && break
  printf '%s' "$h" | grep -q '"load_error": *"' && die "Model failed to load: $h"
  sleep 15
done
printf '%s' "$h" | grep -q '"model_loaded": *true' || die "Model did not load in time"
pass "model loaded"

log "Generating (this takes a few minutes)"
python3 scripts/convert.py inputs/smoke.png --url http://localhost:18080 \
  --format stl --resolution 512 --out outputs/smoke-test

for f in outputs/smoke-test/smoke.glb outputs/smoke-test/smoke.stl; do
  [[ -s "$f" ]] || die "Expected artifact missing or empty: $f"
done

docker run --rm -v "$PWD/outputs/smoke-test:/d" "$IMAGE" python - <<'PY'
import sys, trimesh
for name in ("smoke.glb", "smoke.stl"):
    m = trimesh.load(f"/d/{name}", force="mesh")
    if len(m.faces) == 0:
        sys.exit(f"{name} has no faces")
    print(f"ok    {name}: {len(m.vertices)} verts, {len(m.faces)} faces")
PY
pass "generated and validated a real mesh"

echo
echo "Smoke test passed — this image is fit to publish."
