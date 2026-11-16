#!/usr/bin/env bash
# Run the test suite inside the built image, on one GPU, without touching the
# running service. The working tree's server/ and tests/ are streamed into a
# throwaway container (tar on stdin) and put ahead of the image's own copy.
# Not a bind mount: under WSL2, a bind mount of a /mnt/c checkout can show
# recently rewritten files as missing inside the container.
#
#   bash scripts/test.sh                       # image geppetto:${GEPPETTO_TARGET:-nvidia-cuda}
#   GPU_ID=1 bash scripts/test.sh -k raster    # pick a GPU, pass pytest args
#   GEPPETTO_TEST_GLB=outputs/<job>/asset.glb bash scripts/test.sh
#                                              # also compare on a real UV layout
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${GEPPETTO_IMAGE:-geppetto:${GEPPETTO_TARGET:-nvidia-cuda}}"
FILES=(server tests)
if [ -n "${GEPPETTO_TEST_GLB:-}" ]; then FILES+=("$GEPPETTO_TEST_GLB"); fi
tar -c --exclude=__pycache__ "${FILES[@]}" | docker run --rm -i --gpus all \
  -e CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
  -e GEPPETTO_TEST_GLB="${GEPPETTO_TEST_GLB:+/repo/$GEPPETTO_TEST_GLB}" \
  -e PYTHONPATH=/opt/TRELLIS.2:/repo/server \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" bash -c 'mkdir -p /repo && tar -x -C /repo \
    && pip install -q pytest >/dev/null 2>&1 \
    && python -m pytest -q -p no:cacheprovider /repo/tests "$@"' _ "$@"
