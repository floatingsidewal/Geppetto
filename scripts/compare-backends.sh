#!/usr/bin/env bash
# Compare export backends (open vs nvdiffrast) and matting models (BiRefNet vs
# RMBG-2.0) on a few inputs, in a throwaway container on one GPU. Does not use
# or disturb the running service. See compare_backends.py for what it measures.
#
#   GPU_ID=1 bash scripts/compare-backends.sh outputs/compare-01 \
#       inputs/photo.jpg inputs/cutout.png:alpha inputs/flat.png:nomatte
#
# Needs the BiRefNet and RMBG-2.0 weights in the volume (fetch-models.sh
# --with-rmbg2). Results (GLBs, matting sheets, report.json) land in the
# output folder, which must not exist yet or be empty.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$1"; shift
[ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && { echo "$OUT is not empty" >&2; exit 1; }
mkdir -p "$OUT"
IMAGE="${GEPPETTO_IMAGE:-geppetto:${GEPPETTO_TARGET:-nvidia-cuda}}"
ARGS=()
for spec in "$@"; do ARGS+=("/work/in/$spec"); done
# Inputs and the working tree go in on stdin (a bind mount of a /mnt/c checkout
# can show recently written files as missing under WSL2). Results are written
# straight to the bind-mounted output folder, not streamed back on stdout: the
# CUDA base image's entrypoint prints a banner there.
tar -c --exclude=__pycache__ server scripts/compare_backends.py "${@%%:*}" \
  | docker run --rm -i --gpus all --ipc host \
      -e CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
      -e HF_HOME=/models -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      -e PYTHONPATH=/opt/TRELLIS.2:/work/in/server \
      -e OPENCV_IO_ENABLE_OPENEXR=1 \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v trellis2-models:/models \
      -v "$PWD/$OUT:/work/out" \
      "$IMAGE" bash -c 'mkdir -p /work/in && tar -x -C /work/in \
        && sed -i "s#/opt/server#/work/in/server#" /work/in/scripts/compare_backends.py \
        && python /work/in/scripts/compare_backends.py /work/out "$@"' _ "${ARGS[@]}"
echo "results in $OUT"
