#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build.sh — build a Geppetto image for one target.
#
#   bash scripts/build.sh nvidia-cuda
#   bash scripts/build.sh dgx-spark
#   bash scripts/build.sh nvidia-cuda --tag ghcr.io/floatingsidewal/geppetto:nvidia-cuda-v1.0.0
#
# Target definitions live in docker/targets/<target>.args. This turns them into
# --build-arg flags so the Dockerfile stays single-source for both targets.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

TARGET="${1:-}"
shift || true

[[ -n "$TARGET" ]] || die "Usage: build.sh <target> [--tag IMAGE]   (targets: $(ls docker/targets/*.args 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.args$//' | tr '\n' ' '))"

ARGS_FILE="docker/targets/${TARGET}.args"
[[ -f "$ARGS_FILE" ]] || die "No such target: ${TARGET} (missing ${ARGS_FILE})"

TAG="geppetto:${TARGET}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Refuse to build a target for the wrong CPU architecture. The CUDA extensions
# are compiled from source against the host, so a mismatch wastes a long build
# and produces an image that cannot run anywhere.
HOST_ARCH="$(uname -m)"
case "$TARGET" in
  nvidia-cuda) WANT_ARCH="x86_64" ;;
  dgx-spark)   WANT_ARCH="aarch64" ;;
  *)           WANT_ARCH="" ;;
esac
if [[ -n "$WANT_ARCH" && "$HOST_ARCH" != "$WANT_ARCH" ]]; then
  die "Target '${TARGET}' must be built on ${WANT_ARCH}; this host is ${HOST_ARCH}.
Cross-building these CUDA extensions under emulation is not practical — build
on the target hardware (that is what the self-hosted release runners are for)."
fi

BUILD_ARGS=()
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  BUILD_ARGS+=(--build-arg "$line")
done < "$ARGS_FILE"

log "Building ${TAG}  (target=${TARGET}, host=${HOST_ARCH})"
printf '    %s\n' "${BUILD_ARGS[@]#--build-arg }"

if [[ "$TARGET" == "dgx-spark" ]]; then
  warn "The dgx-spark target is UNVERIFIED — see docker/targets/dgx-spark.args for the known risks."
fi

docker build -f docker/Dockerfile -t "$TAG" "${BUILD_ARGS[@]}" .

log "Built ${TAG}"
docker image inspect "$TAG" --format '    size: {{.Size}} bytes
    target: {{index .Config.Labels "io.geppetto.target"}}
    cuda arch: {{index .Config.Labels "io.geppetto.cuda-arch"}}'

cat <<EOF

Next: verify it on this machine's GPU
  bash scripts/smoke-test.sh ${TAG}

EOF
