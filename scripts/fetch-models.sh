#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch-models.sh — verify Hugging Face access and pre-download every weight
# the pipeline needs into the `trellis2-models` volume.
#
#   bash scripts/fetch-models.sh
#
# Run once. After this the container starts offline-clean.
# Total download: ~20 GB.
#
#   bash scripts/fetch-models.sh --with-rmbg2   # also fetch briaai/RMBG-2.0
#
# The default background remover is ZhengPeng7/BiRefNet (MIT), fetched here.
# briaai/RMBG-2.0 (CC BY-NC 4.0, non-commercial) is only needed if you opt in
# to it with GEPPETTO_MATTING=rmbg2 or `"matting": "rmbg2"`.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

WITH_RMBG2=0
for arg in "$@"; do
  case "$arg" in
    --with-rmbg2) WITH_RMBG2=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f .env ]] || die "No .env found. Run: cp .env.example .env   then set HF_TOKEN."
# shellcheck disable=SC1091
set -a; . ./.env; set +a

[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is empty in .env. See the comments in .env.example."

# Any Geppetto image can do the downloading — it only needs huggingface_hub —
# so this works the same on either build target.
IMAGE="${GEPPETTO_IMAGE:-geppetto:nvidia-cuda}"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "Image $IMAGE not found. Build it first: bash scripts/build.sh <target>
(or set GEPPETTO_IMAGE to a published one, e.g. ghcr.io/floatingsidewal/geppetto:nvidia-cuda-latest)"

log "Downloading weights into the trellis2-models volume"

# -i is load-bearing: the program below arrives on stdin, and without it
# `python -` reads EOF, runs nothing, and exits 0 — a silent no-download.
docker run --rm -i \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=0 \
  -e WITH_RMBG2="$WITH_RMBG2" \
  -v trellis2-models:/models \
  "$IMAGE" \
  python - <<'PY'
import os
import sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

api = HfApi()
try:
    who = api.whoami()
    print(f"[hf] authenticated as {who.get('name')}")
except Exception as exc:
    sys.exit(f"[hf] token rejected: {exc}")

# (repo_id, allow_patterns, why, gate_url)
TARGETS = [
    ("microsoft/TRELLIS.2-4B", None,
     "the 4B generator itself (~16 GB)", None),
    ("microsoft/TRELLIS-image-large", ["ckpts/ss_dec_conv3d_16l8_fp16*"],
     "sparse-structure decoder — pipeline.json pulls this one file from TRELLIS v1", None),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", None,
     "image conditioning encoder — GATED, manual approval by Meta",
     "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"),
    ("ZhengPeng7/BiRefNet", None,
     "background removal (default matting) — MIT, not gated", None),
]
if os.environ.get("WITH_RMBG2") == "1":
    TARGETS.append(
        ("briaai/RMBG-2.0", None,
         "background removal, opt-in — GATED, auto approval, CC BY-NC 4.0 (non-commercial)",
         "https://huggingface.co/briaai/RMBG-2.0"))

failed = []
for repo_id, allow, why, gate_url in TARGETS:
    print(f"\n[hf] {repo_id}\n     {why}")
    try:
        snapshot_download(repo_id=repo_id, allow_patterns=allow, resume_download=True)
        print(f"[ok] {repo_id}")
    except GatedRepoError:
        failed.append((repo_id, f"ACCESS NOT GRANTED — accept the licence at {gate_url}"))
        print(f"[GATED] {repo_id}: access not granted to this token")
    except HfHubHTTPError as exc:
        code = getattr(exc.response, "status_code", "?")
        hint = f"accept the licence at {gate_url}" if gate_url else str(exc)
        failed.append((repo_id, f"HTTP {code} — {hint}"))
        print(f"[FAIL] {repo_id}: HTTP {code}")
    except Exception as exc:
        failed.append((repo_id, str(exc)))
        print(f"[FAIL] {repo_id}: {exc}")

if failed:
    print("\n" + "=" * 70)
    print("Some weights could not be downloaded:\n")
    for repo_id, reason in failed:
        print(f"  - {repo_id}\n      {reason}\n")
    print("Note: facebook/dinov3-... uses MANUAL approval — Meta reviews the")
    print("request, which can take hours to days. The pipeline cannot run")
    print("without it. briaai/RMBG-2.0 is auto-approved and only needed if")
    print("you opt in to it (GEPPETTO_MATTING=rmbg2).")
    print("=" * 70)
    sys.exit(1)

print("\n[done] All weights present in /models.")
PY

log "Weights cached. Next: docker compose up -d"
