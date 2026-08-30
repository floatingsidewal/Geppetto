#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch-models.sh — verify Hugging Face access and pre-download every weight
# the pipeline needs into the `trellis2-models` volume.
#
#   bash scripts/fetch-models.sh
#
# Run once. After this the container starts offline-clean.
# Total download: ~19 GB.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f .env ]] || die "No .env found. Run: cp .env.example .env   then set HF_TOKEN."
# shellcheck disable=SC1091
set -a; . ./.env; set +a

[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is empty in .env. See the comments in .env.example."

docker image inspect trellis2-infer:latest >/dev/null 2>&1 \
  || die "Image trellis2-infer:latest not found. Build it first: docker compose build"

log "Downloading weights into the trellis2-models volume"

# -i is load-bearing: the program below arrives on stdin, and without it
# `python -` reads EOF, runs nothing, and exits 0 — a silent no-download.
docker run --rm -i \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/models \
  -v trellis2-models:/models \
  trellis2-infer:latest \
  python - <<'PY'
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
    ("briaai/RMBG-2.0", None,
     "background removal — GATED, auto approval, CC BY-NC 4.0",
     "https://huggingface.co/briaai/RMBG-2.0"),
]

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
    print("without it. briaai/RMBG-2.0 is auto-approved and only needed when")
    print("preprocess_image=true; pre-cut RGBA inputs work without it.")
    print("=" * 70)
    sys.exit(1)

print("\n[done] All weights present in /models.")
PY

log "Weights cached. Next: docker compose up -d"
