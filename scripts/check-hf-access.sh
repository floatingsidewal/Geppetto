#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check-hf-access.sh — has Hugging Face granted this token access to the two
# gated models TRELLIS.2 needs?
#
# Needs nothing but curl — run it before the image is built, and re-run it
# while waiting on Meta's manual review of DINOv3.
#
#   bash scripts/check-hf-access.sh                 # reads HF_TOKEN from .env
#   HF_TOKEN=hf_xxx bash scripts/check-hf-access.sh # or pass it inline
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.."

GREEN=$'\033[1;32m'; RED=$'\033[1;31m'; YELLOW=$'\033[1;33m'; DIM=$'\033[2m'; OFF=$'\033[0m'

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "${RED}No HF_TOKEN.${OFF} Create a read token at https://huggingface.co/settings/tokens"
  echo "then either put it in .env or run: HF_TOKEN=hf_xxx bash scripts/check-hf-access.sh"
  exit 2
fi

whoami_json=$(curl -sf -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/api/whoami-v2) || {
  echo "${RED}Token rejected by Hugging Face.${OFF} Check it at https://huggingface.co/settings/tokens"
  exit 1
}
user=$(printf '%s' "$whoami_json" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')
echo "Authenticated as ${GREEN}${user:-unknown}${OFF}"
echo

# repo | probe file | note
TARGETS=(
  "facebook/dinov3-vitl16-pretrain-lvd1689m|config.json|image conditioning encoder — MANUAL approval by Meta, required"
  "briaai/RMBG-2.0|config.json|background removal — auto approval, optional (skip with preprocess_image=false)"
  "microsoft/TRELLIS.2-4B|pipeline.json|the 4B generator — ungated"
  "microsoft/TRELLIS-image-large|README.md|sparse-structure decoder from TRELLIS v1 — ungated"
)

blocked=0
for entry in "${TARGETS[@]}"; do
  IFS='|' read -r repo probe note <<<"$entry"
  code=$(curl -s -o /dev/null -w '%{http_code}' -L \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/${repo}/resolve/main/${probe}")

  case "$code" in
    200)     status="${GREEN}GRANTED${OFF}" ;;
    401|403) status="${RED}BLOCKED${OFF}"; blocked=$((blocked+1)) ;;
    404)     status="${YELLOW}404 (repo or probe file moved)${OFF}" ;;
    *)       status="${YELLOW}HTTP ${code}${OFF}" ;;
  esac

  printf '%-46s %s\n' "$repo" "$status"
  printf '%s%s%s\n' "$DIM" "    $note" "$OFF"
  [[ "$code" =~ ^(401|403)$ ]] && printf '    accept at: https://huggingface.co/%s\n' "$repo"
done

echo
if (( blocked == 0 )); then
  echo "${GREEN}All required weights are reachable.${OFF} Next: bash scripts/fetch-models.sh"
else
  echo "${YELLOW}${blocked} repo(s) still blocked.${OFF}"
  echo "Track pending requests at https://huggingface.co/settings/gated-repos"
  echo "${DIM}DINOv3 is reviewed by a human at Meta — hours to days is normal.${OFF}"
  exit 1
fi
