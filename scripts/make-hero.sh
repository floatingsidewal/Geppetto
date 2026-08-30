#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# make-hero.sh — generate the README hero image using Geppetto itself.
#
#   bash scripts/make-hero.sh [input-image] [output-png]
#
# Runs one real conversion with a turntable preview, pulls a frame out of the
# MP4, and lays the source photo beside the rendered result. The banner is
# therefore an honest artefact of the pipeline at the pinned commit, not a
# mock-up — regenerate it whenever the output would actually change.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

REUSE=false
# Seconds into the turntable to grab. The camera orbits, so this chooses the
# viewing angle: 0 is the front, matching how the subject was photographed;
# later values swing round to the back, which rarely flatters a banner.
AT="${GEPPETTO_HERO_AT:-0}"
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --reuse) REUSE=true; shift ;;
    --at)    AT="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

IMAGE_IN="${1:-inputs/example-0a34fa.webp}"
OUT="${2:-docs/hero.png}"
URL="${GEPPETTO_URL:-http://localhost:8080}"
GEPPETTO_IMAGE="${GEPPETTO_IMAGE:-geppetto:nvidia-cuda}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f "$IMAGE_IN" ]] || die "No such input: $IMAGE_IN"
mkdir -p docs outputs/hero

log "Generating with a turntable preview (a few minutes)"
python3 scripts/convert.py "$IMAGE_IN" --url "$URL" --preview \
  --resolution 1024 --texture-size 2048 --out outputs/hero

stem="$(basename "${IMAGE_IN%.*}")"
mp4="outputs/hero/${stem}.mp4"
[[ -f "$mp4" ]] || die "No preview video produced — is nvdiffrast working in this image?"

log "Composing the banner"
# -i is load-bearing: the program below arrives on stdin, and without it
# `python -` reads EOF, runs nothing, and exits 0 — reporting success while
# producing no image at all.
docker run --rm -i \
  -v "$PWD/outputs/hero:/work" -v "$PWD/$(dirname "$IMAGE_IN"):/src:ro" -v "$PWD/docs:/docs" \
  "$GEPPETTO_IMAGE" python - "$stem" "$(basename "$IMAGE_IN")" "$(basename "$OUT")" "$AT" <<'PY'
import subprocess, sys
from PIL import Image, ImageChops

stem, src_name, out_name = sys.argv[1], sys.argv[2], sys.argv[3]
at = sys.argv[4] if len(sys.argv) > 4 else "0"

# One frame from the turntable. `--at` selects the viewing angle, since the
# camera orbits the model over the clip.
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-ss", at, "-i", f"/work/{stem}.mp4",
     "-frames:v", "1", "/work/frame.png"],
    check=True,
)

frame = Image.open("/work/frame.png").convert("RGB")

# Upstream's preview frame is a contact sheet: a large lit render and a normal
# map on the top row, then four smaller passes underneath. Only the first panel
# belongs in a banner, so take the top-left quadrant and then trim the black
# letterboxing around the model itself.
panel = frame.crop((0, 0, frame.width // 2, int(frame.height * 0.62)))
bbox = ImageChops.difference(panel, Image.new("RGB", panel.size, (0, 0, 0))).getbbox()
if bbox:
    pad = 12
    panel = panel.crop((
        max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
        min(bbox[2] + pad, panel.width), min(bbox[3] + pad, panel.height),
    ))

src = Image.open(f"/src/{src_name}").convert("RGB")
gen = panel

H = 420
def fit(im):
    w = max(int(im.width * H / im.height), 1)
    return im.resize((w, H), Image.LANCZOS)

src, gen = fit(src), fit(gen)
gap, pad = 24, 24
bg = (247, 246, 243)
canvas = Image.new("RGB", (src.width + gen.width + gap + pad * 2, H + pad * 2), bg)
canvas.paste(src, (pad, pad))
canvas.paste(gen, (pad + src.width + gap, pad))
canvas.save(f"/docs/{out_name}")
print(f"wrote docs/{out_name}  {canvas.size[0]}x{canvas.size[1]}")
PY

[[ -s "$OUT" ]] || die "Compose step reported success but $OUT is missing or empty."

log "Done: $OUT ($(stat -c%s "$OUT") bytes)"
