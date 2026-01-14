#!/usr/bin/env bash
# build_prime_double_feature.sh
#
# Concats:
#   1) bit_bucket_prime_towers_4k.mp4
#   2) PrimeBucketDensityPlusAndDecay.mp4
# and lays *continuous* music across the whole thing (Number*.wav),
# but NEVER extends past the combined video length.
# If music is shorter than the videos, it will end and the remainder will be silence.
#
# Usage:
#   chmod +x build_prime_double_feature.sh
#   ./build_prime_double_feature.sh
#
# Optional overrides:
#   MUSIC_DIR="/path/to/wavs" OUT="output.mp4" ./build_prime_double_feature.sh

set -euo pipefail

# --------- CONFIG (edit if you want) ----------
V1="/home/onojk123/Downloads/bit_bucket_prime_towers_4k.mp4"
V2="/home/onojk123/projects/geom_factor/media/videos/prime_bucket_density_plus_and_decay/1080p60/PrimeBucketDensityPlusAndDecay.mp4"

MUSIC_DIR="${MUSIC_DIR:-/home/onojk123/Downloads/Solar Circuit Mechanism}"
OUT="${OUT:-BitBucket_Towers_then_Density_SolarCircuitMechanism.mp4}"
# ---------------------------------------------

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: Missing dependency: $1" >&2; exit 1; }; }
need ffmpeg
need ffprobe
need python3

[[ -f "$V1" ]] || { echo "ERROR: Video 1 not found: $V1" >&2; exit 1; }
[[ -f "$V2" ]] || { echo "ERROR: Video 2 not found: $V2" >&2; exit 1; }
[[ -d "$MUSIC_DIR" ]] || { echo "ERROR: MUSIC_DIR not found: $MUSIC_DIR" >&2; exit 1; }

# Collect wavs (natural-ish sort)
mapfile -t WAVS < <(ls -1 "$MUSIC_DIR"/Number*.wav 2>/dev/null | sort -V || true)
if [[ ${#WAVS[@]} -eq 0 ]]; then
  echo "ERROR: No Number*.wav files found in: $MUSIC_DIR" >&2
  exit 1
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "==> Using videos:"
echo "    1) $V1"
echo "    2) $V2"
echo "==> Using music dir:"
echo "    $MUSIC_DIR"
echo "==> Tracks found: ${#WAVS[@]}"

# Make an audio concat list
AUDIO_LIST="$TMP/audio.txt"
: > "$AUDIO_LIST"
for f in "${WAVS[@]}"; do
  printf "file '%s'\n" "$f" >> "$AUDIO_LIST"
done

# Combine all WAVs into one stream (re-encode to consistent format)
AUDIO_COMBINED="$TMP/combined_audio.wav"
echo "==> Concatenating audio..."
ffmpeg -hide_banner -loglevel error -y \
  -f concat -safe 0 -i "$AUDIO_LIST" \
  -ar 48000 -ac 2 \
  "$AUDIO_COMBINED"

# Get target size/fps from V1 so both videos match for concatenation
read -r W H FPS_FRAC < <(
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,avg_frame_rate \
    -of default=nk=1:nw=1 "$V1" | awk 'NR==1{w=$0} NR==2{h=$0} NR==3{fps=$0} END{print w, h, fps}'
)

# Compute FPS float (ffmpeg fps filter accepts fractions too, but we normalize to float)
FPS_FLOAT="$(python3 - <<PY
from fractions import Fraction
fps = Fraction("$FPS_FRAC")
print(float(fps))
PY
)"

# Compute total duration = dur(V1)+dur(V2)
D1="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$V1")"
D2="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$V2")"
TOTAL_DUR="$(python3 - <<PY
d1=float("$D1"); d2=float("$D2")
print(d1+d2)
PY
)"

echo "==> Target render:"
echo "    Size: ${W}x${H}"
echo "    FPS : $FPS_FLOAT (from $FPS_FRAC)"
echo "    Total video duration (s): $TOTAL_DUR"
echo "==> Building final MP4: $OUT"

# Final render:
# - Scale/fps both videos to match V1
# - Concat video streams
# - Audio: apad (so it can outlast itself) then atrim to EXACT total video duration
ffmpeg -hide_banner -y \
  -i "$V1" \
  -i "$V2" \
  -i "$AUDIO_COMBINED" \
  -filter_complex "\
[0:v]scale=${W}:${H}:flags=lanczos,fps=${FPS_FLOAT},setsar=1[v0]; \
[1:v]scale=${W}:${H}:flags=lanczos,fps=${FPS_FLOAT},setsar=1[v1]; \
[v0][v1]concat=n=2:v=1:a=0[v]; \
[2:a]apad,atrim=0:${TOTAL_DUR},asetpts=N/SR/TB[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 320k \
  -movflags +faststart \
  "$OUT"

echo "==> Done."
echo "Output: $OUT"

# Quick verify streams + duration
echo "==> ffprobe summary:"
ffprobe -v error -show_entries format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate \
  -of default=nk=1:nw=1 "$OUT"
