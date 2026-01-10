#!/usr/bin/env bash
set -euo pipefail

VIDEO="/home/onojk123/projects/geom_factor/media/videos/prime_drive_intro/480p15/PrimeDriveIntro.mp4"
MUSIC_DIR="/home/onojk123/Downloads/Explodefuture2342343"
OUT="/home/onojk123/projects/geom_factor/media/videos/prime_drive_intro/480p15/PrimeDriveIntro_with_music.mp4"

# Audio encoding target
A_CODEC="aac"
A_BITRATE="192k"

# ---------- helpers ----------
dur_sec() {
  # Prints duration in seconds as a float
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1"
}

# Escape single quotes for ffmpeg concat list:  file '...'
escape_for_concat() {
  sed "s/'/'\\\\''/g"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---------- checks ----------
have_cmd ffprobe || { echo "ERROR: ffprobe not found."; exit 1; }
have_cmd ffmpeg  || { echo "ERROR: ffmpeg not found."; exit 1; }

[[ -f "$VIDEO" ]] || { echo "ERROR: video not found: $VIDEO"; exit 1; }
[[ -d "$MUSIC_DIR" ]] || { echo "ERROR: music dir not found: $MUSIC_DIR"; exit 1; }

mapfile -d '' WAVS < <(find "$MUSIC_DIR" -maxdepth 1 -type f -iname "*.wav" -print0 | sort -z)

(( ${#WAVS[@]} > 0 )) || { echo "ERROR: no .wav files found in: $MUSIC_DIR"; exit 1; }

V_DUR="$(dur_sec "$VIDEO")"
echo "Video duration: $V_DUR seconds"
echo "Music folder: $MUSIC_DIR"
echo

# ---------- pick unique tracks until we cover the video ----------
picked_list="$(mktemp)"
picked_sum="0.0"
picked_count=0

# pick in sorted filename order
for w in "${WAVS[@]}"; do
  a_dur="$(dur_sec "$w")"

  # Skip zero-length / weird files
  awk -v d="$a_dur" 'BEGIN{ if (d <= 0.01) exit 1; }' || continue

  echo "Adding: $(basename "$w")  (${a_dur}s)"
  printf "file '%s'\n" "$(printf "%s" "$w" | escape_for_concat)" >> "$picked_list"
  picked_sum="$(awk -v s="$picked_sum" -v d="$a_dur" 'BEGIN{ printf "%.6f", (s + d) }')"
  picked_count=$((picked_count + 1))

  # Stop once we have enough
  awk -v s="$picked_sum" -v v="$V_DUR" 'BEGIN{ exit (s+1e-6 >= v) ? 0 : 1 }' && break
done

echo
echo "Selected $picked_count unique tracks"
echo "Total selected duration: $picked_sum seconds"
echo

# ---------- enforce your rule ----------
if ! awk -v s="$picked_sum" -v v="$V_DUR" 'BEGIN{ exit (s+1e-6 >= v) ? 0 : 1 }'; then
  echo "ABORTING (per your rule): unique tracks do NOT cover the whole video."
  echo "Need >= $V_DUR seconds, but only got $picked_sum seconds from unique WAVs."
  echo "Add more unique tracks (or allow looping) and rerun."
  rm -f "$picked_list"
  exit 2
fi

# ---------- build a single audio track trimmed to video ----------
TMP_AUDIO="$(mktemp --suffix=.m4a)"

echo "Concatenating audio and trimming to video length..."
ffmpeg -hide_banner -y \
  -f concat -safe 0 -i "$picked_list" \
  -t "$V_DUR" \
  -c:a "$A_CODEC" -b:a "$A_BITRATE" \
  "$TMP_AUDIO"

# ---------- mux audio onto the video ----------
echo "Muxing audio onto video..."
ffmpeg -hide_banner -y \
  -i "$VIDEO" -i "$TMP_AUDIO" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy \
  -c:a "$A_CODEC" -b:a "$A_BITRATE" \
  -movflags +faststart \
  "$OUT"

echo
echo "DONE:"
echo "$OUT"

rm -f "$picked_list" "$TMP_AUDIO"
