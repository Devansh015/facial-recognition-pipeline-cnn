#!/usr/bin/env bash
set -euo pipefail

# Download and extract dlib's 68-point facial landmark predictor into data/models
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data/models"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FILE="shape_predictor_68_face_landmarks.dat.bz2"

echo "Downloading $URL ..."
if command -v curl >/dev/null 2>&1; then
  curl -L -o "$FILE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$FILE" "$URL"
else
  echo "Error: neither curl nor wget is installed." >&2
  exit 1
fi

echo "Extracting $FILE ..."
bzip2 -df "$FILE"

ls -lh "$DATA_DIR/shape_predictor_68_face_landmarks.dat" || true

echo "Done. File saved to: $DATA_DIR/shape_predictor_68_face_landmarks.dat"
