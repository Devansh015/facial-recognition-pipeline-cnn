#!/usr/bin/env bash
set -euo pipefail

# Download and extract LFW into data/lfw
mkdir -p "$(dirname "$0")/../data"
cd "$(dirname "$0")/../data"

PRIMARY_URL="http://vis-www.cs.umass.edu/lfw/lfw.tgz"
MIRROR_URL="https://ndownloader.figshare.com/files/5976015"  # Known mirror of lfw.tgz
FILENAME="lfw.tgz"

fetch() {
  local url="$1"
  echo "Downloading LFW from $url ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 3 -o "$FILENAME" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 -O "$FILENAME" "$url"
  else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
  fi
}

if ! fetch "$PRIMARY_URL"; then
  echo "Primary URL failed, trying mirror ..." >&2
  fetch "$MIRROR_URL"
fi

echo "Extracting $FILENAME ..."
tar -xzvf "$FILENAME" || { echo "Extraction failed" >&2; exit 1; }
rm -f "$FILENAME"

echo "Done. LFW is at $(pwd)/lfw"
