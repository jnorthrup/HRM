#!/usr/bin/env bash
# Download a representative subset of the IETF RFC corpus (≈ 100 most‑cited RFCs)

set -euo pipefail

BASE_DIR="dataset/raw-data/rfc"
INDEX_URL="https://www.rfc-editor.org/rfc-index.txt"
TARGET_COUNT=100

mkdir -p "$BASE_DIR"

# Download index
curl -sL "$INDEX_URL" -o "$BASE_DIR/index.txt"

# Extract the first TARGET_COUNT RFC numbers (skipping the header lines)
rfc_numbers=$(sed -n '12,$p' "$BASE_DIR/index.txt" | awk '{print $1}' | head -n "$TARGET_COUNT")

# Download each RFC
for num in $rfc_numbers; do
    url="https://www.rfc-editor.org/rfc/rfc${num}.txt"
    echo "Downloading RFC $num..."
    curl -sL "$url" -o "$BASE_DIR/rfc${num}.txt"
done

echo "Download complete. ${TARGET_COUNT} RFCs stored in $BASE_DIR."