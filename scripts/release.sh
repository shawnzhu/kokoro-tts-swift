#!/bin/bash
set -euo pipefail

# Release script for KokoroCoreML CoreML models.
#
# Exports PyTorch → CoreML (int8 quantized), converts voices to binary,
# compiles, packages, and uploads as a GitHub release.
#
# Usage:
#   ./scripts/release.sh              # tag: models-YYYY-MM-DD
#   ./scripts/release.sh --dry-run    # export + package without uploading

REPO="Jud/kokoro-coreml"
TAG="models-$(date +%Y-%m-%d)"
DRY_RUN="${1:-}"
EXPORT_DIR="models_export"
TARBALL="kokoro-models.tar.gz"

echo "=== KokoroCoreML Release: $TAG ==="

# 1. Export CoreML models (includes int8 quantization)
echo ""
echo "Step 1: Exporting CoreML models (int8 quantized)..."
PYTHONPATH=scripts .venv/bin/python scripts/export_coreml.py --output-dir "$EXPORT_DIR"

# 2. Compile to .mlmodelc
echo ""
echo "Step 2: Compiling models..."
for pkg in "$EXPORT_DIR"/*.mlpackage; do
    name=$(basename "$pkg" .mlpackage)
    echo "  Compiling $name..."
    xcrun coremlcompiler compile "$pkg" "$EXPORT_DIR"
    if [ -d "$EXPORT_DIR/$name.mlmodelc" ]; then
        echo "    ✓ $name.mlmodelc"
    else
        echo "    ✗ Compilation failed for $name"
        exit 1
    fi
done

# 3. Voice data — convert JSON to binary if needed
VOICE_DIR="$EXPORT_DIR/voices"
if [ ! -d "$VOICE_DIR" ]; then
    # Look for JSON voices in the model directory
    JSON_VOICE_DIR="$HOME/Library/Application Support/kokoro-coreml/models/kokoro/voices"
    if [ ! -d "$JSON_VOICE_DIR" ]; then
        JSON_VOICE_DIR="$HOME/Library/Application Support/kokoro-tts/models/kokoro/voices"
    fi
    if [ -d "$JSON_VOICE_DIR" ]; then
        echo ""
        echo "Step 3: Converting voice embeddings to binary..."
        PYTHONPATH=scripts .venv/bin/python -c "
from export_coreml import convert_voices_to_binary
convert_voices_to_binary('$JSON_VOICE_DIR', '$VOICE_DIR')
"
    else
        echo ""
        echo "Step 3: Voice data not found"
        echo "  Copy voice embeddings from your model directory:"
        echo "    cp -r ~/Library/Application\\ Support/kokoro-coreml/models/kokoro/voices $EXPORT_DIR/"
        exit 1
    fi
else
    # If voices dir exists but has JSON files, convert them
    if ls "$VOICE_DIR"/*.json >/dev/null 2>&1; then
        echo ""
        echo "Step 3: Converting voice embeddings to binary..."
        PYTHONPATH=scripts .venv/bin/python -c "
from export_coreml import convert_voices_to_binary
convert_voices_to_binary('$VOICE_DIR', '$VOICE_DIR')
"
        rm -f "$VOICE_DIR"/*.json
    else
        echo ""
        echo "Step 3: Voice data present ($(ls "$VOICE_DIR"/*.bin 2>/dev/null | wc -l | tr -d ' ') voices)"
    fi
fi

# 4. Package
echo ""
echo "Step 4: Packaging..."
cd "$EXPORT_DIR"
tar czf "../$TARBALL" \
    kokoro_frontend.mlmodelc \
    kokoro_backend.mlmodelc \
    voices
cd ..

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Created $TARBALL ($SIZE)"

# 5. Upload
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo ""
    echo "Dry run — skipping upload. To upload manually:"
    echo "  gh release create $TAG $TARBALL --repo $REPO --title 'Models ($TAG)'"
else
    echo ""
    echo "Step 5: Uploading to GitHub release $TAG..."
    gh release create "$TAG" "$TARBALL" \
        --repo "$REPO" \
        --title "Models ($TAG)" \
        --notes "Kokoro-82M CoreML models (int8 quantized) and voice embeddings (float32 binary)."
    echo "  ✓ https://github.com/$REPO/releases/tag/$TAG"
fi

echo ""
echo "Done."
