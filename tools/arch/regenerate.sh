#!/bin/bash
# Regenerate architecture diagrams from code annotations
# Run this after code changes to update the 3D visualization

set -e

cd /home/melchior/speech3
source venv/bin/activate

echo "🔄 Regenerating architecture data..."
python tools/arch/generate_architecture.py

echo "✅ Done! View at: https://api.speakfit.app/static/architecture.html"
echo "   JSON data: https://api.speakfit.app/static/architecture-data.json"
