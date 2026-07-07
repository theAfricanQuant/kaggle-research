#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: ./bootstrap.sh <competition-folder-name> [target-directory]"
  echo ""
  echo "Creates a new competition project folder with all kaggle-research files."
  echo ""
  echo "Examples:"
  echo "  ./bootstrap.sh house-prices-competition"
  echo "  ./bootstrap.sh zindi-food-prices"
  echo "  ./bootstrap.sh playground-jan-2021"
  echo "  ./bootstrap.sh house-prices ~/Documents/07_DataScience/competition"
  exit 1
fi

TARGET="$1"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT="${2:-$REPO_DIR}"
DEST="$PARENT/$TARGET"

if [ -d "$DEST" ]; then
  echo "Error: folder '$TARGET' already exists at $DEST"
  exit 1
fi

echo "==> Creating $DEST from kaggle-research template..."

mkdir -p "$DEST"
rsync -a --exclude='.git' --exclude='.gitignore' --exclude='bootstrap.sh' "$REPO_DIR/" "$DEST/"

cd "$DEST"

# Update pyproject.toml name to match folder
sed -i "s/name = \"kaggle-research\"/name = \"$TARGET\"/" pyproject.toml

# Remove old state (don't carry over previous runs)
rm -f state/log.json

# Initialise fresh git
git init
git add -A
git commit -m "initial: $TARGET — from kaggle-research template"

echo ""
echo "✅ Done: $TARGET created and ready."
echo ""
echo "Next steps:"
echo "  cd $TARGET"
echo "  uv sync"
echo "  uv run main.py --competition \"<competition-slug>\" --iterations 50"
echo ""
echo "(Run from anywhere: $REPO_DIR/bootstrap.sh <name> [target-dir])"
