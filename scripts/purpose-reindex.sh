#!/bin/bash
#
# purpose-reindex — rebuild the Purpose (Tool B) index, then strip vendored /
# build-output symbols that `purpose index` does not skip on its own.
#
# Why this exists:
#   `purpose index` scans ~20 source extensions but its built-in skip list only
#   covers node_modules / target / .git / build output. This repo commits a
#   vendored Cargo registry at buhera-os/.cargo-home/registry/ (~50k symbols of
#   clap, windows-sys, etc.). Left in, it drowns real framework symbols and every
#   `purpose ask` returns dependency noise. This wrapper removes those paths from
#   .purpose/index.json after each index build so `purpose ask` stays sharp.
#
# Usage:
#   scripts/purpose-reindex.sh          # from anywhere inside the repo
#
# Run this instead of a bare `purpose index` whenever the index is stale.

set -euo pipefail

# Resolve repo root from this script's location (scripts/ is at the repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INDEX="$REPO_ROOT/.purpose/index.json"

echo "[purpose-reindex] indexing $REPO_ROOT ..."
purpose index --root "$REPO_ROOT"

if [ ! -f "$INDEX" ]; then
  echo "[purpose-reindex] ERROR: $INDEX not produced by 'purpose index'." >&2
  exit 1
fi

echo "[purpose-reindex] filtering vendored / build-output symbols ..."
python "$SCRIPT_DIR/purpose_filter_index.py" "$INDEX"

echo "[purpose-reindex] done. Ask with:  purpose ask \"<question>\""
