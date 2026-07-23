#!/usr/bin/env python
"""Strip vendored / build-output symbols from a Purpose Tool B index.

`purpose index` writes .purpose/index.json as {"root": ..., "symbols": [
    {"name", "kind", "file", "line", "snippet"}, ...]}.
Its built-in skip list misses this repo's committed Cargo registry at
buhera-os/.cargo-home/registry/ (~50k dependency symbols). This filter removes
those and other non-framework paths so `purpose ask` ranks over real code/prose.

Usage:
    python purpose_filter_index.py [path/to/index.json]
Defaults to .purpose/index.json relative to the current directory.
Writes a one-time backup of the raw index to <index>.raw.bak (never overwritten).
"""
import json
import os
import shutil
import sys

# Path fragments (forward-slash normalized) that mark a symbol as noise.
BAD_FRAGMENTS = (
    ".cargo-home",      # committed vendored Cargo registry (the big one)
    "node_modules",
    "vendor/pylon",     # vendored sub-package
    ".next/",           # Next.js build output
    "/dist/",
    "/build/",
    "site-packages",    # any stray Python deps
    ".venv/",
)


def is_noise(sym):
    f = sym.get("file", "").replace("\\", "/")
    # a path segment literally named `target` (Rust/Java build dir)
    if "/target/" in f or f.endswith("/target"):
        return True
    return any(frag in f for frag in BAD_FRAGMENTS)


def main():
    index_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(".purpose", "index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"index not found: {index_path}")

    data = json.load(open(index_path, encoding="utf-8"))
    symbols = data.get("symbols", [])
    total = len(symbols)

    backup = index_path + ".raw.bak"
    if not os.path.exists(backup):
        shutil.copy(index_path, backup)  # preserve the first raw index only

    kept = [s for s in symbols if not is_noise(s)]
    data["symbols"] = kept
    json.dump(data, open(index_path, "w", encoding="utf-8"), ensure_ascii=False)

    dropped = total - len(kept)
    print(f"[purpose-filter] kept {len(kept)} / {total} symbols (dropped {dropped} vendored/build)")


if __name__ == "__main__":
    main()
