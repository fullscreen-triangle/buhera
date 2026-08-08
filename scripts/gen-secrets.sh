#!/usr/bin/env bash
# gen-secrets.sh — generate Buhera's runtime secrets into a local .env file.
#
# Run this ONCE, ON THE VM that will run Buhera. It creates a .env holding fresh
# random credentials. .env is .gitignore'd; it must never be committed, emailed,
# or pasted into chat. If it leaks, rotate every value and restart the stack.
#
# This script does NOT generate SSH keys. SSH keys are generated with
# `ssh-keygen` on the machine that will hold the private half — see
# long-grass/docs/buhera-integration/server-integration.md §2.
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=".env"

if [[ -e "$ENV_FILE" ]]; then
  echo "refusing to overwrite existing $ENV_FILE" >&2
  echo "back it up and remove it first if you really want fresh secrets." >&2
  exit 1
fi

# 24 random bytes, URL-safe, no shell-hostile characters
rand() { openssl rand -base64 24 | tr -d '\n' | tr '+/' '-_'; }

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl not found; install it (apt-get install -y openssl)" >&2
  exit 1
fi

umask 077   # the file we create is readable only by us

cat > "$ENV_FILE" <<EOF
# Buhera runtime secrets — generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by gen-secrets.sh
# DO NOT COMMIT. DO NOT SHARE. chmod 600. Rotate all values if this file leaks.

# --- database ---
POSTGRES_PASSWORD=$(rand)

# --- cache ---
REDIS_PASSWORD=$(rand)

# --- dashboards ---
GRAFANA_ADMIN_PASSWORD=$(rand)

# --- application API token (for authenticating callers to the buhera API) ---
BUHERA_API_TOKEN=$(rand)

# --- outbound / companion services: fill these in by hand ---
# Ask the author for the real URLs/keys and paste them below.
# PURPOSE_URL=
# ZANGALEWA_URL=
# INTERCEPTOR_URL=
# EXTERNAL_API_KEY=
EOF

chmod 600 "$ENV_FILE"
echo "wrote $ENV_FILE (chmod 600)."
echo "next: open it, fill in any companion-service URLs, then bring the stack up."
echo "reminder: this file is secret. it must never leave this machine unencrypted."
