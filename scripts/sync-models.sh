#!/usr/bin/env bash
# Refresh crates/ai/data/providers from the @earendil-works/pi-ai npm package.
# Usage: scripts/sync-models.sh [version]   (default: latest)
set -euo pipefail
version="${1:-latest}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
(cd "$work" && npm pack --silent "@earendil-works/pi-ai@${version}" >/dev/null && tar xzf ./*.tgz)
rm -f "$root"/crates/ai/data/providers/*.json
cp "$work"/package/dist/providers/data/*.json "$root"/crates/ai/data/providers/
echo "Synced $(ls "$root"/crates/ai/data/providers | wc -l) catalogs from pi-ai $(node -p "require('$work/package/package.json').version")"
