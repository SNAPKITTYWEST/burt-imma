#!/bin/bash
# Sovereign Node Identity Generator for BURT-IMMA
#
# Creates a node IDENTITY REQUEST (not an authorized credential).
# Contact: jessica@collectivekitty.com

set -e

SOVEREIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SOVEREIGN_DIR")"

NODE_ID="burt-imma-$(date +%s)"
CREATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(cd "$REPO_ROOT" && git rev-parse HEAD)

echo "[*] Generating Sovereign Node Key for BURT-IMMA"
echo "    Node ID: $NODE_ID"
echo "    Created: $CREATED_AT"

openssl genpkey -algorithm Ed25519 -out "$SOVEREIGN_DIR/.node_sk" 2>/dev/null
openssl pkey -in "$SOVEREIGN_DIR/.node_sk" -pubout -out "$SOVEREIGN_DIR/node_pk.pem" 2>/dev/null

PUB_KEY_HEX=$(openssl pkey -in "$SOVEREIGN_DIR/node_pk.pem" -pubin -outform DER 2>/dev/null | xxd -p | tr -d '\n')

cat > "$SOVEREIGN_DIR/node.json" <<EOF
{
  "node_id": "$NODE_ID",
  "algorithm": "Ed25519",
  "public_key_hex": "$PUB_KEY_HEX",
  "created_at_utc": "$CREATED_AT",
  "repository": "SNAPKITTYWEST/burt-imma",
  "git_commit": "$GIT_COMMIT",
  "version": "0.1.0"
}
EOF

chmod 400 "$SOVEREIGN_DIR/.node_sk"
echo "[*] Node key generated. NEVER commit .node_sk"

exit 0
