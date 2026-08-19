#!/bin/bash
# Generate BURT-IMMA Authority Keypair (Ed25519)
#
# The authority private key (authority_sk.pem) MUST remain secure and off-repo.
# The authority public key (authority_pk.pem) is distributed to verifiers.
#
# Contact: jessica@collectivekitty.com

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTHORITY_SK="$SCRIPT_DIR/authority_sk.pem"
AUTHORITY_PK="$SCRIPT_DIR/authority_pk.pem"

echo "Generating BURT-IMMA Authority Keypair (Ed25519)..."

if [ -f "$AUTHORITY_SK" ]; then
  echo "WARNING: authority_sk.pem already exists"
  exit 0
fi

openssl genpkey -algorithm Ed25519 -out "$AUTHORITY_SK"
openssl pkey -in "$AUTHORITY_SK" -pubout -out "$AUTHORITY_PK"

chmod 600 "$AUTHORITY_SK"
chmod 644 "$AUTHORITY_PK"

echo "Authority Keypair Generated"
echo "  Private Key: $AUTHORITY_SK (NEVER COMMIT)"
echo "  Public Key:  $AUTHORITY_PK"

exit 0
