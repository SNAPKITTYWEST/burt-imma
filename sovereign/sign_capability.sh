#!/bin/bash
# Sign a Capability with Authority Private Key (BURT-IMMA)
#
# Usage: ./sovereign/sign_capability.sh <capability.json>
# Contact: jessica@collectivekitty.com

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTHORITY_SK="$SCRIPT_DIR/authority_sk.pem"
CAPABILITY_FILE="${1:-}"

if [ -z "$CAPABILITY_FILE" ]; then
  echo "ERROR: Usage: $0 <capability.json>" >&2
  exit 1
fi

if [ ! -f "$CAPABILITY_FILE" ]; then
  echo "ERROR: Capability file not found: $CAPABILITY_FILE" >&2
  exit 1
fi

if [ ! -f "$AUTHORITY_SK" ]; then
  echo "ERROR: Authority private key not found" >&2
  echo "HINT: Generate with: ./sovereign/generate_authority_key.sh" >&2
  exit 2
fi

TEMP_NORMALIZE="/tmp/burt-normalize-$$.py"
TEMP_MSG="/tmp/burt-msg-$$.bin"
TEMP_SIG="/tmp/burt-sig-$$.bin"

trap "rm -f '$TEMP_NORMALIZE' '$TEMP_MSG' '$TEMP_SIG'" EXIT

cat > "$TEMP_NORMALIZE" << 'PYTHON_EOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(json.dumps(data, sort_keys=True, separators=(',', ':')), end='')
PYTHON_EOF

CAPABILITY_CANONICAL=$(python3 "$TEMP_NORMALIZE" "$CAPABILITY_FILE")
echo -n "$CAPABILITY_CANONICAL" > "$TEMP_MSG"

openssl pkeyutl -sign -inkey "$AUTHORITY_SK" -in "$TEMP_MSG" -out "$TEMP_SIG" 2>/dev/null
SIGNATURE_HEX=$(xxd -p -c 256 < "$TEMP_SIG" | tr -d '\n')

echo "$CAPABILITY_CANONICAL|$SIGNATURE_HEX"

exit 0
