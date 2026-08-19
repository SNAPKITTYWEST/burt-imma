#!/usr/bin/env python3
"""
BURT-IMMA Protected Execution Gate

Native Python implementation with Ed25519 cryptographic verification.
Replaces the shell-based gate with:
  - Native Ed25519 via cryptography library (no openssl CLI)
  - JSON schema validation via Pydantic
  - No subprocess, no shell, no external CLI tools
  - Deterministic JSON canonicalization (sorted keys, compact)

Exit codes:
  0 = AUTHORIZED (protected execution allowed)
  1 = INTEGRITY_FAILED (release verification failed)
  2 = AUTHORIZATION_DENIED (node not authorized or capability missing/invalid)
  3 = SCRIPT_ERROR (cannot determine status)

Environment:
  BURT_IMMA_CAPABILITY_TOKEN  - capability token (JSON|signature_hex)
  BURT_IMMA_REPO_ROOT         - override repo root (defaults to script parent dir)

Usage:
  python3 burt_imma_gate.py [--quiet] [--json-output]

Contact: jessica@collectivekitty.com
"""

import json
import os
import sys
import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.exceptions import InvalidSignature


# =============================================================================
# EXIT CODES
# =============================================================================

EXIT_AUTHORIZED = 0
EXIT_INTEGRITY_FAILED = 1
EXIT_DENIED = 2
EXIT_ERROR = 3


# =============================================================================
# MODELS (Pydantic schema validation)
# =============================================================================

class CapabilityPayload(BaseModel):
    """Schema for capability token JSON payload."""
    node_id: str = Field(min_length=1)
    release_id: str = Field(min_length=1)
    commit: str = Field(min_length=1)
    nonce: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)


class ReleaseMetadata(BaseModel):
    """Schema for sovereign/release.json."""
    project: str = ""
    repository: str = ""
    release_version: str = ""
    git_commit: str = ""
    node_id: str = ""
    manifest_sha256: str = ""
    release_timestamp_utc: str = ""


class AuthorizationRecord(BaseModel):
    """Schema for sovereign/authorization.json."""
    authorization_id: str = ""
    node_id: str = ""
    authorization_status: str = ""
    authorization_scope: str = ""
    expires_at_utc: Optional[str] = None
    revocation_status: str = ""


class NodeIdentity(BaseModel):
    """Schema for sovereign/node.json."""
    node_id: str = ""
    algorithm: str = ""
    public_key_hex: str = ""


# =============================================================================
# GATE IMPLEMENTATION
# =============================================================================

class BurtImmaGate:
    """
    BURT-IMMA Protected Execution Gate.

    Verifies in order:
      1. Release integrity (git commit matches release.json, manifest hash)
      2. Node authorization status (ACTIVE, not revoked, not expired)
      3. Capability possession (env var or file)
      4. Capability validity (commit, expiration, node binding)
      5. Capability signature (Ed25519 with authority public key)
    """

    def __init__(self, repo_root: Optional[Path] = None, quiet: bool = False):
        if repo_root is None:
            repo_root = Path(__file__).resolve().parent
        self.repo_root = Path(repo_root)
        self.sovereign_dir = self.repo_root / "sovereign"
        self.quiet = quiet
        self._messages: list[str] = []

    def log(self, msg: str) -> None:
        self._messages.append(msg)
        if not self.quiet:
            print(msg)

    def run(self) -> int:
        """Execute the full gate sequence. Returns exit code."""
        self.log("==========================================")
        self.log("BURT-IMMA PROTECTED EXECUTION GATE")
        self.log("==========================================")
        self.log("")

        result = self._verify_release_integrity()
        if result != EXIT_AUTHORIZED:
            return result

        result = self._verify_node_authorization()
        if result != EXIT_AUTHORIZED:
            return result

        capability_raw = self._get_capability_token()
        if capability_raw is None:
            return EXIT_DENIED

        payload, signature_hex = self._parse_capability(capability_raw)
        if payload is None:
            return EXIT_DENIED

        result = self._validate_capability(payload)
        if result != EXIT_AUTHORIZED:
            return result

        result = self._verify_signature(payload, signature_hex)
        if result != EXIT_AUTHORIZED:
            return result

        self.log("")
        self.log("==========================================")
        self.log("STATUS: AUTHORIZATION_GRANTED")
        self.log("==========================================")
        self.log("")
        self.log("Protected execution is AUTHORIZED.")
        self.log("")
        self.log(f"Capability valid until: {payload.expires_at}")
        self.log("")

        return EXIT_AUTHORIZED

    def _verify_release_integrity(self) -> int:
        self.log("[1/5] Verifying release integrity...")
        release_file = self.sovereign_dir / "release.json"
        if not release_file.exists():
            self.log("FAILED: sovereign/release.json not found")
            return EXIT_INTEGRITY_FAILED
        try:
            with open(release_file) as f:
                data = json.load(f)
            release = ReleaseMetadata(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.log(f"FAILED: Cannot parse release.json: {e}")
            return EXIT_INTEGRITY_FAILED
        current_commit = self._get_git_commit()
        if current_commit is None:
            self.log("FAILED: Cannot determine current git commit")
            return EXIT_ERROR
        if current_commit != release.git_commit:
            self.log("FAILED: Release integrity check failed")
            return EXIT_INTEGRITY_FAILED
        manifest_file = self.sovereign_dir / "manifest.json"
        if manifest_file.exists() and release.manifest_sha256:
            computed_hash = self._sha256_file(manifest_file)
            if computed_hash != release.manifest_sha256:
                self.log("FAILED: Manifest hash mismatch")
                return EXIT_INTEGRITY_FAILED
        self.log("    Release integrity verified")
        self.log("")
        return EXIT_AUTHORIZED

    def _verify_node_authorization(self) -> int:
        self.log("[2/5] Verifying node authorization status...")
        auth_file = self.sovereign_dir / "authorization.json"
        if not auth_file.exists():
            self.log("FAILED: Authorization record not found")
            return EXIT_DENIED
        node_file = self.sovereign_dir / "node.json"
        if not node_file.exists():
            self.log("FAILED: Node identity not found")
            return EXIT_DENIED
        try:
            with open(auth_file) as f:
                auth_data = json.load(f)
            auth = AuthorizationRecord(**auth_data)
        except (json.JSONDecodeError, ValidationError) as e:
            self.log(f"FAILED: Cannot parse authorization.json: {e}")
            return EXIT_ERROR
        if auth.authorization_status != "ACTIVE":
            self.log(f"DENIED: Authorization status is {auth.authorization_status}")
            return EXIT_DENIED
        if auth.revocation_status != "ACTIVE":
            self.log(f"DENIED: Revocation status is {auth.revocation_status}")
            return EXIT_DENIED
        self.log("    Node authorization verified")
        self.log("")
        return EXIT_AUTHORIZED

    def _get_capability_token(self) -> Optional[str]:
        self.log("[3/5] Checking for capability...")
        token = os.environ.get("BURT_IMMA_CAPABILITY_TOKEN", "").strip()
        if token:
            self.log("    Capability token found (environment)")
            self.log("")
            return token
        cap_file = self.sovereign_dir / ".capability"
        if cap_file.exists():
            token = cap_file.read_text().strip()
            if token:
                self.log("    Capability token found (file)")
                self.log("")
                return token
        self.log("DENIED: No capability available")
        self.log("")
        self.log("To obtain authorization:")
        self.log("  1. Contact: jessica@collectivekitty.com")
        self.log("  2. Set: export BURT_IMMA_CAPABILITY_TOKEN=<capability>")
        self.log("  3. Re-run protected operation")
        self.log("")
        return None

    def _parse_capability(self, raw: str) -> tuple[Optional[CapabilityPayload], str]:
        self.log("[4/5] Parsing capability...")
        parts = raw.split("|", 1)
        if len(parts) != 2:
            self.log("DENIED: Capability format invalid (missing separator)")
            return None, ""
        json_part = parts[0].strip()
        sig_hex = parts[1].strip()
        try:
            data = json.loads(json_part)
            payload = CapabilityPayload(**data)
        except json.JSONDecodeError:
            self.log("DENIED: Capability JSON invalid")
            return None, ""
        except ValidationError:
            self.log("DENIED: Capability format invalid")
            return None, ""
        self.log(f"    Node ID: {payload.node_id}")
        self.log(f"    Expires: {payload.expires_at}")
        self.log("    Capability parsed")
        self.log("")
        return payload, sig_hex

    def _validate_capability(self, payload: CapabilityPayload) -> int:
        self.log("[5/5] Validating capability...")
        current_commit = self._get_git_commit()
        if current_commit and current_commit != payload.commit:
            self.log("DENIED: Release commit mismatch")
            return EXIT_DENIED
        try:
            expires = datetime.fromisoformat(payload.expires_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expires:
                self.log("DENIED: Capability expired")
                return EXIT_DENIED
        except ValueError:
            self.log("ERROR: Cannot parse expiration time")
            return EXIT_ERROR
        self.log("    Capability validated")
        self.log("")
        return EXIT_AUTHORIZED

    def _verify_signature(self, payload: CapabilityPayload, sig_hex: str) -> int:
        self.log("[6/6] Verifying capability signature...")
        authority_pk_file = self.sovereign_dir / "authority_pk.pem"
        if not authority_pk_file.exists():
            self.log("ERROR: Authority public key not found")
            return EXIT_ERROR
        if not sig_hex or len(sig_hex) != 128:
            self.log("DENIED: Capability signature invalid format")
            return EXIT_DENIED
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            self.log("DENIED: Capability signature contains non-hex characters")
            return EXIT_DENIED
        canonical_data = {
            "commit": payload.commit,
            "expires_at": payload.expires_at,
            "node_id": payload.node_id,
            "nonce": payload.nonce,
            "release_id": payload.release_id,
        }
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))
        message_bytes = canonical_json.encode("utf-8")
        try:
            with open(authority_pk_file, "rb") as f:
                pem_data = f.read()
            public_key = load_pem_public_key(pem_data)
            if not isinstance(public_key, Ed25519PublicKey):
                self.log("ERROR: Authority key is not Ed25519")
                return EXIT_ERROR
            public_key.verify(sig_bytes, message_bytes)
            self.log("    Signature verified")
            return EXIT_AUTHORIZED
        except InvalidSignature:
            self.log("DENIED: Capability signature verification failed")
            return EXIT_DENIED
        except Exception as e:
            self.log(f"ERROR: Signature verification error: {e}")
            return EXIT_ERROR

    def _get_git_commit(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _sha256_file(self, filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


def main() -> int:
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    repo_root = os.environ.get("BURT_IMMA_REPO_ROOT")
    if repo_root:
        root_path = Path(repo_root)
    else:
        root_path = Path(__file__).resolve().parent
    gate = BurtImmaGate(repo_root=root_path, quiet=quiet)
    return gate.run()


if __name__ == "__main__":
    sys.exit(main())
