"""
RSA-2048 key management for InferenceToken miners.

Keys are used to encrypt prompts (requester → miner) and outputs
(miner → requester) so neither party can read the other's data on-chain.

Storage:
  ~/.inference-miner/keys/private.pem   (kept secret — never log or transmit)
  ~/.inference-miner/keys/public.der    (posted on-chain in registerMiner())
"""

import os
import logging
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

log = logging.getLogger("miner.keys")

DEFAULT_KEY_DIR = Path.home() / ".inference-miner" / "keys"
PRIV_FILE = "private.pem"
PUB_FILE  = "public.der"


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate RSA-2048 keypair. Returns (private_key_pem, public_key_der)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_der


def load_or_generate(key_dir: Path = DEFAULT_KEY_DIR) -> tuple[bytes, bytes]:
    """
    Load keys from disk, or generate and persist new ones if absent.
    Returns (private_key_pem, public_key_der).
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path = key_dir / PRIV_FILE
    pub_path  = key_dir / PUB_FILE

    if priv_path.exists() and pub_path.exists():
        priv_pem = priv_path.read_bytes()
        pub_der  = pub_path.read_bytes()
        log.info("Loaded RSA keypair from %s", key_dir)
    else:
        log.info("Generating new RSA-2048 keypair at %s ...", key_dir)
        priv_pem, pub_der = generate_keypair()
        # Secure permissions before writing
        priv_path.write_bytes(priv_pem)
        priv_path.chmod(0o600)
        pub_path.write_bytes(pub_der)
        log.info("RSA keypair generated and saved.")

    return priv_pem, pub_der


def decrypt(private_key_pem: bytes, ciphertext: bytes) -> bytes:
    """Decrypt RSA-OAEP+SHA-256 ciphertext with a PEM private key."""
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    return private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def encrypt(public_key_der: bytes, plaintext: bytes) -> bytes:
    """Encrypt plaintext with RSA-OAEP+SHA-256 using a DER public key."""
    public_key = serialization.load_der_public_key(public_key_der)
    return public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def pubkey_hex(public_key_der: bytes) -> str:
    """Return the DER public key as a hex string (for on-chain registration)."""
    return "0x" + public_key_der.hex()
