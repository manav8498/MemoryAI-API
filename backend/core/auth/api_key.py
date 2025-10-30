"""
API Key generation and verification utilities.
"""
import hashlib
import secrets
from typing import Tuple

from backend.core.config import settings


def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, prefix)
        - full_key: The complete API key to return to user (only shown once)
        - key_hash: SHA256 hash to store in database
        - prefix: First 7 chars for identification
    """
    # Generate random key (32 bytes = 64 hex chars)
    random_key = secrets.token_hex(32)

    # Create full key with prefix
    full_key = f"{settings.API_KEY_PREFIX}_{random_key}"

    # Create hash for storage
    key_hash = hash_api_key(full_key)

    # Extract prefix for display (e.g., "mem_sk_abc1234")
    prefix = full_key[:17] if len(full_key) >= 17 else full_key[:10]

    return full_key, key_hash, prefix


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA256.

    Args:
        api_key: Plain text API key

    Returns:
        SHA256 hash of the key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored hash.

    Args:
        provided_key: API key provided by user
        stored_hash: SHA256 hash stored in database

    Returns:
        True if key matches, False otherwise
    """
    computed_hash = hash_api_key(provided_key)
    return secrets.compare_digest(computed_hash, stored_hash)
