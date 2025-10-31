"""
API Key generation and verification utilities.
"""
import secrets
from typing import Tuple
import bcrypt

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
    Hash an API key using bcrypt.

    Bcrypt is preferred over SHA256 for API keys because:
    - Built-in salt (prevents rainbow table attacks)
    - Computationally expensive (slows brute-force attacks)
    - Adaptive work factor (can increase security over time)

    Args:
        api_key: Plain text API key

    Returns:
        Bcrypt hash of the key (stored as UTF-8 decoded string)
    """
    # Generate salt and hash the API key
    salt = bcrypt.gensalt(rounds=12)  # 12 rounds = good balance of security/performance
    hashed = bcrypt.hashpw(api_key.encode('utf-8'), salt)

    # Return as string for database storage
    return hashed.decode('utf-8')


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored bcrypt hash.

    Args:
        provided_key: API key provided by user
        stored_hash: Bcrypt hash stored in database

    Returns:
        True if key matches, False otherwise
    """
    try:
        # bcrypt.checkpw requires bytes for both arguments
        return bcrypt.checkpw(
            provided_key.encode('utf-8'),
            stored_hash.encode('utf-8')
        )
    except Exception:
        # If hashing fails (e.g., invalid hash format), return False
        return False
