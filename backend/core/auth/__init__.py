"""
Authentication utilities package.
"""
from backend.core.auth.jwt import create_access_token, verify_token
from backend.core.auth.password import hash_password, verify_password
from backend.core.auth.api_key import generate_api_key, hash_api_key, verify_api_key
from backend.core.auth.dependencies import get_current_user, get_current_active_user


__all__ = [
    "create_access_token",
    "verify_token",
    "hash_password",
    "verify_password",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "get_current_user",
    "get_current_active_user",
]
