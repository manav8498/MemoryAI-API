#!/usr/bin/env python3.11
"""
Generate a valid JWT access token for load testing.
"""
from datetime import datetime, timedelta
from jose import jwt
import sys

# Same config as the API
JWT_SECRET_KEY = "your-secret-key-here"  # Will be set from .env or default
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 43200  # 30 days

# User ID from the existing token
USER_ID = "b1bc75e9-237d-4b5c-aff8-7daa1e07c5a6"

def create_access_token():
    """Create a valid JWT access token."""
    data = {
        "sub": USER_ID,
    }

    expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = data.copy()
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",  # CRITICAL: This field was missing!
    })

    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    return encoded_jwt

if __name__ == "__main__":
    # Try to load secret from .env
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        secret = os.getenv("JWT_SECRET_KEY")
        if secret:
            JWT_SECRET_KEY = secret
            print("✅ Loaded JWT_SECRET_KEY from .env")
    except Exception as e:
        print(f"⚠️  Could not load .env: {e}")
        print("Using default secret key...")

    token = create_access_token()
    print("\n" + "="*80)
    print("🔑 GENERATED VALID ACCESS TOKEN")
    print("="*80)
    print(token)
    print("="*80)
    print(f"\n✅ Token valid for {JWT_ACCESS_TOKEN_EXPIRE_MINUTES/60/24:.0f} days")
    print(f"👤 User ID: {USER_ID}")
    print("\nUse this token in your load tests!\n")
