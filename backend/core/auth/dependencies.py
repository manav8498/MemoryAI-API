"""
FastAPI dependencies for authentication.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.core.auth.jwt import verify_token
from backend.core.auth.api_key import verify_api_key
from backend.models.user import User
from backend.models.api_key import APIKey
from backend.core.logging_config import logger


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_from_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user from JWT token.

    Args:
        credentials: HTTP Authorization header with Bearer token
        db: Database session

    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None

    token = credentials.credentials

    # Verify JWT token
    payload = verify_token(token)
    if not payload:
        return None

    user_id: str = payload.get("sub")
    if not user_id:
        return None

    # Get user from database
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)
    )
    user = result.scalar_one_or_none()

    return user


async def get_current_user_from_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user from API key.

    Args:
        credentials: HTTP Authorization header with Bearer token
        db: Database session

    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None

    provided_key = credentials.credentials

    # Check if it's an API key (starts with prefix)
    from backend.core.config import settings
    if not provided_key.startswith(settings.API_KEY_PREFIX):
        return None

    # Get all active API keys (we need to check hash)
    result = await db.execute(
        select(APIKey).where(APIKey.is_active == True)
    )
    api_keys = result.scalars().all()

    # Find matching key
    matched_key = None
    for api_key in api_keys:
        if verify_api_key(provided_key, api_key.key_hash):
            matched_key = api_key
            break

    if not matched_key:
        return None

    # Update last used timestamp
    from datetime import datetime
    matched_key.last_used_at = datetime.utcnow()
    await db.commit()

    # Get associated user
    result = await db.execute(
        select(User).where(User.id == matched_key.user_id, User.is_active == True)
    )
    user = result.scalar_one_or_none()

    return user


async def get_current_user(
    jwt_user: Optional[User] = Depends(get_current_user_from_jwt),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
) -> User:
    """
    Get current authenticated user (from JWT or API key).

    Args:
        jwt_user: User from JWT token
        api_key_user: User from API key

    Returns:
        Authenticated user

    Raises:
        HTTPException: If not authenticated
    """
    user = jwt_user or api_key_user

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Get current verified user.

    Args:
        current_user: Current active user

    Returns:
        Verified user

    Raises:
        HTTPException: If user email is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified",
        )

    return current_user
