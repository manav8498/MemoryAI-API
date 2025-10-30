"""
Authentication endpoints.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr

from backend.core.database import get_db
from backend.core.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    generate_api_key,
)
from backend.models.user import User
from backend.models.api_key import APIKey
from backend.core.logging_config import logger


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class UserRegister(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    user_id: str


class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str


class APIKeyResponse(BaseModel):
    """API key response schema."""
    id: str
    name: str
    key: str  # Only returned on creation
    prefix: str
    created_at: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.

    Creates a new user account and returns an access token.
    """
    # Check if user already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user_id = str(uuid.uuid4())
    new_user = User(
        id=user_id,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    logger.info(f"New user registered: {user_data.email}")

    # Create access token
    access_token = create_access_token(data={"sub": user_id})

    return TokenResponse(
        access_token=access_token,
        user_id=user_id,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Login with email and password.

    Returns an access token for authenticated requests.
    """
    # Get user by email
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    logger.info(f"User logged in: {credentials.email}")

    # Create access token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
    )


@router.post("/token", response_model=TokenResponse)
async def login_oauth(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    OAuth2 compatible token endpoint.

    Uses username field for email and returns access token.
    """
    # Get user by email (username field in OAuth2)
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
    )


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new API key.

    The full API key is only returned once. Store it securely.
    """
    # Generate API key
    full_key, key_hash, prefix = generate_api_key()

    # Create API key record
    api_key = APIKey(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        key_hash=key_hash,
        name=key_data.name,
        prefix=prefix,
    )

    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    logger.info(f"API key created for user {current_user.email}: {key_data.name}")

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=full_key,  # Only returned here
        prefix=prefix,
        created_at=api_key.created_at,
    )


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """
    Get current user information.
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "tier": current_user.tier,
        "created_at": current_user.created_at,
    }
