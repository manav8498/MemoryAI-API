"""
API routes package.
"""
from fastapi import APIRouter

from backend.api.v1 import api_router as v1_router


# Main API router
api_router = APIRouter()

# Include v1 routes
api_router.include_router(v1_router)


__all__ = ["api_router"]
