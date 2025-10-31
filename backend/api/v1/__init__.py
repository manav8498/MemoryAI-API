"""
API v1 routes.
"""
from fastapi import APIRouter

from backend.api.v1.endpoints import auth, users, collections, memories, search, profile
from backend.api.routes import (
    rl,
    procedural,
    temporal,
    working_memory,
    consolidation,
    memory_tools,
    world_model,
)


# Create v1 router
api_router = APIRouter()

# Include endpoint routers - Original endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(collections.router, prefix="/collections", tags=["Collections"])
api_router.include_router(memories.router, prefix="/memories", tags=["Memories"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(profile.router, prefix="/profile", tags=["User Profiles"])

# Include new advanced feature endpoints
api_router.include_router(rl.router, tags=["RL Training"])
api_router.include_router(procedural.router, tags=["Procedural Memory"])
api_router.include_router(temporal.router, tags=["Temporal Knowledge"])
api_router.include_router(working_memory.router, tags=["Working Memory"])
api_router.include_router(consolidation.router, tags=["Memory Consolidation"])
api_router.include_router(memory_tools.router, tags=["Self-Editing Memory"])
api_router.include_router(world_model.router, tags=["World Model & Planning"])


__all__ = ["api_router"]
