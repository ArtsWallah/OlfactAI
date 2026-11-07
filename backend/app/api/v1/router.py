"""API v1 master router aggregating sub-routers."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import chemistry, health

api_router = APIRouter()

# Health routes under v1
api_router.include_router(health.router)

# Chemistry domain endpoints
api_router.include_router(chemistry.router)

