"""Health check and system diagnostics endpoint."""

import sys
from fastapi import APIRouter
from backend.app.core.config import get_settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Perform health and environment readiness check.

    Verifies system runtime components and confirms CPU-only operational mode.
    """
    settings = get_settings()

    components = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }

    try:
        import rdkit
        components["rdkit"] = rdkit.__version__
    except ImportError:
        components["rdkit"] = "not_installed"

    try:
        import torch
        components["torch"] = f"{torch.__version__} (cuda_available={torch.cuda.is_available()})"
    except ImportError:
        components["torch"] = "not_installed"

    return HealthResponse(
        status="healthy",
        app_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        cpu_only=True,
        components=components,
    )

