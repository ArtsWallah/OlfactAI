"""OlfactAI FastAPI Application Entrypoint.

Initializes the API server, attaches middlewares, registers routers,
and sets up global exception handlers.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    OlfactAIException,
    generic_exception_handler,
    olfactai_exception_handler,
    validation_exception_handler,
)
from backend.app.core.logging import get_logger, setup_logging
from backend.app.schemas.health import HealthResponse

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info("Operating in strict CPU-only execution mode.")
    app.state.debug = settings.DEBUG
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME} gracefully.")


def create_application() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    app.add_exception_handler(OlfactAIException, olfactai_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Root health endpoint (GET /health)
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Root health check",
    )
    def root_health() -> HealthResponse:
        from backend.app.api.v1.endpoints.health import health_check
        return health_check()

    # Mount v1 API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app


app = create_application()

