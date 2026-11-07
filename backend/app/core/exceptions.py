"""Custom domain exceptions and global error handlers for OlfactAI."""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from backend.app.core.logging import get_logger

logger = get_logger("exceptions")


class ErrorDetail(BaseModel):
    """Structured error detail schema."""
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Standardized API error response body."""
    success: bool = False
    error: ErrorDetail


class OlfactAIException(Exception):
    """Base exception for all OlfactAI domain errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class InvalidSMILESError(OlfactAIException):
    """Raised when an invalid SMILES string is supplied."""

    def __init__(
        self,
        message: str = "Invalid chemical SMILES representation.",
        smiles: Optional[str] = None,
        details: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            code="INVALID_SMILES",
            status_code=422,
            details={"smiles": smiles, **(details or {})} if smiles else details,
        )


class MolecularProcessingError(OlfactAIException):
    """Raised when molecular descriptor or fingerprint calculation fails."""

    def __init__(
        self,
        message: str = "Failed to calculate molecular features.",
        details: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            code="MOLECULAR_PROCESSING_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class ModelNotTrainedError(OlfactAIException):
    """Raised when inference is attempted on an untrained placeholder model."""

    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
    ):
        msg = message or f"Model '{model_name}' has not been trained yet."
        super().__init__(
            message=msg,
            code="MODEL_NOT_TRAINED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"model_name": model_name},
        )


class ResourceNotFoundError(OlfactAIException):
    """Raised when a requested resource (dataset, model, etc.) is not found."""

    def __init__(
        self,
        resource_type: str,
        identifier: str,
    ):
        super().__init__(
            message=f"{resource_type} '{identifier}' was not found.",
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "identifier": identifier},
        )


async def olfactai_exception_handler(
    request: Request, exc: OlfactAIException
) -> JSONResponse:
    """Handle custom OlfactAI application exceptions."""
    logger.warning(
        f"Domain exception on {request.method} {request.url.path}: "
        f"[{exc.code}] {exc.message}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors from Pydantic."""
    logger.info(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": exc.errors(),
            },
        },
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected server exceptions."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during processing.",
                "details": str(exc) if request.app.state.debug else None,
            },
        },
    )
