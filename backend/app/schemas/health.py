"""Health check response schemas."""

from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status payload."""

    status: str = Field("healthy", description="Overall system health status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Software semantic version")
    environment: str = Field(..., description="Current running environment")
    cpu_only: bool = Field(True, description="Indicates CPU-only execution constraint adherence")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Health status of critical backend subsystems (RDKit, PyTorch CPU, etc.)",
    )

