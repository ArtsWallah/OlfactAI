"""Model registry for versioning, storing, and loading smell prediction models."""

from pathlib import Path
from typing import Dict, List, Optional
from backend.app.core.exceptions import ResourceNotFoundError
from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.registry")


class ModelRegistry:
    """Manages trained and placeholder smell prediction models across versions."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, BaseSmellModel] = {}

    def register(self, model: BaseSmellModel) -> None:
        """Register an in-memory model instance."""
        key = f"{model.name}:{model.version}"
        self._loaded_models[key] = model
        logger.info(f"Registered model {key}")

    def get(self, name: str, version: Optional[str] = None) -> BaseSmellModel:
        """Retrieve a registered model by name and optional version."""
        if version:
            key = f"{name}:{version}"
            if key in self._loaded_models:
                return self._loaded_models[key]
        else:
            # Match latest version matching name
            candidates = [k for k in self._loaded_models if k.startswith(f"{name}:")]
            if candidates:
                candidates.sort()
                return self._loaded_models[candidates[-1]]

        raise ResourceNotFoundError("Model", f"{name}:{version or 'latest'}")

    def list_models(self) -> List[Dict[str, str]]:
        """List all currently registered models."""
        return [
            {
                "name": model.name,
                "version": model.version,
                "is_trained": str(model.is_trained),
                "labels_count": str(len(model.target_labels)),
            }
            for model in self._loaded_models.values()
        ]

