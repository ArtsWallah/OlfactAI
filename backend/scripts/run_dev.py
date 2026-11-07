"""Convenience launcher for the OlfactAI development server."""

import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.core.config import get_settings


def main():
    """Launch Uvicorn development server."""
    settings = get_settings()
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} on {settings.HOST}:{settings.PORT}")
    print(f"API Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()

