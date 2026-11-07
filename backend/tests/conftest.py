"""Pytest configuration and global fixtures."""

import sys
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

# Ensure root workspace is on PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app.main import create_application
from backend.tests.fixtures.molecules import INVALID_SMILES, SAMPLE_MOLECULES


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient instance for integration tests."""
    app = create_application()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_molecules():
    """Return dictionary of verified test molecules."""
    return SAMPLE_MOLECULES


@pytest.fixture
def invalid_smiles_list():
    """Return list of invalid SMILES for negative test suites."""
    return INVALID_SMILES

