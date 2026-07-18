# conftest.py — Chapter 16 test configuration
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring live services",
    )
