# chapter_06/tests/conftest.py
# Shared pytest configuration for Chapter 6 test suite.
import pytest
import sys
import os

# Ensure imports resolve from the code root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
