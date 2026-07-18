"""exercises/service_scaffold.py
Exercise 13.1: Build a FastAPI service using awb_commons create_app().
Difficulty: ★★★☆☆ | Estimated time: 30 minutes

Goal:
    Create a health-checked, JWT-authenticated FastAPI service using
    the awb_commons create_app() factory.

Requirements:
    1. Use create_app() from awb_commons to instantiate the app.
    2. Expose a POST /v1/health-check endpoint.
    3. Validate the incoming Bearer JWT using verify_jwt_rs256().
    4. Return a HealthResponse JSON with service_name, version,
       status, and timestamp fields.
    5. Log the request using get_structured_logger().

Success criteria:
    pytest tests/test_exercise_1.py — all 3 tests must pass.

Notes:
    - The test suite supplies a pre-signed test JWT via the
      TEST_JWT environment variable.
    - Do NOT hardcode JWT_PUBLIC_KEY — read from os.environ.
    - Starter imports are provided; fill in the TODO sections.

Solution: github.com/lorvenio/ai-banking-risk-platform/
          chapter_013/solutions/solution_exercise_1.py
"""
import os
import sys

# Allow running from repo root: python exercises/service_scaffold.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import Request, HTTPException, status

# TODO: Import create_app, verify_jwt_rs256, get_structured_logger
#       from awb_commons.
# from awb_commons import create_app, verify_jwt_rs256
# from awb_commons import get_structured_logger
# from awb_commons.app_factory import HealthResponse

# TODO: Initialise structured logger.
# logger = get_structured_logger(__name__, service_name="exercise-13-1")

# TODO: Create the FastAPI app using create_app().
# The service_name should be "exercise-13-1" and version "1.0.0".
# app = create_app(...)


# TODO: Add a POST /v1/health-check endpoint.
# The endpoint must:
#   a) Extract the Bearer token from the Authorization header.
#   b) Call verify_jwt_rs256() — raise HTTP 401 on failure.
#   c) Return a HealthResponse with service_name, version, timestamp.
#   d) Log the request using logger.info().
#
# @app.post("/v1/health-check", response_model=HealthResponse)
# async def health_check_authed(request: Request):
#     ...


# ── Run with: uvicorn exercises.service_scaffold:app --reload ──────
# The test suite imports `app` from this module directly.
