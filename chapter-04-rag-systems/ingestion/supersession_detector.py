"""Supersession detector — ingestion-pipeline entry point.

The full implementation lives in awb_commons.rag.supersession_detector and
is re-exported here at the path referenced in Chapter 4: the ingestion
pipeline imports from `ingestion.supersession_detector`.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from awb_commons.rag.supersession_detector import *  # noqa: F401,F403
