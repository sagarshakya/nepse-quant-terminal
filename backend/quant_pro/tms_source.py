"""
TMS data source stub.

Live brokerage (TMS19) not included in public release.
"""
from __future__ import annotations
from enum import Enum


class TMSSource(str, Enum):
    PAPER = "paper"
    LIVE = "live"
