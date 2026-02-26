"""
intelligent_dashboard.py — HIM Dashboard Launcher
Version: 1.0.0

Changelog:
- 1.0.0 (2026-02-26):
  - FIX: This file previously contained an old MentorExecutor (confusing + wrong name).
  - NEW: Now acts as a launcher for the Intelligent Dashboard (api_server.py).
Notes:
- Production-safe: delegates to api_server.main()
"""

from __future__ import annotations

import logging

import api_server


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("HIM")


def main() -> None:
    logger.info("Launching HIM Intelligent Dashboard...")
    logger.info("Open: http://127.0.0.1:5000/")
    api_server.main()


if __name__ == "__main__":
    main()