"""
Compatibility import for older API code.

The logging setup is shared by both the API and UI containers and now lives in
``core.logging_config``. New code should import from there directly.
"""

from core.logging_config import setup_logging

__all__ = ["setup_logging"]
