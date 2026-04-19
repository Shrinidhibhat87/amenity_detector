"""Core package.

Keep this module lightweight. It is executed before any ``core.*`` submodule
import, including shared helpers used by the UI Docker image. Heavy imports such
as ``amenity_data_manager`` pull in API-only packages like ``db`` and should be
imported directly from their submodules instead.
"""

__all__: list[str] = []
