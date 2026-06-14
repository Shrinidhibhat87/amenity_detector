"""Locality enrichment package.

Turns a free-text location (PIN code, street, or Stadtteil) into structured
nearby points of interest plus a written neighbourhood blurb, using only the
free OpenStreetMap stack:

  - :mod:`core.locality.geocode`  — Nominatim free-text → lat/lon + bounding box.
  - :mod:`core.locality.overpass` — Overpass POI lookup around a coordinate.
  - :mod:`core.locality.agent`    — the tool-calling agent that orchestrates both.

All network results are cached so we stay within OSM fair-use limits and keep
running cost at $0. Data is © OpenStreetMap contributors (ODbL) — surface that
attribution wherever the results are shown.
"""

__all__: list[str] = []
