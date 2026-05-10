"""URL-safe slug derivation for property listings.

A slug is the human-readable, lowercase, hyphen-separated portion of a public
URL — e.g. ``/public/listings/frankfurt-3bhk-abcdef``. Slugs are stored on
``Property.slug`` and are stable for the lifetime of the property.

`slugify()` converts an arbitrary string to a slug fragment.
`make_slug()` builds a final slug for a property by appending the first six
hex characters of the property's UUID, which guarantees uniqueness even when
two properties share the same name.
"""

from __future__ import annotations

import re
import unicodedata

DEFAULT_MAX_LENGTH = 150
ID_SUFFIX_LENGTH = 6
TOTAL_SLUG_CAP = 160
EMPTY_FALLBACK = "property"

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_HEX_CHARS = re.compile(r"[0-9a-f]")


def slugify(text: str, *, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Return a URL-safe slug fragment derived from ``text``.

    The transformation is:
      1. Unicode-normalise to NFKD and drop combining marks (``ü`` -> ``u``).
      2. ASCII-encode with ``ignore`` so anything still non-ASCII drops out.
      3. Lowercase, replace non-alphanumerics with a single ``-``.
      4. Strip leading/trailing hyphens.
      5. Truncate to ``max_length``, preferring to cut at the previous ``-``.
      6. Fall back to :data:`EMPTY_FALLBACK` if the result is empty.
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    hyphenated = _NON_SLUG_CHARS.sub("-", lowered).strip("-")

    if len(hyphenated) > max_length:
        truncated = hyphenated[:max_length]
        last_hyphen = truncated.rfind("-")
        if last_hyphen > 0:
            truncated = truncated[:last_hyphen]
        hyphenated = truncated.strip("-")

    return hyphenated or EMPTY_FALLBACK


def make_slug(name: str, property_id: str) -> str:
    """Build the final ``Property.slug`` value for a new property.

    Always appends ``-<6 hex chars>`` from ``property_id`` so two properties
    with the same name still produce distinct slugs. Total length is capped
    at :data:`TOTAL_SLUG_CAP`.

    Raises:
        ValueError: if ``property_id`` does not contain at least six hex
            characters (e.g. an empty or non-UUID string).
    """
    hex_chars = "".join(_HEX_CHARS.findall(property_id.lower()))
    if len(hex_chars) < ID_SUFFIX_LENGTH:
        raise ValueError(
            f"property_id must contain at least {ID_SUFFIX_LENGTH} hex characters; "
            f"got {property_id!r}"
        )
    suffix = hex_chars[:ID_SUFFIX_LENGTH]

    base_cap = TOTAL_SLUG_CAP - 1 - ID_SUFFIX_LENGTH  # 1 for the joining hyphen
    base = slugify(name, max_length=base_cap)
    return f"{base}-{suffix}"
