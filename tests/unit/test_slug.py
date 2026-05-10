"""Unit tests for core.slug — URL-safe slug derivation for property listings.

Slugs are used for the public-facing route /public/listings/{slug} (Phase 12)
and as a stable, human-readable identifier alongside the UUID primary key.

Contract:
  - slugify() lowercases, transliterates, replaces non-alphanumerics with '-',
    collapses runs, trims, caps length, and falls back to a non-empty default.
  - make_slug() always appends a 6-char id suffix so two properties with the
    same name still produce unique slugs.
"""

import pytest

from core.slug import make_slug, slugify


class TestSlugify:
    def test_basic_lowercase_with_spaces(self) -> None:
        assert slugify("Frankfurt House 1") == "frankfurt-house-1"

    def test_strips_punctuation(self) -> None:
        assert slugify("Hello!! World??") == "hello-world"

    def test_trims_surrounding_whitespace(self) -> None:
        assert slugify("   trimmed   ") == "trimmed"

    def test_collapses_internal_runs(self) -> None:
        assert slugify("a    b---c") == "a-b-c"

    def test_transliterates_accents(self) -> None:
        # German umlauts and French accents should reduce to ASCII.
        assert slugify("München Wohnung") == "munchen-wohnung"
        assert slugify("Café Déjà Vu") == "cafe-deja-vu"

    def test_handles_slashes_and_backslashes(self) -> None:
        assert slugify("a/b\\c") == "a-b-c"

    def test_caps_length(self) -> None:
        long = "a" * 500
        result = slugify(long, max_length=80)
        assert len(result) <= 80
        assert result == "a" * 80

    def test_does_not_truncate_mid_word_when_possible(self) -> None:
        # Truncation should prefer a hyphen boundary if one is reachable.
        result = slugify("aaaa bbbb cccc dddd", max_length=10)
        assert len(result) <= 10
        assert not result.endswith("-")

    def test_empty_name_falls_back(self) -> None:
        assert slugify("") == "property"

    def test_punctuation_only_falls_back(self) -> None:
        assert slugify("---!!!") == "property"

    def test_keeps_digits(self) -> None:
        assert slugify("Apartment 42B") == "apartment-42b"

    def test_idempotent(self) -> None:
        once = slugify("Some Property Name")
        twice = slugify(once)
        assert once == twice


class TestMakeSlug:
    def test_appends_6char_id_suffix(self) -> None:
        slug = make_slug("Test Name", "abcdef12-3456-7890-abcd-ef1234567890")
        assert slug == "test-name-abcdef"

    def test_suffix_uses_first_six_hex_chars_of_id(self) -> None:
        # Strips hyphen so suffix is always exactly 6 hex chars.
        slug = make_slug("X", "ab-cdef12-3456-7890-abcd-ef1234567890")
        assert slug.endswith("-abcdef")

    def test_two_same_names_get_different_slugs(self) -> None:
        s1 = make_slug("Same Name", "11111111-1111-1111-1111-111111111111")
        s2 = make_slug("Same Name", "22222222-2222-2222-2222-222222222222")
        assert s1 != s2
        assert s1.endswith("-111111")
        assert s2.endswith("-222222")

    def test_empty_name_falls_back_to_property(self) -> None:
        slug = make_slug("", "deadbeef-0000-0000-0000-000000000000")
        assert slug == "property-deadbe"

    def test_total_length_within_160(self) -> None:
        very_long = "a" * 500
        slug = make_slug(very_long, "abcdef12-0000-0000-0000-000000000000")
        assert len(slug) <= 160
        assert slug.endswith("-abcdef")

    def test_rejects_id_shorter_than_six_chars(self) -> None:
        with pytest.raises(ValueError):
            make_slug("anything", "abc")
