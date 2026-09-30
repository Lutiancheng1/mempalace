"""Unit tests for mempalace.date_window — shared since/before parsing (#463).

The same helpers back the ``list_drawers`` filter (#1128) via aliases in
``mcp_server``; behavioral coverage for that surface lives in
``test_mcp_server.py``. These tests pin the module contract directly.
"""

from datetime import datetime

import pytest

from mempalace.date_window import (
    compat_date_bounds,
    filed_at_in_window,
    parse_date_bound,
    parse_window,
)


class TestParseDateBound:
    def test_none_and_blank_mean_no_filter(self):
        assert parse_date_bound(None) is None
        assert parse_date_bound("") is None
        assert parse_date_bound("   ") is None

    def test_date_only(self):
        assert parse_date_bound("2026-04-01") == datetime(2026, 4, 1)

    def test_naive_datetime(self):
        assert parse_date_bound("2026-04-01T09:30:00") == datetime(2026, 4, 1, 9, 30)

    def test_fractional_seconds(self):
        assert parse_date_bound("2026-04-01T09:30:00.250000") == datetime(
            2026, 4, 1, 9, 30, 0, 250000
        )

    def test_zulu_datetime_parses_on_39_floor(self):
        assert parse_date_bound("2026-04-01T09:30:00Z") == datetime(2026, 4, 1, 9, 30)

    def test_zulu_date_only(self):
        # Regression: appending "+00:00" instead of stripping Z broke this
        # exact shape on Python 3.9/3.10 (caught by review on #1891).
        assert parse_date_bound("2026-04-01Z") == datetime(2026, 4, 1)

    def test_offset_dropped_wall_clock(self):
        assert parse_date_bound("2026-04-01T09:30:00+05:00") == datetime(2026, 4, 1, 9, 30)

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="since"):
            parse_date_bound(20260401, "since")

    def test_garbage_rejected_names_field(self):
        with pytest.raises(ValueError, match="before"):
            parse_date_bound("next tuesday", "before")


class TestParseWindow:
    def test_both_none(self):
        assert parse_window() == (None, None)

    def test_valid_window(self):
        since_dt, before_dt = parse_window("2026-04-01", "2026-04-10")
        assert since_dt == datetime(2026, 4, 1)
        assert before_dt == datetime(2026, 4, 10)

    def test_inverted_window_rejected(self):
        with pytest.raises(ValueError, match="must be earlier than"):
            parse_window("2026-04-10", "2026-04-01")

    def test_equal_bounds_rejected(self):
        with pytest.raises(ValueError, match="must be earlier than"):
            parse_window("2026-04-01", "2026-04-01")

    def test_invalid_since_names_field(self):
        with pytest.raises(ValueError, match="since"):
            parse_window("nope", None)


class TestFiledAtInWindow:
    def test_no_bounds_accepts_anything(self):
        assert filed_at_in_window("2026-01-01T00:00:00", None, None)

    def test_since_inclusive(self):
        since = datetime(2026, 1, 2)
        assert filed_at_in_window("2026-01-02T00:00:00", since, None)
        assert not filed_at_in_window("2026-01-01T23:59:59", since, None)

    def test_before_exclusive(self):
        before = datetime(2026, 1, 4)
        assert filed_at_in_window("2026-01-03T23:59:59", None, before)
        assert not filed_at_in_window("2026-01-04T00:00:00", None, before)

    def test_missing_filed_at_excluded_when_bound_active(self):
        assert not filed_at_in_window(None, datetime(2026, 1, 1), None)
        assert not filed_at_in_window("", datetime(2026, 1, 1), None)

    def test_unparseable_filed_at_excluded(self):
        assert not filed_at_in_window("not-a-date", datetime(2026, 1, 1), None)

    def test_aware_filed_at_compared_wall_clock(self):
        # diary_ingest stamps aware UTC ("+00:00"); the offset is dropped and
        # the wall-clock fields are compared.
        since = datetime(2026, 1, 5)
        assert filed_at_in_window("2026-01-05T00:00:00+00:00", since, None)
        assert not filed_at_in_window("2026-01-04T23:59:59+00:00", since, None)


# ── compat_date_bounds (fork filed_after/filed_before aliases) ─────────


class TestCompatDateBounds:
    def test_filed_after_maps_to_since_inclusive(self):
        assert compat_date_bounds(filed_after="2026-06-03") == ("2026-06-03", None)

    def test_date_only_filed_before_covers_whole_day(self):
        # Inclusive upper bound: through the end of 06-17 → exclusive 06-18.
        assert compat_date_bounds(filed_before="2026-06-17") == (None, "2026-06-18T00:00:00")

    def test_datetime_filed_before_adds_one_microsecond(self):
        since, before = compat_date_bounds(filed_before="2026-06-17T23:59:59")
        assert since is None
        assert before == "2026-06-17T23:59:59.000001"

    def test_both_bounds_together(self):
        since, before = compat_date_bounds("2026-06-03", "2026-06-17")
        assert since == "2026-06-03"
        assert before == "2026-06-18T00:00:00"

    def test_explicit_native_bounds_win(self):
        assert compat_date_bounds(
            filed_after="2026-01-01", filed_before="2026-01-31",
            since="2026-06-03", before="2026-06-04",
        ) == ("2026-06-03", "2026-06-04")

    def test_no_bounds_pass_through(self):
        assert compat_date_bounds() == (None, None)

    def test_blank_compat_values_are_ignored(self):
        assert compat_date_bounds(filed_after="  ", filed_before="") == (None, None)

    def test_resulting_window_parses(self):
        since, before = compat_date_bounds("2026-06-03", "2026-06-03")
        # Same-day inclusive range must form a valid non-inverted window.
        since_dt, before_dt = parse_window(since, before)
        assert since_dt == datetime(2026, 6, 3)
        assert before_dt == datetime(2026, 6, 4)
