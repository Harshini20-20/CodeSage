"""Deterministic tests for Phase 2E source-reference validation."""
import pytest

from src.answer_evidence import validate_source_citations


def test_valid_source_references_are_preserved():
    result = validate_source_citations("This is explained in [SOURCE 1] and [SOURCE 2].", 2)
    assert result.answer == "This is explained in [SOURCE 1] and [SOURCE 2]."
    assert result.invalid_source_references == ()


def test_invalid_source_reference_is_removed_and_reported():
    result = validate_source_citations("The call is in [SOURCE 7].", 2)
    assert "[SOURCE 7]" not in result.answer
    assert "[invalid source reference removed]" in result.answer
    assert result.invalid_source_references == (7,)
    assert "Evidence note" in result.answer


def test_zero_sources_invalidates_any_source_marker():
    result = validate_source_citations("See [SOURCE 1].", 0)
    assert result.invalid_source_references == (1,)


def test_repeated_invalid_references_are_reported_once():
    result = validate_source_citations("[SOURCE 3], then [SOURCE 3].", 1)
    assert result.invalid_source_references == (3,)
    assert result.answer.count("[invalid source reference removed]") == 2


@pytest.mark.parametrize("bad_count", [-1, 1.5, True, "2"])
def test_invalid_source_count_raises(bad_count):
    with pytest.raises(ValueError, match="source_count"):
        validate_source_citations("Answer", bad_count)
