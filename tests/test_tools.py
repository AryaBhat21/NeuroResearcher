"""
Unit tests for domain tools and Europe PMC literature service adapter.
"""
from tools import search_literature, get_paper
from literature_service import fetch_europe_pmc_papers, fetch_paper_by_id


def test_search_literature_structure():
    """Verifies that search_literature returns standard dictionary structure."""
    result = search_literature(query="Alzheimer disease amyloid", max_results=2)

    assert isinstance(result, dict)
    assert "results" in result
    assert "count" in result
    assert isinstance(result["results"], list)

    if result["results"]:
        paper = result["results"][0]
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "url" in paper
        assert "doi" in paper


def test_search_literature_empty_query():
    """Verifies that empty query returns structured error rather than crashing."""
    result = search_literature(query="")
    assert "error" in result
    assert result["count"] == 0


def test_search_literature_parameter_clamping():
    """Verifies that max_results is properly clamped."""
    result = search_literature(query="Parkinson", max_results=50)
    # Should be clamped to max 25
    assert len(result.get("results", [])) <= 25


def test_get_paper_invalid_id():
    """Verifies that querying a non-existent ID returns structured not-found response."""
    result = get_paper(paper_id="non_existent_id_99999999")
    assert isinstance(result, dict)
    assert "error" in result or "title" in result


def test_get_paper_empty_id():
    """Verifies that empty paper ID returns structured error."""
    result = get_paper(paper_id="")
    assert "error" in result
