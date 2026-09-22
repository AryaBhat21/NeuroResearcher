"""
Agent tools for neurology biomedical literature search and paper inspection.
Exposed directly to LLM function calling interfaces.
"""
import logging
from typing import Dict, Any, Optional
from literature_service import fetch_europe_pmc_papers, fetch_paper_by_id

logger = logging.getLogger("neuro_research.tools")


def search_literature(
    query: str,
    max_results: Optional[int] = 5,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Search live peer-reviewed biomedical literature (via Europe PMC / PubMed)
    for neurological and medical research queries.

    Args:
        query: Specific scientific keywords or biomedical entities
               (e.g., 'saccadic eye movement Alzheimer disease', 'alpha-synuclein Parkinson').
        max_results: Maximum number of articles to return (1-10 recommended, default 5).
        start_year: Optional 4-digit beginning publication year filter (e.g. 2020).
        end_year: Optional 4-digit ending publication year filter (e.g. 2025).

    Returns:
        Structured dictionary with count and list of paper records containing
        title, authors, abstract, publication year, journal, DOI, PMID, and URL.
    """
    effective_max = max_results if (max_results is not None and max_results > 0) else 5
    logger.info(
        f"[TOOL EXECUTION] search_literature(query='{query}', "
        f"years={start_year}-{end_year}, max_results={effective_max})"
    )
    return fetch_europe_pmc_papers(
        query=query,
        max_results=effective_max,
        start_year=start_year,
        end_year=end_year
    )


def get_paper(paper_id: str) -> Dict[str, Any]:
    """
    Retrieve full details, journal citation metadata, and complete abstract
    for a specific biomedical paper using its PMID or Europe PMC ID.

    Args:
        paper_id: The string ID or PMID of the scientific paper
                  (e.g. '35123456', 'PMC1012345', or Europe PMC identifier).

    Returns:
        Dictionary containing title, authors, abstract, journal, DOI, and URL.
    """
    clean_id = str(paper_id).strip()
    logger.info(f"[TOOL EXECUTION] get_paper(paper_id='{clean_id}')")
    return fetch_paper_by_id(paper_id=clean_id)