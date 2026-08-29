from typing import Dict, Any, Optional
from literature_service import fetch_europe_pmc_papers, fetch_paper_by_id

def search_literature(
    query: str, 
    start_year: Optional[int] = None, 
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Search live biomedical and scientific literature (via Europe PMC / PubMed) for a research query.
    
    Args:
        query: Scientific search keywords (e.g. 'eye movement Alzheimer biomarkers').
        start_year: Optional start publication year filter (e.g. 2020).
        end_year: Optional end publication year filter (e.g. 2026).
        
    Returns:
        A dictionary containing a list of real scientific paper metadata records.
    """
    print(f"[TOOL EXECUTION] Running real search_literature() on Europe PMC with query: '{query}' (years: {start_year}-{end_year})")
    return fetch_europe_pmc_papers(query=query, max_results=5, start_year=start_year, end_year=end_year)

def get_paper(paper_id: str) -> Dict[str, Any]:
    """
    Retrieve full details, journal info, and abstract of a specific medical paper by its ID or PMID.
    
    Args:
        paper_id: The string ID or PMID of the scientific paper (e.g. '35123456' or 'PMC1012345').
        
    Returns:
        A dictionary containing detailed paper metadata, abstract, DOI, and direct source link.
    """
    print(f"[TOOL EXECUTION] Running real get_paper() on Europe PMC for ID: '{paper_id}'")
    return fetch_paper_by_id(paper_id=str(paper_id))