"""
Literature service adapter for biomedical literature retrieval.
Integrates with Europe PMC REST API, transforming external JSON payloads
into structured, sanitized scientific article schemas.
"""
import logging
import datetime
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("neuro_research.literature")

EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def fetch_europe_pmc_papers(
    query: str,
    max_results: int = 5,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    HTTP client to query scientific literature from the Europe PMC API.
    Transforms raw external JSON into the internal application schema.

    Args:
        query: Search keywords or phrases.
        max_results: Maximum records to retrieve (clamped between 1 and 25).
        start_year: Optional starting publication year filter.
        end_year: Optional ending publication year filter.

    Returns:
        Dictionary with count and list of parsed scientific paper records.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return {
            "error": "Query string must not be empty.",
            "count": 0,
            "results": []
        }

    # Bound max_results safely
    effective_max = max(1, min(int(max_results), 25))

    # Construct the query string with optional year range filtering
    search_query = clean_query
    if start_year or end_year:
        current_year = datetime.datetime.now().year
        s_year = int(start_year) if start_year else 1900
        e_year = int(end_year) if end_year else current_year
        search_query += f" AND (FIRST_PDATE:[{s_year} TO {e_year}])"

    params = {
        "query": search_query,
        "format": "json",
        "pageSize": effective_max,
        "resultType": "core"  # 'core' provides abstractText and complete metadata
    }

    headers = {
        "User-Agent": "NeurologyResearchAgent/1.0 (mailto:researcher@neuroagent.org)"
    }

    logger.info(f"Querying Europe PMC: '{search_query}' (pageSize={effective_max})")

    try:
        response = requests.get(
            EUROPE_PMC_BASE_URL,
            params=params,
            headers=headers,
            timeout=10.0
        )

        if response.status_code == 429:
            logger.warning("Europe PMC rate limit hit (HTTP 429).")
            return {
                "error": "Europe PMC rate limit exceeded. Please try again in a few moments.",
                "count": 0,
                "results": []
            }
        elif response.status_code >= 500:
            logger.error(f"Europe PMC server error HTTP {response.status_code}")
            return {
                "error": f"Europe PMC server error (HTTP {response.status_code}). Service may be temporarily unavailable.",
                "count": 0,
                "results": []
            }
        elif response.status_code != 200:
            logger.warning(f"Europe PMC request failed with HTTP {response.status_code}")
            return {
                "error": f"API request failed with HTTP status {response.status_code}.",
                "count": 0,
                "results": []
            }

        data = response.json()

    except requests.exceptions.Timeout:
        logger.error("Request to Europe PMC timed out (10s threshold).")
        return {
            "error": "Request to Europe PMC timed out (10s limit). Check network connection.",
            "count": 0,
            "results": []
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while reaching Europe PMC: {e}")
        return {
            "error": f"Network error while reaching Europe PMC: {str(e)}",
            "count": 0,
            "results": []
        }
    except ValueError:
        logger.error("Received non-JSON response from Europe PMC API.")
        return {
            "error": "Received malformed response from Europe PMC API.",
            "count": 0,
            "results": []
        }

    raw_results = data.get("resultList", {}).get("result", [])

    if not raw_results:
        return {
            "message": f"No scientific papers found matching query: '{query}'",
            "count": 0,
            "results": []
        }

    parsed_papers: List[Dict[str, Any]] = []

    for paper in raw_results:
        author_string = paper.get("authorString", "")
        authors_list = [a.strip() for a in author_string.split(",") if a.strip()] if author_string else ["Unknown Authors"]

        paper_id = paper.get("id") or paper.get("pmid") or "N/A"
        source_db = paper.get("source", "MED")
        paper_url = f"https://europepmc.org/article/{source_db}/{paper_id}" if paper_id != "N/A" else "https://europepmc.org"

        pub_year = paper.get("pubYear")
        valid_year = int(pub_year) if str(pub_year).isdigit() else None

        parsed_paper = {
            "id": str(paper_id),
            "title": str(paper.get("title", "Untitled Paper")).rstrip('.'),
            "authors": authors_list[:5],
            "abstract": str(paper.get("abstractText", "Abstract not available for this article.")),
            "publication_year": valid_year,
            "doi": str(paper.get("doi", "N/A")),
            "journal": str(paper.get("journalTitle", "Unknown Journal")),
            "source": f"Europe PMC ({source_db})",
            "url": paper_url
        }
        parsed_papers.append(parsed_paper)

    return {
        "count": len(parsed_papers),
        "results": parsed_papers
    }


def fetch_paper_by_id(paper_id: str) -> Dict[str, Any]:
    """
    Fetch details and abstract for a specific paper by PMID or Europe PMC ID.
    """
    clean_id = str(paper_id).strip()
    if not clean_id:
        return {"error": "Paper ID must not be empty.", "paper_id": paper_id}

    logger.info(f"Looking up paper by ID: '{clean_id}'")
    search_res = fetch_europe_pmc_papers(query=f"ext_id:{clean_id} OR id:{clean_id}", max_results=1)

    results = search_res.get("results", [])
    if results:
        return results[0]

    return {
        "error": f"Paper with ID '{paper_id}' was not found in Europe PMC.",
        "paper_id": clean_id
    }
