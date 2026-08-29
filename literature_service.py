import requests
from typing import Dict, Any, List, Optional
import datetime

EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def fetch_europe_pmc_papers(
    query: str, 
    max_results: int = 5, 
    start_year: Optional[int] = None, 
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Standalone HTTP client to fetch scientific literature from Europe PMC API.
    Transforms raw external JSON into internal application schema.
    """
    # 1. Construct the query string with optional year filtering
    search_query = query.strip()
    
    if start_year or end_year:
        current_year = datetime.datetime.now().year
        s_year = start_year if start_year else 1900
        e_year = end_year if end_year else current_year
        search_query += f" AND (FIRST_PDATE:[{s_year} TO {e_year}])"
        
    params = {
        "query": search_query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core"  # 'core' provides abstractText and full metadata
    }
    
    headers = {
        "User-Agent": "NeurologyResearchAgent/1.0 (mailto:researcher@neuroagent.org)"
    }
    
    try:
        response = requests.get(
            EUROPE_PMC_BASE_URL, 
            params=params, 
            headers=headers, 
            timeout=10.0
        )
        
        # Check HTTP status codes
        if response.status_code == 429:
            return {
                "error": "Europe PMC rate limit exceeded. Please try again in a few moments.",
                "results": []
            }
        elif response.status_code >= 500:
            return {
                "error": f"Europe PMC server error (HTTP {response.status_code}). Service may be down.",
                "results": []
            }
        elif response.status_code != 200:
            return {
                "error": f"API request failed with HTTP status {response.status_code}.",
                "results": []
            }
            
        data = response.json()
        
    except requests.exceptions.Timeout:
        return {
            "error": "Request to Europe PMC timed out (10s limit). Check connection.",
            "results": []
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network error while reaching Europe PMC: {str(e)}",
            "results": []
        }
    except ValueError:
        return {
            "error": "Received malformed/non-JSON response from Europe PMC API.",
            "results": []
        }

    # 2. Parse raw response into standardized application schema (Adapter Pattern)
    raw_results = data.get("resultList", {}).get("result", [])
    
    if not raw_results:
        return {
            "message": f"No scientific papers found matching query: '{query}'",
            "results": []
        }
        
    parsed_papers: List[Dict[str, Any]] = []
    
    for paper in raw_results:
        # Format author list safely
        author_string = paper.get("authorString", "")
        authors_list = [a.strip() for a in author_string.split(",") if a.strip()] if author_string else ["Unknown Authors"]
        
        # Construct paper URL
        paper_id = paper.get("id") or paper.get("pmid") or "N/A"
        source_db = paper.get("source", "MED")
        paper_url = f"https://europepmc.org/article/{source_db}/{paper_id}" if paper_id != "N/A" else "https://europepmc.org"
        
        parsed_paper = {
            "id": str(paper_id),
            "title": paper.get("title", "Untitled Paper").rstrip('.'),
            "authors": authors_list[:5],  # Limit to top 5 authors for readability
            "abstract": paper.get("abstractText", "Abstract not available for this article."),
            "publication_year": int(paper.get("pubYear")) if str(paper.get("pubYear", "")).isdigit() else None,
            "doi": paper.get("doi", "N/A"),
            "journal": paper.get("journalTitle", "Unknown Journal"),
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
    Fetch full details and abstract for a specific paper by ID or PMID from Europe PMC.
    """
    # Clean paper_id
    clean_id = str(paper_id).strip()
    
    # Query Europe PMC by article ID or PMID
    search_res = fetch_europe_pmc_papers(query=f"ext_id:{clean_id} OR id:{clean_id}", max_results=1)
    
    results = search_res.get("results", [])
    if results:
        return results[0]
    
    return {
        "error": f"Paper with ID '{paper_id}' was not found in Europe PMC.",
        "paper_id": paper_id
    }

