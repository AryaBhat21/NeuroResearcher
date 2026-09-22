"""
Research & Evaluation Harness for NeuroResearch Agent.
Provides reproducible evaluation benchmarks for:
1. Intent Classification & Tool Routing Accuracy
2. Multi-turn Coreference & Context Retention
3. Citation Traceability & Scientific Verifiability
4. Latency Profiling

Run via: python evaluate.py
"""
import time
import json
import logging
from typing import List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from agent import run_agent_turn, DeterministicResearchAgent

logging.basicConfig(level=logging.WARNING)

# In-memory evaluation database to ensure clean, isolated benchmark runs
EVAL_DB_URL = "sqlite:///:memory:"
eval_engine = create_engine(EVAL_DB_URL, connect_args={"check_same_thread": False})
EvalSession = sessionmaker(autocommit=False, autoflush=False, bind=eval_engine)
Base.metadata.create_all(bind=eval_engine)


# ---------------------------------------------------------------------------
# Benchmark Test Cases
# ---------------------------------------------------------------------------

TOOL_ROUTING_BENCHMARK = [
    {
        "id": "TR-01",
        "query": "Find recent clinical studies on retinal nerve fiber layer thinning in Parkinson disease",
        "expected_tool": "search_literature",
        "domain": "Neuro-Ophthalmology / Parkinson's"
    },
    {
        "id": "TR-02",
        "query": "Search literature for tau PET imaging in frontotemporal lobar degeneration",
        "expected_tool": "search_literature",
        "domain": "Molecular Neuroimaging / FTLD"
    },
    {
        "id": "TR-03",
        "query": "Look up paper PMID: 35123456",
        "expected_tool": "get_paper",
        "domain": "Specific Paper Retrieval"
    },
    {
        "id": "TR-04",
        "query": "Explain the general physiological role of the blood-brain barrier",
        "expected_tool": None,
        "domain": "Neurobiology Fundamentals"
    },
    {
        "id": "TR-05",
        "query": "What is the biochemical structure of dopamine?",
        "expected_tool": None,
        "domain": "Neurochemistry Fundamentals"
    },
    {
        "id": "TR-06",
        "query": "Show studies on plasma p-tau217 as a diagnostic biomarker in early Alzheimer disease",
        "expected_tool": "search_literature",
        "domain": "Fluid Biomarkers / Alzheimer's"
    }
]

MULTITURN_BENCHMARK = [
    {
        "id": "MT-01",
        "turn_1": "What are the primary saccadic eye movement abnormalities in Alzheimer disease?",
        "turn_2": "How do they differ from progressive supranuclear palsy?",
        "expected_turn_2_tool": "search_literature",
        "target_coreference_concept": "saccadic",
        "domain": "Oculomotor Biomarkers"
    },
    {
        "id": "MT-02",
        "turn_1": "What is the diagnostic utility of neurofilament light chain in amyotrophic lateral sclerosis?",
        "turn_2": "How is it measured in cerebrospinal fluid versus plasma?",
        "expected_turn_2_tool": "search_literature",
        "target_coreference_concept": "neurofilament",
        "domain": "Fluid Biomarkers / Motor Neuron Disease"
    }
]


# ---------------------------------------------------------------------------
# Benchmark Runners
# ---------------------------------------------------------------------------

def run_tool_routing_eval() -> Dict[str, Any]:
    """Evaluates agent tool selection accuracy against ground truth annotations."""
    db = EvalSession()
    correct = 0
    total = len(TOOL_ROUTING_BENCHMARK)
    results = []
    latencies = []

    for test in TOOL_ROUTING_BENCHMARK:
        start_time = time.perf_counter()
        resp = run_agent_turn(db, conversation_id=None, user_message=test["query"])
        elapsed = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed)

        actual_tool = resp.tools_used[0] if resp.tools_used else None
        is_match = (actual_tool == test["expected_tool"])
        if is_match:
            correct += 1

        results.append({
            "id": test["id"],
            "query": test["query"],
            "expected_tool": test["expected_tool"],
            "actual_tool": actual_tool,
            "latency_ms": round(elapsed, 1),
            "passed": is_match
        })

    db.close()
    accuracy = (correct / total) * 100.0
    avg_latency = sum(latencies) / len(latencies)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "average_latency_ms": round(avg_latency, 1),
        "results": results
    }


def run_multiturn_eval() -> Dict[str, Any]:
    """Evaluates multi-turn context retention and pronoun/coreference resolution."""
    db = EvalSession()
    correct = 0
    total = len(MULTITURN_BENCHMARK)
    results = []

    for test in MULTITURN_BENCHMARK:
        # Turn 1
        t1_resp = run_agent_turn(db, conversation_id=None, user_message=test["turn_1"])
        cid = t1_resp.conversation_id

        # Turn 2 (dependent on Turn 1)
        t2_resp = run_agent_turn(db, conversation_id=cid, user_message=test["turn_2"])

        actual_tool = t2_resp.tools_used[0] if t2_resp.tools_used else None
        tool_matched = (actual_tool == test["expected_turn_2_tool"])

        # Check if response or retrieved query incorporates the antecedent concept
        text_retained = test["target_coreference_concept"].lower() in t2_resp.response.lower()

        passed = tool_matched and text_retained
        if passed:
            correct += 1

        results.append({
            "id": test["id"],
            "turn_1": test["turn_1"],
            "turn_2": test["turn_2"],
            "resolved_concept": test["target_coreference_concept"],
            "actual_tool": actual_tool,
            "concept_retained": text_retained,
            "passed": passed
        })

    db.close()
    return {
        "accuracy": (correct / total) * 100.0,
        "correct": correct,
        "total": total,
        "results": results
    }


def run_citation_traceability_eval() -> Dict[str, Any]:
    """Verifies that retrieved literature contains full verifiable provenance fields."""
    db = EvalSession()
    resp = run_agent_turn(
        db,
        conversation_id=None,
        user_message="Find peer-reviewed studies on alpha-synuclein seeding assays in Parkinson disease"
    )
    db.close()

    sources = resp.sources
    total_sources = len(sources)
    valid_citations = 0

    for s in sources:
        has_title = bool(s.get("title") and s.get("title") != "Untitled Paper")
        has_id = bool(s.get("id") and s.get("id") != "N/A")
        has_url = bool(s.get("url") and s.get("url").startswith("http"))
        has_authors = bool(s.get("authors") and len(s.get("authors")) > 0)

        if has_title and has_id and has_url and has_authors:
            valid_citations += 1

    traceability_rate = (valid_citations / total_sources * 100.0) if total_sources > 0 else 0.0

    return {
        "sources_retrieved": total_sources,
        "valid_citations": valid_citations,
        "traceability_rate": round(traceability_rate, 1)
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint & Report Generation
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" NeuroResearch Agent — Research & Evaluation Benchmark Suite")
    print("=" * 70)

    print("\n[1/3] Running Tool Routing & Intent Benchmark...")
    routing = run_tool_routing_eval()
    print(f" -> Accuracy: {routing['accuracy']:.1f}% ({routing['correct']}/{routing['total']})")
    print(f" -> Mean Latency: {routing['average_latency_ms']} ms")
    for r in routing["results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {status} {r['id']}: expected={r['expected_tool']}, actual={r['actual_tool']} ({r['latency_ms']}ms)")

    print("\n[2/3] Running Multi-Turn Coreference Resolution Benchmark...")
    multiturn = run_multiturn_eval()
    print(f" -> Retention Accuracy: {multiturn['accuracy']:.1f}% ({multiturn['correct']}/{multiturn['total']})")
    for r in multiturn["results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {status} {r['id']}: target concept '{r['resolved_concept']}' retained={r['concept_retained']}")

    print("\n[3/3] Running Citation Provenance & Traceability Benchmark...")
    citations = run_citation_traceability_eval()
    print(f" -> Provenance Verification Rate: {citations['traceability_rate']}% "
          f"({citations['valid_citations']}/{citations['sources_retrieved']} papers fully resolved)")

    summary = {
        "tool_routing_accuracy_pct": routing["accuracy"],
        "mean_latency_ms": routing["average_latency_ms"],
        "multiturn_resolution_accuracy_pct": multiturn["accuracy"],
        "citation_provenance_rate_pct": citations["traceability_rate"]
    }

    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" Benchmark Run Complete. Results saved to eval_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
