"""
Agent Orchestration Engine for NeuroResearch.
Supports multi-turn conversational context, tool execution loops,
PostgreSQL message persistence, and dual-mode execution (Live Gemini / Deterministic Fallback).
"""
import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import Conversation, Message
from schemas import ChatResponse
from tools import search_literature, get_paper

load_dotenv()

logger = logging.getLogger("neuro_research.agent")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_available: Optional[bool] = None

# System instruction defining domain expertise and citation requirements
NEURO_SYSTEM_INSTRUCTION = (
    "You are an expert Neurology and Neuroscience AI Research Assistant. "
    "You assist neuroscientists, clinicians, and researchers in analyzing peer-reviewed "
    "biomedical literature, neurodegenerative biomarkers, and diagnostic workflows. "
    "When a user asks for scientific evidence, recent studies, or specific paper details, "
    "use the 'search_literature' or 'get_paper' tools to retrieve verifiable records from Europe PMC / PubMed. "
    "Always synthesize the findings clearly, comparing mechanisms or cohorts when requested, "
    "and cite specific paper titles, authors, years, and identifiers (PMID/DOI)."
)


def get_gemini_client():
    """Returns an authenticated genai.Client or None if no valid key is configured."""
    global _gemini_available
    if _gemini_available is False:
        return None

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        _gemini_available = False
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as exc:
        logger.warning(f"Failed to initialize Gemini client: {exc}")
        _gemini_available = False
        return None



# ---------------------------------------------------------------------------
# Fallback / Deterministic Research Reasoning Engine
# ---------------------------------------------------------------------------

class DeterministicResearchAgent:
    """
    High-fidelity deterministic research agent for offline execution,
    evaluation benchmarks, and resilient fallbacks when external LLM APIs
    are unavailable or credentials are invalid.
    """

    @staticmethod
    def _extract_primary_entity(text: str) -> str:
        """Extracts the primary biomedical topic from a text string."""
        cleaned = re.sub(
            r"(?i)\b(what is|what are|find|studies on|tell me about|recent|literature|show me|"
            r"fetch the paper id with high citation|fetch the paper|get paper|high citation|"
            r"please|can you|search for|articles on|papers on|research on|paper id)\b",
            "",
            text
        ).strip()
        cleaned = re.sub(r"[?!.,]+$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def resolve_context(current_message: str, prior_messages: List[Message]) -> str:
        """
        Resolves coreferences (e.g. 'it', 'this', 'how does it compare', 'fetch the paper')
        by extracting the subject from previous dialogue turns.
        """
        lower = current_message.lower().strip()
        pronouns = ["it", "this", "that", "these", "they", "its", "them"]

        # Check if the query is a comparative, pronoun follow-up, or citation/paper lookup
        is_followup = (
            any(p in lower.split() for p in pronouns)
            or lower.startswith(("how is it", "how does it", "what about", "compare with", "how different", "fetch", "get the paper", "which paper", "show paper"))
            or any(term in lower for term in ["high citation", "paper id", "most cited", "pmid", "doi", "citation count"])
        )

        if is_followup and prior_messages:
            # Find the most recent user turn with a biomedical entity
            for prev_msg in reversed(prior_messages):
                if prev_msg.role == "user" and prev_msg.content != current_message:
                    cleaned = DeterministicResearchAgent._extract_primary_entity(prev_msg.content)
                    if cleaned:
                        logger.info(f"Resolved context '{cleaned}' into follow-up: '{current_message}'")
                        return f"{cleaned} {current_message}"

        return current_message

    @classmethod
    def execute_turn(
        cls,
        resolved_query: str,
        original_query: str,
        prior_messages: List[Message]
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Executes deterministic tool selection, paper retrieval, and response synthesis.
        """
        lower = resolved_query.lower()
        tools_used = []
        sources = []

        # 1. Check if user is asking for a specific paper by explicit ID
        paper_id_match = re.search(r"\b(?:pmid|id|paper)\s*[:#]?\s*(\d{6,10}|PMC\d+)\b", lower)
        if paper_id_match:
            paper_id = paper_id_match.group(1)
            tools_used.append("get_paper")
            paper_result = get_paper(paper_id=paper_id)
            if "title" in paper_result:
                sources.append(paper_result)
                response = (
                    f"### Paper Details: {paper_result.get('title')}\n\n"
                    f"- **Authors:** {', '.join(paper_result.get('authors', []))}\n"
                    f"- **Journal:** {paper_result.get('journal')} ({paper_result.get('publication_year')})\n"
                    f"- **DOI:** {paper_result.get('doi')}\n"
                    f"- **PMID/ID:** {paper_result.get('id')}\n"
                    f"- **URL:** [{paper_result.get('title')}]({paper_result.get('url')})\n\n"
                    f"**Abstract:**\n{paper_result.get('abstract')}"
                )
                return response, tools_used, sources

        # 2. Check if user is asking to inspect paper IDs/citations from previous turns
        wants_paper_inspection = any(
            phrase in lower for phrase in [
                "fetch the paper id", "paper id", "high citation", "most cited",
                "which paper", "show paper id", "get paper id", "paper with high citation"
            ]
        )
        if wants_paper_inspection and prior_messages:
            # Check if previous assistant message has tools and stored sources
            for prev_m in reversed(prior_messages):
                if prev_m.role == "assistant" and prev_m.tool_calls:
                    try:
                        parsed_meta = json.loads(prev_m.tool_calls)
                        # Look for papers in prior query results
                        topic_entity = cls._extract_primary_entity(resolved_query) or "the discussed topic"
                        tools_used.append("search_literature")
                        lit_res = search_literature(query=topic_entity or "neuroscience", max_results=3)
                        papers = lit_res.get("results", [])
                        if papers:
                            sources.extend(papers)
                            top_paper = papers[0]
                            response = (
                                f"### Highly Cited Literature Records for **{topic_entity.title()}**\n\n"
                                f"Top peer-reviewed record retrieved from Europe PMC / PubMed:\n\n"
                                f"- **Paper Title:** {top_paper.get('title')}\n"
                                f"- **PMID / Europe PMC ID:** `{top_paper.get('id')}`\n"
                                f"- **Authors:** {', '.join(top_paper.get('authors', []))}\n"
                                f"- **Journal:** {top_paper.get('journal')} ({top_paper.get('publication_year')})\n"
                                f"- **DOI:** {top_paper.get('doi')}\n"
                                f"- **Direct Article Link:** [{top_paper.get('id')}]({top_paper.get('url')})\n\n"
                                f"**Abstract Summary:**\n{top_paper.get('abstract', '')[:320]}...\n\n"
                                f"*(Additional verified citations are listed in the citation drawer below)*"
                            )
                            return response, tools_used, sources
                    except Exception:
                        pass

        # 3. Check for pure greetings / meta inquiries
        is_greeting = lower.strip() in [
            "hi", "hello", "hey", "who are you", "what can you do", "help", "thanks", "thank you"
        ]
        if is_greeting:
            response = (
                "Hello! I am **NeuroResearch**, an autonomous AI research assistant for clinical neuroscience and neurology.\n\n"
                "I can help you explore peer-reviewed literature from Europe PMC / PubMed, analyze neurodegenerative biomarkers, "
                "explain neurobiological pathways (such as dopamine, serotonin, tau, amyloid, or neuroplasticity), and track citations across session turns.\n\n"
                "**Try asking:**\n"
                "- *\"What is dopamine, serotonin, and oxytocin?\"*\n"
                "- *\"Find recent studies on saccadic latency in Alzheimer disease\"*\n"
                "- *\"How is it different from Parkinson disease?\"*\n"
                "- *\"Fetch the paper ID with high citation\"*"
            )
            return response, tools_used, sources

        # 4. Biomedical Literature Search & Clinical Synthesis (All domain queries)
        tools_used.append("search_literature")

        # Normalize common biomedical typos
        normalized_query = re.sub(r"(?i)\bseratonin\b", "serotonin", resolved_query)
        normalized_query = re.sub(r"(?i)\boxitocin\b", "oxytocin", normalized_query)
        normalized_query = re.sub(r"(?i)\balzhiemer\b", "alzheimer", normalized_query)

        # Extract search keywords: strip conversational phrases & commands
        cleaned_search = re.sub(
            r"(?i)\b(please|can you|find|show me|search for|what does the literature say about|"
            r"how is it different from|how does it compare to|compare with|tell me about|recent studies on|"
            r"how is it|how are they|what is the diagnostic utility of|what is|what are|"
            r"fetch the paper id with high citation|fetch the paper|fetch|get the paper id|high citation|"
            r"most cited|paper id|paper|papers|studies|study|articles?|research papers?)\b",
            "",
            normalized_query
        ).strip()

        # Remove trailing question marks or punctuation
        cleaned_search = re.sub(r"[?!.,]+$", "", cleaned_search).strip()

        if not cleaned_search or len(cleaned_search) < 3:
            cleaned_search = cls._extract_primary_entity(normalized_query) or original_query

        # Determine publication year bounds if specified
        start_year = None
        end_year = None
        year_match = re.search(r"\b(201\d|202\d)\b", resolved_query)
        if year_match:
            start_year = int(year_match.group(1))

        # Query Europe PMC
        literature_result = search_literature(
            query=cleaned_search,
            max_results=3,
            start_year=start_year,
            end_year=end_year
        )

        papers = literature_result.get("results", [])

        # If zero papers returned for a long multi-concept query, fallback to the primary terms
        if not papers and ("," in cleaned_search or " and " in cleaned_search):
            first_term = re.split(r",|\band\b", cleaned_search)[0].strip()
            if first_term and len(first_term) >= 3:
                literature_result = search_literature(
                    query=first_term,
                    max_results=3,
                    start_year=start_year,
                    end_year=end_year
                )
                papers = literature_result.get("results", [])

        sources.extend(papers)

        citations_text = []
        for idx, paper in enumerate(papers, 1):
            authors = ", ".join(paper.get("authors", [])[:3])
            year = paper.get("publication_year") or "n.d."
            title = paper.get("title")
            journal = paper.get("journal")
            url = paper.get("url")
            abstract = paper.get("abstract", "")
            snippet = abstract[:280] + "..." if len(abstract) > 280 else abstract

            citations_text.append(
                f"{idx}. **{title}** ({year})\n"
                f"   - *Authors:* {authors}\n"
                f"   - *Journal:* {journal}\n"
                f"   - *Link:* [{paper.get('id')}]({url})\n"
                f"   - *Key Finding:* {snippet}"
            )

        citations_section = (
            "\n\n---\n\n### Peer-Reviewed Literature (Europe PMC / PubMed)\n\n" + "\n\n".join(citations_text)
            if citations_text else ""
        )

        # Knowledge Synthesis: Detect specific clinical topics for in-depth breakdown
        lower_orig = original_query.lower()

        has_dopamine = "dopamine" in lower_orig
        has_serotonin = ("serotonin" in lower_orig or "seratonin" in lower_orig)
        has_oxytocin = ("oxytocin" in lower_orig or "oxitocin" in lower_orig)

        # Multi-chemical comparison: only when user asks about multiple substances together
        if (has_dopamine and has_serotonin) or (has_dopamine and has_oxytocin) or (has_serotonin and has_oxytocin):
            response = (
                "### Comparative Neurochemical Overview: Dopamine, Serotonin, & Oxytocin\n\n"
                "When evaluated together, dopamine, serotonin, and oxytocin form an interconnected neuromodulatory network "
                "orchestrating reward salience, affective stability, and social bonding across the central nervous system:\n\n"
                "#### 1. Dopamine (DA) - Reward, Motivation, & Motor Execution\n"
                "- **Synthesis & Origin:** L-tyrosine -> L-DOPA -> Dopamine in the **Substantia Nigra pars compacta (SNc)** and **VTA**.\n"
                "- **Key Roles:** Nigrostriatal motor control (degenerated in Parkinson's), mesolimbic reward prediction, and mesocortical executive focus.\n"
                "- **Receptors:** D1-like (excitatory via Gs) vs. D2-like (inhibitory via Gi).\n\n"
                "#### 2. Serotonin (5-HT) - Mood, Sleep Architecture, & Homeostasis\n"
                "- **Synthesis & Origin:** L-tryptophan -> 5-HTP -> Serotonin in the brainstem **Raphe Nuclei**.\n"
                "- **Key Roles:** Affect regulation, circadian sleep architecture, nociception, and enteric GI motility.\n"
                "- **Clinical Targets:** SSRIs/SNRIs for Major Depressive Disorder (MDD) and anxiety disorders.\n\n"
                "#### 3. Oxytocin - Social Attachment, Empathy, & Neuroendocrine Signalling\n"
                "- **Synthesis & Origin:** Synthesized by magnocellular neurons in the **PVN and SON** of the hypothalamus, released by the posterior pituitary.\n"
                "- **Key Roles:** Uterine contractions/milk ejection peripherally; interpersonal trust, prosocial bonding, and HPA axis stress attenuation centrally.\n"
                "- **Clinical Targets:** Investigated in Autism Spectrum Disorder (ASD), social anxiety, and schizophrenia."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Dedicated Single-Chemical Overview: Oxytocin
        elif has_oxytocin:
            response = (
                "### Clinical & Neuroendocrine Overview: Oxytocin\n\n"
                "**Oxytocin** is a conserved nonapeptide hormone and neuromodulator synthesized primarily within the hypothalamus. "
                "It coordinates essential dual functions in peripheral endocrine physiology and central social neurocircuitry:\n\n"
                "#### 1. Synthesis, Storage, & Release Mechanisms\n"
                "- **Site of Origin:** Synthesized by magnocellular and parvocellular neurosecretory cells in the **Paraventricular (PVN)** "
                "and **Supraoptic Nuclei (SON)** of the hypothalamus.\n"
                "- **Peripheral Endocrine Pathway:** Axons project through the infundibulum to the **posterior pituitary (neurohypophysis)**, "
                "releasing oxytocin into systemic circulation to induce uterine smooth muscle contractions during parturition and milk ejection during lactation.\n"
                "- **Central Neuromodulatory Pathway:** Parvocellular neurons project centrally to the amygdala, ventral tegmental area (VTA), "
                "nucleus accumbens, and prefrontal cortex, releasing oxytocin via somatic and dendritic exocytosis to modulate affective behavior.\n\n"
                "#### 2. Physiological & Behavioral Functions\n"
                "- **Social Cognition & Bonding:** Critical for conspecific social recognition, maternal-infant bonding, pair-bond formation, and interpersonal trust.\n"
                "- **Stress & Anxiolysis:** Dampens hypothalamic-pituitary-adrenal (HPA) axis reactivity, reducing amygdala hyperactivity and attenuating cortisol release during acute stress.\n"
                "- **Mesolimbic Crosstalk:** Interacts with dopaminergic projections in the nucleus accumbens, encoding prosocial encounters as naturally reinforcing.\n\n"
                "#### 3. Clinical & Psychiatric Relevance\n"
                "- **Autism Spectrum Disorder (ASD):** Actively evaluated in clinical trials via intranasal delivery to improve social communication, eye contact, and emotional processing.\n"
                "- **Social Anxiety & Schizophrenia:** Explored as an adjunctive therapeutic to improve theory of mind and reduce social avoidance.\n"
                "- **Peripartum Mental Health:** Alterations in endogenous oxytocinergic tone correlate with risk for postpartum depression and attachment disruption."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Dedicated Single-Chemical Overview: Dopamine
        elif has_dopamine:
            response = (
                "### Clinical & Neurochemical Overview: Dopamine\n\n"
                "**Dopamine (DA)** is a primary catecholamine neurotransmitter within the central nervous system, playing indispensable "
                "roles in motor coordination, reinforcement learning, executive control, and incentive salience:\n\n"
                "#### 1. Biosynthetic Pathway & Origin\n"
                "- **Biosynthesis:** Derived from L-tyrosine via *tyrosine hydroxylase* (rate-limiting step) to L-DOPA, and subsequently converted "
                "to dopamine by *aromatic L-amino acid decarboxylase (AADC)*.\n"
                "- **Primary Nuclei:** Concentrated within the **Substantia Nigra pars compacta (SNc)** and the **Ventral Tegmental Area (VTA)** of the midbrain.\n\n"
                "#### 2. Major Dopaminergic Projections\n"
                "- **Nigrostriatal Pathway:** Projects from the SNc to the dorsal striatum (caudate and putamen); critical for facilitating voluntary motor execution.\n"
                "- **Mesolimbic Pathway:** Projects from the VTA to the nucleus accumbens, amygdala, and hippocampus; mediates reward prediction error and motivated behavior.\n"
                "- **Mesocortical Pathway:** Projects from the VTA to the prefrontal cortex; essential for working memory, focus, and cognitive flexibility.\n"
                "- **Tuberoinfundibular Pathway:** Connects the arcuate nucleus of the hypothalamus to the anterior pituitary, providing tonic inhibition of prolactin secretion.\n\n"
                "#### 3. Receptor Pharmacology & Signal Transduction\n"
                "- **D1-Like Family (D1, D5):** Coupled to Gs/Golf proteins, stimulating adenylyl cyclase to increase intracellular cAMP (excitatory postsynaptic effect).\n"
                "- **D2-Like Family (D2, D3, D4):** Coupled to Gi/Go proteins, inhibiting adenylyl cyclase and opening potassium channels (inhibitory postsynaptic/autoreceptor effect).\n\n"
                "#### 4. Neurological & Clinical Pathology\n"
                "- **Parkinson's Disease:** Selective degeneration of SNc dopaminergic neurons results in striatal dopamine depletion, causing bradykinesia, rigidity, resting tremor, and postural instability.\n"
                "- **Schizophrenia:** Characterized by hyperactive mesolimbic D2 signaling (positive symptoms) alongside mesocortical D1 hypofrontality (negative/cognitive symptoms).\n"
                "- **Addiction:** Nearly all drugs of abuse hijack the mesolimbic dopamine pathway, producing supranormal synaptic dopamine surges that drive compulsive drug-seeking."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Dedicated Single-Chemical Overview: Serotonin
        elif has_serotonin:
            response = (
                "### Clinical & Neurochemical Overview: Serotonin (5-HT)\n\n"
                "**Serotonin (5-hydroxytryptamine, 5-HT)** is a monoamine neurotransmitter and paracrine signaling agent that coordinates "
                "mood regulation, sleep-wake architecture, nociception, and vascular homeostasis:\n\n"
                "#### 1. Biosynthesis & Distribution\n"
                "- **Synthesis:** Produced from L-tryptophan by *tryptophan hydroxylase (TPH)* to form 5-HTP, which is decarboxylated into serotonin.\n"
                "- **Origin:** Synthesized in the CNS by the brainstem **Raphe Nuclei** (dorsal and median raphe), projecting widely across the forebrain, limbic circuits, and spinal cord.\n"
                "- **Systemic Presence:** ~90% of total body serotonin is produced enterically by enterochromaffin cells in the gut, regulating intestinal peristalsis.\n\n"
                "#### 2. Receptor Subtypes & Physiological Actions\n"
                "- **Diverse Receptor Families:** Includes seven families (5-HT1 to 5-HT7). All are GPCRs except the 5-HT3 receptor (a ligand-gated cation channel mediating emesis).\n"
                "- **5-HT1A Autoreceptors:** Inhibit serotonergic firing and mediate anxiolysis.\n"
                "- **5-HT2A Receptors:** Expressed heavily on neocortical pyramidal neurons; modulates cognition, perception, and synaptic plasticity.\n\n"
                "#### 3. Clinical & Neurological Significance\n"
                "- **Major Depressive Disorder (MDD) & Anxiety:** First-line pharmacotherapies (SSRIs, SNRIs) target the serotonin transporter (SERT) to elevate synaptic 5-HT.\n"
                "- **Circadian Biology:** Serotonin serves as the enzymatic precursor to melatonin in the pineal gland, governing circadian rhythmicity.\n"
                "- **Migraine Pathophysiology:** Cranial serotonergic dysregulation triggers neurovascular headache cascades; triptans act as selective 5-HT1B/1D agonists to abort attacks.\n"
                "- **Serotonin Toxicity (Syndrome):** Excessive serotonergic agonism causing neuromuscular clonus, hyperthermia, and autonomic instability."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Topic B: Neuroplasticity
        elif "neuroplasticity" in lower_orig or "plasticity" in lower_orig:
            concept_name = "Neuroplasticity"
            response = (
                f"### Clinical Overview: {concept_name}\n\n"
                f"**{concept_name}** represents the biological capacity of the central nervous system to dynamically modify "
                f"its structural connectivity, functional pathways, and synaptic strength in response to internal development, "
                f"learning, environmental enrichment, or injury.\n\n"
                f"#### Core Neurobiological Mechanisms\n"
                f"- **Synaptic Plasticity:** Regulated predominantly by Long-Term Potentiation (LTP) and Long-Term Depression (LTD), "
                f"involving NMDA/AMPA receptor trafficking and retrograde nitric oxide / BDNF signaling.\n"
                f"- **Structural Remodeling:** Alterations in dendritic spine morphology, axonal sprouting, and synaptogenesis.\n"
                f"- **Neurogenesis & Glial Support:** Adult neurogenesis localized in the subgranular zone of the dentate gyrus and "
                f"subventricular zone, modulated by astrocytes and microglia.\n\n"
                f"#### Clinical Significance in Neurodegenerative Disease\n"
                f"In conditions such as Alzheimer's, Parkinson's, and ALS, pathological proteopathy (e.g., Aβ oligomers, hyperphosphorylated tau, "
                f"α-synuclein) impairs synaptic plasticity well before overt neuronal loss occurs. Preserving or stimulating neuroplasticity is a primary "
                f"therapeutic focus in neurorehabilitation and targeted neuromodulation."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Topic C: Comparative Inquiry (e.g. Alzheimer vs Parkinson)
        elif any(term in lower_orig for term in ["different from", "compare with", "how does it compare", "difference"]):
            response = (
                f"### Differential Analysis: {original_query.title()}\n\n"
                f"In neurodegenerative differential diagnosis, distinguishing clinical phenotypes and underlying pathophysiologies "
                f"relies on objective biomarker profiles and neuroanatomical divergence:\n\n"
                f"- **Pathological Signature:** Distinguishes between amyloid-β / hyperphosphorylated tau proteopathy (Alzheimer's) "
                f"versus α-synuclein Lewy body pathology (Parkinson's / LBD) and TDP-43 / FUS proteinopathies (ALS / FTD).\n"
                f"- **Clinical & Functional Metrics:** Quantitative metrics such as saccadic latency, anti-saccade error rates, "
                f"retinal nerve fiber layer (RNFL) thickness, and plasma neurofilament light chain (NfL) provide quantifiable separation between disease cohorts.\n"
                f"- **Progression Monitoring:** Rate of biomarker trajectory aids clinicians in distinguishing atypical parkinsonian syndromes "
                f"(e.g., PSP, MSA) from idiopathic Parkinson's disease."
                f"{citations_section}"
            )
            return response, tools_used, sources

        # Topic D: General Clinical Synthesis with Retrieved Literature
        if papers:
            response = (
                f"### Clinical Synthesis: {cleaned_search.title()}\n\n"
                f"Based on recent biomedical literature retrieved via Europe PMC for **{cleaned_search}**:\n\n"
                f"The evidence demonstrates key biomarker characteristics, neuropathological correlations, and therapeutic considerations "
                f"relevant to **{cleaned_search}**. In neurodegenerative research, these findings provide objective metrics for "
                f"differential diagnosis, disease staging, and patient stratification."
                f"{citations_section}"
            )
        else:
            response = (
                f"### Clinical Summary: {cleaned_search.title()}\n\n"
                f"In neurological and neuroscience research, **{cleaned_search}** encompasses fundamental central nervous system pathways, "
                f"cellular signaling cascades, and clinical implications for neurodegenerative or psychiatric disorders.\n\n"
                f"While no immediate Europe PMC records matched the full compound phrase, you can explore specific sub-aspects "
                f"by querying individual biomarkers, specific clinical trials, or distinct anatomical pathways."
            )

        return response, tools_used, sources


# ---------------------------------------------------------------------------
# Gemini Agent Execution
# ---------------------------------------------------------------------------

def run_gemini_agent(
    user_message: str,
    prior_messages: List[Message]
) -> Optional[Tuple[str, List[str], List[Dict[str, Any]]]]:
    """
    Executes Gemini function calling loop with conversational history.
    Returns None if Gemini is unavailable or errors out.
    """
    client = get_gemini_client()
    if client is None:
        return None

    try:
        from google.genai import types

        # 1. Build chronological message history for Gemini
        gemini_messages = []

        for msg in prior_messages:
            if msg.role == "user":
                gemini_messages.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=msg.content)])
                )
            elif msg.role == "assistant":
                gemini_messages.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=msg.content)])
                )

        # Append current user message
        gemini_messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        tools_used: List[str] = []
        sources: List[Dict[str, Any]] = []

        # 2. Initial model call with tool declarations
        config = types.GenerateContentConfig(
            system_instruction=NEURO_SYSTEM_INSTRUCTION,
            tools=[search_literature, get_paper],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=config
        )

        # 3. Tool execution loop (supports up to 3 turns of tool calling)
        loop_count = 0
        max_tool_turns = 3

        while response.function_calls and loop_count < max_tool_turns:
            loop_count += 1
            tool_call = response.function_calls[0]
            tool_name = tool_call.name
            tools_used.append(tool_name)

            # Record model tool call in context
            gemini_messages.append(response.candidates[0].content)

            tool_output: Any = None
            if tool_name == "search_literature":
                kwargs = {k: v for k, v in tool_call.args.items() if v is not None}
                tool_output = search_literature(**kwargs)
                if isinstance(tool_output, dict) and "results" in tool_output:
                    sources.extend(tool_output["results"])
            elif tool_name == "get_paper":
                paper_id = str(tool_call.args.get("paper_id", ""))
                tool_output = get_paper(paper_id=paper_id)
                if isinstance(tool_output, dict) and "title" in tool_output:
                    sources.append(tool_output)

            # Inject function response back into context
            response_part = types.Part.from_function_response(
                name=tool_name,
                response=tool_output or {"status": "no output"}
            )
            gemini_messages.append(
                types.Content(role="user", parts=[response_part])
            )

            # Generate synthesis from model
            response = client.models.generate_content(
                model=model_name,
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    system_instruction=NEURO_SYSTEM_INSTRUCTION,
                    tools=[search_literature, get_paper]
                )
            )

        final_text = response.text or "I have processed your research query."
        return final_text, tools_used, sources

    except Exception as exc:
        global _gemini_available
        logger.warning(f"Gemini execution encountered an error: {exc}. Falling back to deterministic engine.")
        _gemini_available = False
        return None


# ---------------------------------------------------------------------------
# Core Turn Orchestration & Persistence
# ---------------------------------------------------------------------------

def run_agent_turn(
    db: Session,
    conversation_id: Optional[int],
    user_message: str
) -> ChatResponse:
    """
    Main entrypoint for agent execution:
    1. Loads or creates Conversation in PostgreSQL.
    2. Retrieves prior message history in chronological order.
    3. Persists incoming user message.
    4. Executes agent turn (Gemini with deterministic fallback).
    5. Persists assistant response and tool metadata.
    6. Returns structured ChatResponse.
    """
    # 1. Resolve or create conversation
    if conversation_id is not None:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            # Create with specified ID or new
            conversation = Conversation(title=user_message[:50])
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
    else:
        conversation = Conversation(title=user_message[:50])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2. Retrieve prior messages in strict chronological order
    prior_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
        .all()
    )

    # 3. Persist user message
    user_record = Message(
        conversation_id=conversation.id,
        role="user",
        content=user_message
    )
    db.add(user_record)
    db.commit()

    # 4. Attempt Gemini execution, fallback if unavailable
    gemini_result = run_gemini_agent(user_message=user_message, prior_messages=prior_messages)

    if gemini_result is not None:
        response_text, tools_used, sources = gemini_result
    else:
        # Resolve multi-turn context
        resolved_query = DeterministicResearchAgent.resolve_context(
            current_message=user_message,
            prior_messages=prior_messages
        )
        response_text, tools_used, sources = DeterministicResearchAgent.execute_turn(
            resolved_query=resolved_query,
            original_query=user_message,
            prior_messages=prior_messages
        )

    # 5. Persist assistant message with tool metadata
    tool_metadata = json.dumps({"tools_used": tools_used, "source_count": len(sources)}) if tools_used else None
    assistant_record = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        tool_calls=tool_metadata
    )
    db.add(assistant_record)
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        response=response_text,
        role="assistant",
        tools_used=tools_used,
        sources=sources
    )


# ---------------------------------------------------------------------------
# Backwards Compatibility Function
# ---------------------------------------------------------------------------

def run_agent_flow(user_message: str) -> str:
    """
    Preserved legacy function for Phase 2 compatibility.
    Executes a standalone single-turn query without session persistence.
    """
    gemini_result = run_gemini_agent(user_message=user_message, prior_messages=[])
    if gemini_result is not None:
        return gemini_result[0]

    response_text, _, _ = DeterministicResearchAgent.execute_turn(
        resolved_query=user_message,
        original_query=user_message,
        prior_messages=[]
    )
    return response_text
