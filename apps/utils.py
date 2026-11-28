"""File paths and intent prompts used by more than one module."""

from pathlib import Path

# change this according to where `data/` should be stored
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "netstore1" / "dev" / "cos487-course-project"

DATA_DIR = ROOT_DIR / "data"

CORPUS_DIR = DATA_DIR / "JSON Files"
QRELS_DIR = DATA_DIR / "qrels"
METRICS_DIR = DATA_DIR / "metrics"
TRAINING_DIR = METRICS_DIR / "grid-search"
GENERATION_DIR = METRICS_DIR / "generation"
QUERY_DIR = DATA_DIR / "queries"
GENERATED_DIR = DATA_DIR / "generated"
INDEX_DIR = DATA_DIR / "indexes"
INDEX_MAP = {
    "paragraph": INDEX_DIR / "paragraph_index_stemmed.json",
    "paper": INDEX_DIR / "paper_index_stemmed.json",
}

# shared config path output by grid-search.py
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


REWRITE = """
Read the scientific question and rewrite it to maximize the chance of retrieving relevant paragraphs from scientific papers. 
This includes keeping the scientific terms that identify the topic, removing irrelevant words, and explicitly stating implied information. 
Do not answer the question. Return only the query, on one line.

Question: {query}
"""

HYDE = """
Write 2 sentences in the style of a research-paper paragraph that would answer this question. 
Do not invent numbers, names, theorem numbers, or results. Reuse the question's own terminology. 
If the question refers to specifics of one particular paper (a section, a theorem,
"the described system", "the twelve scenarios"), output only the question rephrased as a statement.
One line.

Question: {query}
"""

PROMPT_INTENT_DOMINANT = """
Read the question and identify its single dominant intent.
Write one well-suited retrieval query for that intent. Keep the scientific terms that identify the topic.
Do not answer the question. Return only the query, on one line.

Question: {query}
"""

PROMPT_INTENT_MULTI = """
Read the research question and identify its {i} most likely intents, most likely first.
Write one well-suited retrieval query for each intent. The queries must target different information needs.
Keep the scientific terms that identify the topic. Do not answer the question.
Return exactly {i} lines, one query per line. No numbering or explanation.

Question: {query}
"""

PROMPT_INTENT_LIKELIHOOD = """
Read the research question and identify its {i} most likely intents, most likely first.
Write one well-suited retrieval query for each intent, and the likelihood that this intent is what the user wants.
The likelihoods are decimals between 0 and 1 that sum to 1, used as query weights. The queries must target different information needs.
Keep the scientific terms that identify the topic. Do not answer the question.
Return exactly {i} lines. Each line is the likelihood and the query, separated by a single tab. No numbering or explanation.

Question: {query}
"""

PROMPT_MAP = {
    "rewrite": REWRITE,
    "hyde": HYDE,
    "dominant": PROMPT_INTENT_DOMINANT,
    "multi": PROMPT_INTENT_MULTI,
    "likelihood": PROMPT_INTENT_LIKELIHOOD
}
