"""
Preprocessing.py - IR Course Project, Part 1
Team: TheSearchParty

Turns the raw arXiv data into clean token files that the TF-IDF and BM25
scripts can load directly.

Corpus: 34,308 papers in "JSON Files/", each with title, abstract, and a
"paragraphs" dict mapping paragraph IDs (e.g. "2408.00001_8") to text.
That works out to 2,275,574 paragraphs total.

We build two document collections because the project evaluates retrieval at
two levels (see Study_paper_qrel.tsv and Study_paragraph_qrel.tsv):
    paper level     - one document per paper (title + abstract + paragraphs)
    paragraph level - one document per paragraph


WHAT GETS WRITTEN TO DISK, AND WHAT DOES NOT
--------------------------------------------
The files in processed/ hold CLEANED BUT UNSTEMMED words. Cleaning (removing
math, citations, URLs, stopwords) throws away things that are not content, so
it is safe to do once and reuse. Stemming is different: it is lossy and it is
a retrieval decision, not a data-cleaning one. Once "code" is written to disk
as "cod" the original word is gone.

Keeping the saved files unstemmed means:
  - the stemming on/off experiment is a load-time flag, not a 5 minute rebuild
  - the search interface can show real words in result snippets
  - error analysis for the paper is readable ("augmented" not "aug")

So stemming happens when the corpus is LOADED:

    from Preprocessing import load_corpus, load_queries

    docs    = load_corpus("paper", stem=True)     # yields (doc_id, tokens)
    queries = load_queries("study", stem=True)

IMPORTANT: pass the same stem= value to both calls. If the documents are
stemmed and the queries are not, matching terms stop lining up and the model
looks broken for no visible reason.


CLEANING STEPS (identical for documents and queries)
-----------------------------------------------------
    1. remove LaTeX math like $x_i$   - symbol noise, not searchable content
    2. remove LaTeX commands          - \\alpha, \\textbf{...}
    3. remove citations like [12]     - otherwise "12" becomes a fake term
    4. remove URLs
    5. lowercase and split into words (hyphenated words like "text-to-image"
       are kept together since they act as one term in this corpus)
    6. drop stopwords and single characters


OUTPUT FILES (all tab separated, "id <TAB> word word word")
------------------------------------------------------------
    processed/paper_corpus.tsv       one row per paper
    processed/paragraph_corpus.tsv   one row per paragraph
    processed/paper_titles.tsv       id, title, abstract - for the interface
    processed/queries_study.tsv      one row per query
    processed/queries_test.tsv
    processed/queries_train.tsv
    processed/build_info.txt         corpus statistics for the report

Run:  python Preprocessing.py
"""

import glob
import json
import os
import re

# Team name. The project requires result files to be named exactly
# teamName_paper_bm25_study.tsv and so on, and the grading scripts are
# automated, so this is kept here for the model scripts to import rather than
# being retyped in each one.
TEAM_NAME = "TheSearchParty"

# Folders
JSON_DIR = "data/JSON Files"
OUT_DIR = "data/processed"

# Set to a number (e.g. 500) to process only the first N papers for a quick
# test run. None means process the whole corpus.
LIMIT = None

# Common English words that appear in nearly every document, so they do not
# help tell documents apart. Kept conservative on purpose: frequent but
# meaningful words like "data" or "model" are left in, because inverse
# document frequency already discounts terms that appear everywhere.
STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its",
    "itself", "me", "more", "most", "my", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "therefore",
    "these", "they", "this", "those", "through", "thus", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours",
}

# Patterns used for cleaning. Compiled once here so they are not rebuilt for
# every one of the 2.3 million paragraphs.
MATH_PATTERN = re.compile(r"\$[^$]*\$")             # $x_i$, $$...$$
LATEX_PATTERN = re.compile(r"\\[a-zA-Z]+")          # \alpha, \textbf
CITATION_PATTERN = re.compile(r"\[\s*\d+(\s*,\s*\d+)*\s*\]")   # [12], [3, 4, 5]
URL_PATTERN = re.compile(r"https?://\S+")
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # keeps "text-to-image"


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------

def simple_stem(word):
    """Chop off the most common English endings so different forms of a word
    match each other ("models", "modeling", "modeled" -> "model").

    The rules run in a fixed order because the goal is consistency: every
    form of a word has to end up at the SAME stem. A stemmer that turns
    "studies" into one string and "study" into another is worse than no
    stemmer at all, since the query and the document stop matching.

    A stem does not have to be a real word. "code" and "coding" both becoming
    "cod" is fine, and is in fact the point.

    This is a simplified version of the Porter algorithm and it does not get
    every word right. NLTK's PorterStemmer handles the remaining cases and is
    not on the project's banned library list (only rank-bm25, PyTerrier, and
    scikit-learn are named), so it can be swapped in here if the professor
    confirms that outside text tools are allowed.
    """
    if len(word) <= 3:
        return word

    # 1. Plurals. "ss" is left alone so "address" does not become "addres".
    if word.endswith("sses"):        # classes -> class
        word = word[:-2]
    elif word.endswith("ies"):       # studies -> studi
        word = word[:-3] + "i"
    elif word.endswith("ss"):        # address -> address
        pass
    elif word.endswith("s"):         # models -> model
        word = word[:-1]

    # 2. Past tense and gerunds. After chopping, a doubled consonant is
    #    collapsed so "running" -> "runn" -> "run" matches "run".
    for ending in ("ing", "ed"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[: -len(ending)]
            if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break

    # 3. Trailing "y" becomes "i" so study / studies / studying all agree.
    if len(word) > 3 and word.endswith("y"):
        word = word[:-1] + "i"

    # 4. Common derivational endings. The remaining stem has to be at least
    #    5 characters, otherwise short stems get destroyed: with a smaller
    #    limit "document" turns into "docu" and "augmented" into "aug",
    #    which then no longer match "documentation" or "augmentation".
    for ending in ("ization", "iveness", "fulness", "ation", "ively",
                   "ness", "ment", "ive", "ity"):
        if word.endswith(ending) and len(word) - len(ending) >= 5:
            word = word[: -len(ending)]
            break

    # 5. "-ion" only after s or t, so "diffusion" -> "diffus" (matching
    #    "diffuse") while "region" is left alone instead of becoming "reg".
    if word.endswith("ion") and len(word) > 5 and word[-4] in "st":
        word = word[:-3]

    # 6. Trailing "e", so "memorize"/"memorized" and "make"/"making" agree.
    elif len(word) > 3 and word.endswith("e"):
        word = word[:-1]

    return word


# Stemming the same word over and over is wasted work: the corpus has millions
# of tokens but only a few hundred thousand distinct ones. Each distinct word
# is stemmed once and the answer is reused.
_STEM_CACHE = {}


def stem_tokens(tokens):
    """Stem a list of words, reusing results for words already seen."""
    out = []
    for token in tokens:
        stemmed = _STEM_CACHE.get(token)
        if stemmed is None:
            stemmed = simple_stem(token)
            _STEM_CACHE[token] = stemmed
        out.append(stemmed)
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_field(text):
    """Make a value safe to write into one TSV column. Paper titles and
    abstracts can contain tabs and line breaks, and either one would split the
    row into extra columns or extra lines and break every reader downstream."""
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def preprocess(text, stem=False):
    """Clean a piece of text and return its words.

    Used for documents at build time and for queries at search time, so that
    both sides are guaranteed to be treated the same way.

    stem defaults to False because that matches what is saved in processed/.
    The search interface should call preprocess(user_text, stem=True) only if
    the index it is searching was built from stemmed tokens.
    """
    if not text:
        return []

    text = MATH_PATTERN.sub(" ", text)
    text = LATEX_PATTERN.sub(" ", text)
    text = CITATION_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    words = WORD_PATTERN.findall(text.lower())
    tokens = [w for w in words if w not in STOPWORDS and len(w) >= 2]

    return stem_tokens(tokens) if stem else tokens


def build_paper_text(title, abstract, paragraph_texts):
    """Join the parts of a paper into one string for the paper-level document.

    About 87% of papers in this corpus repeat their abstract word for word as
    the first paragraph. Including both copies would count every abstract term
    twice, which inflates term frequency on exactly the words that describe
    what the paper is about. So the abstract is only added if it is not
    already present as a paragraph.
    """
    parts = [title]
    stripped = {p.strip() for p in paragraph_texts}
    if abstract.strip() and abstract.strip() not in stripped:
        parts.append(abstract)
    parts.extend(paragraph_texts)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Building the files
# ---------------------------------------------------------------------------

def build_corpus():
    """Read every paper JSON once and write both document collections."""
    files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if LIMIT:
        files = files[:LIMIT]
    print("Found %d paper files." % len(files))

    n_papers = 0
    n_paragraphs = 0
    n_empty_paragraphs = 0
    n_unreadable = 0
    n_abstract_deduped = 0
    paper_token_total = 0
    para_token_total = 0
    vocabulary = set()

    paper_path = os.path.join(OUT_DIR, "paper_corpus.tsv")
    para_path = os.path.join(OUT_DIR, "paragraph_corpus.tsv")
    title_path = os.path.join(OUT_DIR, "paper_titles.tsv")

    with open(paper_path, "w", encoding="utf-8") as paper_out, \
         open(para_path, "w", encoding="utf-8") as para_out, \
         open(title_path, "w", encoding="utf-8") as title_out:

        for i, path in enumerate(files, start=1):
            # One malformed file should not kill a five minute run.
            try:
                with open(path, encoding="utf-8") as f:
                    paper = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as err:
                print("  Skipping %s: %s" % (os.path.basename(path), err))
                n_unreadable += 1
                continue

            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""
            paragraphs = paper.get("paragraphs", {}) or {}
            paragraph_texts = [t or "" for t in paragraphs.values()]

            if abstract.strip() in {p.strip() for p in paragraph_texts}:
                n_abstract_deduped += 1

            # Paper-level document.
            full_text = build_paper_text(title, abstract, paragraph_texts)
            paper_tokens = preprocess(full_text)
            paper_out.write(paper_id + "\t" + " ".join(paper_tokens) + "\n")
            paper_token_total += len(paper_tokens)
            vocabulary.update(paper_tokens)
            n_papers += 1

            # Title and abstract are kept so the search interface has
            # something readable to display next to each result.
            title_out.write(paper_id + "\t" + clean_field(title) + "\t"
                            + clean_field(abstract) + "\n")

            # Paragraph-level documents: each paragraph stands on its own,
            # since that is the unit the paragraph qrel file judges.
            for para_id, para_text in paragraphs.items():
                para_tokens = preprocess(para_text or "")
                para_out.write(para_id + "\t" + " ".join(para_tokens) + "\n")
                para_token_total += len(para_tokens)
                n_paragraphs += 1
                if not para_tokens:
                    n_empty_paragraphs += 1

            if i % 2000 == 0:
                print("  %d / %d papers done (%d paragraphs)"
                      % (i, len(files), n_paragraphs))

    stats = [
        "Corpus build statistics",
        "=======================",
        "papers                     %d" % n_papers,
        "paragraphs                 %d" % n_paragraphs,
        "unreadable files skipped   %d" % n_unreadable,
        "",
        "vocabulary (unique words)  %d" % len(vocabulary),
        "tokens in paper corpus     %d" % paper_token_total,
        "tokens in paragraph corpus %d" % para_token_total,
        "avg tokens per paper       %.1f" % (paper_token_total / max(n_papers, 1)),
        "avg tokens per paragraph   %.1f" % (para_token_total / max(n_paragraphs, 1)),
        "",
        "papers whose abstract repeats as a paragraph   %d (%.0f%%)"
        % (n_abstract_deduped, 100.0 * n_abstract_deduped / max(n_papers, 1)),
        "paragraphs left with no words after cleaning   %d (%.2f%%)"
        % (n_empty_paragraphs, 100.0 * n_empty_paragraphs / max(n_paragraphs, 1)),
        "",
        "Files hold cleaned, UNSTEMMED words.",
        "Apply stemming at load time with load_corpus(..., stem=True).",
    ]
    report = "\n".join(stats)
    with open(os.path.join(OUT_DIR, "build_info.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print()
    print(report)


def build_queries():
    """Clean the queries the exact same way the documents were cleaned."""
    data_dir = "data"
    for name in ("Study", "Test", "Train"):
        path = os.path.join(data_dir, name + ".json")
        if not os.path.exists(path):
            print("  %s not found, skipping." % path)
            continue

        with open(path, encoding="utf-8") as f:
            queries = json.load(f)

        out_path = os.path.join(OUT_DIR, "queries_%s.tsv" % name.lower())
        with open(out_path, "w", encoding="utf-8") as out:
            for q in queries:
                tokens = preprocess(q["query"])
                out.write(str(q["query_id"]) + "\t" + " ".join(tokens) + "\n")
        print("  %s: %d queries" % (out_path, len(queries)))


# ---------------------------------------------------------------------------
# Loading the files (this is what the TF-IDF and BM25 scripts call)
# ---------------------------------------------------------------------------

def load_corpus(level, stem=False):
    """Yield (doc_id, tokens) for every document in a collection.

    level is "paper" or "paragraph".

    This is a generator, not a list, because the paragraph collection has
    about 2.3 million documents and holding them all in memory at once is
    unnecessary - an inverted index can be built in a single pass:

        index = {}
        for doc_id, tokens in load_corpus("paragraph", stem=True):
            for term in tokens:
                index.setdefault(term, []).append(doc_id)
    """
    if level not in ("paper", "paragraph"):
        raise ValueError("level must be 'paper' or 'paragraph', got %r" % level)

    path = os.path.join(OUT_DIR, "%s_corpus.tsv" % level)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            doc_id, _, token_text = line.partition("\t")
            tokens = token_text.split()
            yield doc_id, (stem_tokens(tokens) if stem else tokens)


def load_queries(name, stem=False):
    """Return a list of (query_id, tokens) for "study", "test" or "train".

    Use the same stem= value here as for load_corpus, otherwise query terms
    and document terms will not line up.
    """
    path = os.path.join(OUT_DIR, "queries_%s.tsv" % name.lower())
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _, token_text = line.partition("\t")
            tokens = token_text.split()
            out.append((qid, stem_tokens(tokens) if stem else tokens))
    return out


def load_titles():
    """Return {paper_id: (title, abstract)} for displaying search results."""
    path = os.path.join(OUT_DIR, "paper_titles.tsv")
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                titles[parts[0]] = (parts[1], parts[2])
    return titles


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_corpus()
    print()
    print("Building query files ...")
    build_queries()
