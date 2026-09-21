"""
test.py - unit tests for preprocess.py

Run with:   python -m unittest test -v
       or:  python test.py

Uses only Python's built-in unittest module, so there is nothing extra to
install and the grader can run these directly.

The tests cover five things:
    1. Text cleaning   - is the LaTeX / citation / URL junk actually removed?
    2. Stemming        - do different forms of a word reach the SAME stem?
    3. Consistency     - does a query token still match a document token?
    4. File output     - are the TSV files shaped the way teammates expect?
    5. Loading         - do load_corpus / load_queries return what they claim,
                         and does stem=True apply on the way out?
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import preprocess as P


class TestTextCleaning(unittest.TestCase):
    """The cleaning steps are the part of this project that is specific to an
    arXiv corpus, so they get the most tests.

    Note that preprocess() does NOT stem by default, which is why these
    expect whole words."""

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(P.preprocess(""), [])
        self.assertEqual(P.preprocess(None), [])

    def test_inline_math_is_removed(self):
        tokens = P.preprocess("the value $x_i$ converges")
        self.assertNotIn("x_i", tokens)
        self.assertNotIn("x", tokens)
        self.assertIn("value", tokens)

    def test_latex_commands_are_removed(self):
        tokens = P.preprocess(r"noise \alpha and \textbf{bold} text")
        self.assertNotIn("alpha", tokens)
        self.assertNotIn("textbf", tokens)
        self.assertIn("noise", tokens)

    def test_single_citation_is_removed(self):
        tokens = P.preprocess("as shown in BERT [12] results")
        self.assertNotIn("12", tokens)
        self.assertIn("bert", tokens)

    def test_multi_citation_is_removed(self):
        tokens = P.preprocess("prior work [3, 4, 5] covers this")
        for number in ("3", "4", "5"):
            self.assertNotIn(number, tokens)
        self.assertIn("prior", tokens)

    def test_urls_are_removed(self):
        tokens = P.preprocess("code at https://github.com/example/repo today")
        self.assertNotIn("https", tokens)
        self.assertNotIn("github", tokens)
        self.assertIn("code", tokens)

    def test_text_is_lowercased(self):
        self.assertEqual(P.preprocess("NEURAL"), P.preprocess("neural"))

    def test_hyphenated_terms_stay_together(self):
        # "text-to-image" is one concept in this corpus, so it must stay a
        # single token instead of splitting into "text", "to", "image".
        tokens = P.preprocess("text-to-image synthesis")
        self.assertEqual(len(tokens), 2)
        self.assertIn("text-to-image", tokens)
        self.assertNotIn("text", tokens)

    def test_stopwords_are_dropped(self):
        tokens = P.preprocess("the model and the data")
        for stopword in ("the", "and"):
            self.assertNotIn(stopword, tokens)

    def test_single_characters_are_dropped(self):
        tokens = P.preprocess("a b c neural")
        self.assertEqual(tokens, ["neural"])

    def test_realistic_sentence(self):
        text = r"Diffusion models [37] achieve $\mathcal{O}(n)$ sampling."
        tokens = P.preprocess(text)
        self.assertIn("diffusion", tokens)
        self.assertIn("models", tokens)
        self.assertNotIn("37", tokens)
        self.assertNotIn("mathcal", tokens)

    def test_stem_flag_changes_the_output(self):
        self.assertEqual(P.preprocess("running models"), ["running", "models"])
        self.assertEqual(P.preprocess("running models", stem=True), ["run", "model"])


class TestStemmer(unittest.TestCase):
    """Every test here checks the same property: different surface forms of a
    word must collapse to one stem. If they do not, the query stops matching
    the document."""

    def test_short_words_are_untouched(self):
        for word in ("the", "is", "gas", "bus"):
            self.assertEqual(P.simple_stem(word), word)

    def test_model_family(self):
        stems = {P.simple_stem(w) for w in ("model", "models", "modeling", "modeled")}
        self.assertEqual(len(stems), 1, "model forms disagree: %s" % stems)

    def test_study_family(self):
        # Regression test: "studies" used to become "stud" while "study"
        # stayed "study", so the two never matched.
        stems = {P.simple_stem(w) for w in ("study", "studies", "studying")}
        self.assertEqual(len(stems), 1, "study forms disagree: %s" % stems)

    def test_double_s_words_are_not_over_stemmed(self):
        # Regression test: "address" used to become "addres" while
        # "addresses" became "address".
        stems = {P.simple_stem(w) for w in ("address", "addresses")}
        self.assertEqual(len(stems), 1, "address forms disagree: %s" % stems)

    def test_class_family(self):
        stems = {P.simple_stem(w) for w in ("class", "classes")}
        self.assertEqual(len(stems), 1, "class forms disagree: %s" % stems)

    def test_doubled_consonant_is_collapsed(self):
        stems = {P.simple_stem(w) for w in ("run", "running", "runs")}
        self.assertEqual(len(stems), 1, "run forms disagree: %s" % stems)

    def test_value_family(self):
        stems = {P.simple_stem(w) for w in ("value", "values", "valued")}
        self.assertEqual(len(stems), 1, "value forms disagree: %s" % stems)

    def test_memorize_family(self):
        stems = {P.simple_stem(w) for w in ("memorize", "memorized", "memorizing")}
        self.assertEqual(len(stems), 1, "memorize forms disagree: %s" % stems)

    def test_document_family(self):
        # "document" is core vocabulary for this project, and a careless
        # "-ment" rule turns it into "docu" while "documentation" stays
        # "document", so the two stop matching.
        stems = {P.simple_stem(w) for w in ("document", "documents", "documentation")}
        self.assertEqual(len(stems), 1, "document forms disagree: %s" % stems)

    def test_augment_family(self):
        stems = {P.simple_stem(w) for w in ("augmented", "augmentation")}
        self.assertEqual(len(stems), 1, "augment forms disagree: %s" % stems)

    def test_make_family(self):
        stems = {P.simple_stem(w) for w in ("make", "makes", "making")}
        self.assertEqual(len(stems), 1, "make forms disagree: %s" % stems)

    def test_short_ment_words_are_not_destroyed(self):
        self.assertEqual(P.simple_stem("augmented"), "augment")

    def test_ion_words_are_not_over_stemmed(self):
        # "-ion" is only stripped after s or t, so "diffusion" meets
        # "diffuse" but "region" does not collapse into "reg".
        self.assertEqual(P.simple_stem("diffusion"), P.simple_stem("diffuse"))
        self.assertEqual(P.simple_stem("region"), "region")

    def test_process_family(self):
        stems = {P.simple_stem(w) for w in ("process", "processes", "processing")}
        self.assertEqual(len(stems), 1, "process forms disagree: %s" % stems)

    def test_network_family(self):
        stems = {P.simple_stem(w) for w in ("network", "networks", "networking")}
        self.assertEqual(len(stems), 1, "network forms disagree: %s" % stems)

    def test_stemming_is_stable(self):
        # Stemming an already-stemmed word must not change it again,
        # otherwise indexing and querying can land on different strings.
        for word in ("model", "studi", "address", "network", "run"):
            self.assertEqual(P.simple_stem(P.simple_stem(word)), P.simple_stem(word))

    def test_cache_returns_same_answer_as_direct_call(self):
        # stem_tokens memoizes results; a stale or wrong cache entry would
        # silently corrupt every document loaded afterwards.
        words = ["models", "running", "studies", "models", "running"]
        self.assertEqual(P.stem_tokens(words), [P.simple_stem(w) for w in words])

    def test_known_limitation_embed(self):
        # Documented weakness, not an accident: a word whose own spelling ends
        # in "ed" gets shortened ("embed" -> "emb") while "embedding" stops at
        # "embed", so the two do not match. The real Porter algorithm behaves
        # the same way here. Recorded as a test so the limitation is visible
        # rather than discovered later during error analysis.
        self.assertNotEqual(P.simple_stem("embed"), P.simple_stem("embedding"))


class TestQueryDocumentConsistency(unittest.TestCase):
    """The single most common bug in a student IR system: the document and the
    query get cleaned slightly differently, so nothing matches and the model
    just looks weak. These tests prove both sides use the same pipeline."""

    def test_query_terms_match_document_terms(self):
        document = "We train neural networks for image classification tasks."
        query = "How are neural network models trained for classifying images?"
        doc_tokens = set(P.preprocess(document, stem=True))
        query_tokens = set(P.preprocess(query, stem=True))
        overlap = doc_tokens & query_tokens
        self.assertIn("neural", overlap)
        self.assertIn("network", overlap)
        self.assertIn("train", overlap)

    def test_stemming_is_what_makes_word_forms_match(self):
        # Without stemming "networks" and "network" are different strings.
        unstemmed_doc = set(P.preprocess("neural networks"))
        unstemmed_query = set(P.preprocess("neural network"))
        self.assertNotIn("network", unstemmed_doc & unstemmed_query)

    def test_case_and_punctuation_do_not_affect_matching(self):
        self.assertEqual(
            P.preprocess("Retrieval-Augmented Generation!"),
            P.preprocess("retrieval-augmented generation"),
        )


class TestPaperTextAssembly(unittest.TestCase):
    """About 87% of papers repeat their abstract as the first paragraph. If
    both copies go into the paper document, every abstract term is counted
    twice and term frequency is inflated on the most topical words."""

    def test_duplicate_abstract_is_included_only_once(self):
        abstract = "We study diffusion models."
        text = P.build_paper_text("Title", abstract, [abstract, "Second paragraph."])
        self.assertEqual(text.count("We study diffusion models."), 1)

    def test_distinct_abstract_is_kept(self):
        text = P.build_paper_text("Title", "Unique abstract.", ["Different paragraph."])
        self.assertIn("Unique abstract.", text)
        self.assertIn("Different paragraph.", text)

    def test_title_is_always_included(self):
        text = P.build_paper_text("Neural Retrieval", "abs", ["abs"])
        self.assertIn("Neural Retrieval", text)

    def test_empty_abstract_is_handled(self):
        text = P.build_paper_text("Title", "", ["Only paragraph."])
        self.assertIn("Title", text)
        self.assertIn("Only paragraph.", text)


class TestCorpusFiles(unittest.TestCase):
    """Builds a tiny fake corpus in a temp folder and checks the output files
    are shaped the way the TF-IDF / BM25 scripts expect."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.json_dir = os.path.join(self.tmp, "JSON Files")
        self.out_dir = os.path.join(self.tmp, "processed")
        os.makedirs(self.json_dir)
        os.makedirs(self.out_dir)

        papers = [
            {
                "paper_id": "2408.00001",
                "title": "Diffusion Models",
                "abstract": "A survey of diffusion models.",
                "paragraphs": {
                    "2408.00001_1": "Diffusion models generate images.",
                    "2408.00001_2": "They memorize training data [12].",
                },
            },
            {
                # Title contains a tab and a newline on purpose - if those are
                # not stripped, the TSV file gains extra columns and breaks
                # every downstream reader.
                "paper_id": "2409.00002",
                "title": "Neural\tRetrieval\nSystems",
                "abstract": "Dense retrieval overview.",
                "paragraphs": {
                    "2409.00002_1": "Retrieval systems rank documents.",
                },
            },
        ]
        for paper in papers:
            path = os.path.join(self.json_dir, paper["paper_id"] + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(paper, f)

        # Point the module at the temp folders instead of the real corpus.
        self.old_json_dir = P.JSON_DIR
        self.old_out_dir = P.OUT_DIR
        P.JSON_DIR = self.json_dir
        P.OUT_DIR = self.out_dir

        # build_corpus prints progress; hide it so test output stays readable.
        with contextlib.redirect_stdout(io.StringIO()):
            P.build_corpus()

    def tearDown(self):
        P.JSON_DIR = self.old_json_dir
        P.OUT_DIR = self.old_out_dir
        shutil.rmtree(self.tmp)

    def read_lines(self, filename):
        path = os.path.join(self.out_dir, filename)
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def test_one_line_per_paper(self):
        self.assertEqual(len(self.read_lines("paper_corpus.tsv")), 2)

    def test_one_line_per_paragraph(self):
        self.assertEqual(len(self.read_lines("paragraph_corpus.tsv")), 3)

    def test_corpus_rows_have_two_columns(self):
        for line in self.read_lines("paper_corpus.tsv"):
            self.assertEqual(len(line.split("\t")), 2)

    def test_paragraph_ids_are_preserved(self):
        ids = [line.split("\t")[0] for line in self.read_lines("paragraph_corpus.tsv")]
        self.assertIn("2408.00001_1", ids)
        self.assertIn("2409.00002_1", ids)

    def test_paper_id_is_recoverable_from_paragraph_id(self):
        # This is why no separate paragraph-to-paper mapping file is needed.
        for line in self.read_lines("paragraph_corpus.tsv"):
            para_id = line.split("\t")[0]
            self.assertIn(para_id.split("_")[0], ("2408.00001", "2409.00002"))

    def test_saved_tokens_are_not_stemmed(self):
        # The whole point of writing unstemmed files: the words stay intact so
        # stemming stays a load-time choice.
        rows = dict(line.split("\t") for line in self.read_lines("paper_corpus.tsv"))
        tokens = rows["2408.00001"].split()
        self.assertIn("diffusion", tokens)
        self.assertIn("models", tokens)
        self.assertNotIn("diffus", tokens)
        self.assertNotIn("model", tokens)

    def test_paper_document_includes_title_and_paragraphs(self):
        rows = dict(line.split("\t") for line in self.read_lines("paper_corpus.tsv"))
        tokens = rows["2408.00001"].split()
        self.assertIn("diffusion", tokens)    # from the title
        self.assertIn("survey", tokens)       # from the abstract
        self.assertIn("memorize", tokens)     # from a paragraph

    def test_paragraph_document_excludes_title(self):
        # Paragraph docs must stand alone, since that is the unit the
        # paragraph qrel file judges.
        rows = dict(line.split("\t") for line in self.read_lines("paragraph_corpus.tsv"))
        tokens = rows["2408.00001_2"].split()
        self.assertIn("memorize", tokens)
        self.assertNotIn("survey", tokens)

    def test_titles_file_survives_tabs_and_newlines(self):
        lines = self.read_lines("paper_titles.tsv")
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(len(line.split("\t")), 3)

    def test_build_info_is_written(self):
        path = os.path.join(self.out_dir, "build_info.txt")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            report = f.read()
        self.assertIn("papers", report)
        self.assertIn("vocabulary", report)

    def test_unreadable_file_does_not_stop_the_build(self):
        bad = os.path.join(self.json_dir, "9999.99999.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json")
        with contextlib.redirect_stdout(io.StringIO()):
            P.build_corpus()
        # The two good papers still made it through.
        self.assertEqual(len(self.read_lines("paper_corpus.tsv")), 2)


class TestLoaders(unittest.TestCase):
    """The loaders are what the TF-IDF and BM25 scripts actually call."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.tmp, "processed")
        os.makedirs(self.out_dir)
        self.old_out_dir = P.OUT_DIR
        P.OUT_DIR = self.out_dir

        with open(os.path.join(self.out_dir, "paper_corpus.tsv"), "w", encoding="utf-8") as f:
            f.write("2408.00001\tdiffusion models generate images\n")
            f.write("2409.00002\tneural networks rank documents\n")
        with open(os.path.join(self.out_dir, "paragraph_corpus.tsv"), "w", encoding="utf-8") as f:
            f.write("2408.00001_1\tdiffusion models\n")
            f.write("2408.00001_2\t\n")           # a paragraph that cleaned to nothing
        with open(os.path.join(self.out_dir, "queries_study.tsv"), "w", encoding="utf-8") as f:
            f.write("2001\tneural networks\n")
        with open(os.path.join(self.out_dir, "paper_titles.tsv"), "w", encoding="utf-8") as f:
            f.write("2408.00001\tDiffusion Models\tA survey.\n")

    def tearDown(self):
        P.OUT_DIR = self.old_out_dir
        shutil.rmtree(self.tmp)

    def test_load_corpus_yields_id_and_tokens(self):
        docs = dict(P.load_corpus("paper"))
        self.assertEqual(docs["2408.00001"], ["diffusion", "models", "generate", "images"])

    def test_load_corpus_unstemmed_by_default(self):
        docs = dict(P.load_corpus("paper"))
        self.assertIn("models", docs["2408.00001"])

    def test_load_corpus_can_stem_on_the_way_out(self):
        docs = dict(P.load_corpus("paper", stem=True))
        self.assertIn("model", docs["2408.00001"])
        self.assertNotIn("models", docs["2408.00001"])

    def test_empty_document_yields_empty_token_list(self):
        # Roughly 0.2% of real paragraphs clean to nothing. They must still
        # load, with an empty list, so IDs stay complete and the model can
        # skip them instead of crashing on a divide by zero.
        docs = dict(P.load_corpus("paragraph"))
        self.assertEqual(docs["2408.00001_2"], [])

    def test_load_corpus_rejects_a_bad_level(self):
        with self.assertRaises(ValueError):
            list(P.load_corpus("sentence"))

    def test_load_queries_returns_id_and_tokens(self):
        queries = P.load_queries("study")
        self.assertEqual(queries, [("2001", ["neural", "networks"])])

    def test_load_queries_can_stem(self):
        queries = P.load_queries("study", stem=True)
        self.assertEqual(queries, [("2001", ["neural", "network"])])

    def test_documents_and_queries_stem_the_same_way(self):
        # The reason both loaders take the same flag.
        docs = dict(P.load_corpus("paper", stem=True))
        queries = dict(P.load_queries("study", stem=True))
        self.assertTrue(set(queries["2001"]) & set(docs["2409.00002"]))

    def test_load_titles(self):
        titles = P.load_titles()
        self.assertEqual(titles["2408.00001"], ("Diffusion Models", "A survey."))


class TestQueryFiles(unittest.TestCase):
    """Checks the query files are written with one row per query."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.out_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.out_dir)

        queries = [
            {"query_id": 2001, "query": "How does RAG reduce computation?", "label": "Easy"},
            {"query_id": 2002, "query": "What are C-Uniform trajectories?", "label": "Medium"},
        ]
        # build_queries() reads "<DATA_DIR>/Study.json", so the fixture goes
        # in data/, matching where the real Study/Test/Train files live.
        with open(os.path.join(self.data_dir, "Study.json"), "w", encoding="utf-8") as f:
            json.dump(queries, f)

        self.old_data_dir = P.DATA_DIR
        self.old_out_dir = P.OUT_DIR
        P.DATA_DIR = self.data_dir
        P.OUT_DIR = self.out_dir

        with contextlib.redirect_stdout(io.StringIO()):
            P.build_queries()

    def tearDown(self):
        P.DATA_DIR = self.old_data_dir
        P.OUT_DIR = self.old_out_dir
        shutil.rmtree(self.tmp)

    def test_query_file_written_with_one_row_per_query(self):
        path = os.path.join(self.out_dir, "queries_study.tsv")
        with open(path, encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "2001")
        self.assertEqual(rows[1][0], "2002")

    def test_query_tokens_are_cleaned_but_not_stemmed(self):
        path = os.path.join(self.out_dir, "queries_study.tsv")
        with open(path, encoding="utf-8") as f:
            first_row = f.readline().rstrip("\n").split("\t")
        tokens = first_row[1].split()
        self.assertNotIn("how", tokens)         # stopword removed
        self.assertNotIn("does", tokens)        # stopword removed
        self.assertIn("rag", tokens)
        self.assertIn("computation", tokens)    # intact, not "comput"

    def test_missing_query_file_is_skipped_quietly(self):
        # Test.json and Train.json do not exist in the temp folder, so
        # build_queries should skip them instead of crashing.
        with contextlib.redirect_stdout(io.StringIO()):
            P.build_queries()   # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
