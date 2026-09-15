/**
 * Mock corpus. Replace nothing here when you go live — swap the transport in
 * searchClient.js instead. This file exists only so the UI has something to rank.
 */

export const DOCUMENTS = [
  {
    id: 'd1',
    title: 'Dense Passage Retrieval for Open-Domain Question Answering',
    snippet:
      'We show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework.',
    authors: 'Karpukhin et al.',
    venue: 'EMNLP',
    year: 2020,
    docType: 'paper',
    score: 0.94,
    url: 'https://arxiv.org/abs/2004.04906',
  },
  {
    id: 'd2',
    title: 'ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction',
    snippet:
      'ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT, then employs a cheap yet powerful interaction step that models their fine-grained similarity.',
    authors: 'Khattab and Zaharia',
    venue: 'SIGIR',
    year: 2020,
    docType: 'paper',
    score: 0.91,
    url: 'https://arxiv.org/abs/2004.12832',
  },
  {
    id: 'd3',
    title: 'The Probabilistic Relevance Framework: BM25 and Beyond',
    snippet:
      'We present the probabilistic relevance model and the development of the BM25 family of ranking functions, with attention to term frequency saturation and document length normalisation.',
    authors: 'Robertson and Zaragoza',
    venue: 'Foundations and Trends in IR',
    year: 2009,
    docType: 'paper',
    score: 0.88,
    url: 'https://dl.acm.org/doi/10.1561/1500000019',
  },
  {
    id: 'd4',
    title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    snippet:
      'We explore models which combine pre-trained parametric and non-parametric memory for language generation, retrieving passages with a dense retriever and conditioning a seq2seq generator on them.',
    authors: 'Lewis et al.',
    venue: 'NeurIPS',
    year: 2020,
    docType: 'paper',
    score: 0.86,
    url: 'https://arxiv.org/abs/2005.11401',
  },
  {
    id: 'd5',
    title: 'MS MARCO: A Human Generated Machine Reading Comprehension Dataset',
    snippet:
      'A large-scale dataset of anonymized real queries sampled from Bing, paired with human generated answers and passages, now a standard benchmark for passage ranking evaluation.',
    authors: 'Bajaj et al.',
    venue: 'NIPS Workshop',
    year: 2016,
    docType: 'dataset',
    score: 0.81,
    url: 'https://microsoft.github.io/msmarco/',
  },
  {
    id: 'd6',
    title: 'Pyserini: A Python Toolkit for Reproducible Information Retrieval Research',
    snippet:
      'Pyserini provides sparse and dense retrieval over a common interface, supporting reproducible baselines for first-stage retrieval on standard test collections.',
    authors: 'Lin et al.',
    venue: 'SIGIR',
    year: 2021,
    docType: 'code',
    score: 0.79,
    url: 'https://github.com/castorini/pyserini',
  },
  {
    id: 'd7',
    title: 'SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking',
    snippet:
      'We propose a sparse model that learns term weighting and expansion jointly, producing representations that remain compatible with an inverted index while improving effectiveness.',
    authors: 'Formal et al.',
    venue: 'SIGIR',
    year: 2021,
    docType: 'paper',
    score: 0.77,
    url: 'https://arxiv.org/abs/2107.05720',
  },
  {
    id: 'd8',
    title: 'Learning to Rank for Information Retrieval',
    snippet:
      'A survey of pointwise, pairwise, and listwise approaches to learning ranking functions, with a treatment of evaluation measures and their optimisation characteristics.',
    authors: 'Liu',
    venue: 'Foundations and Trends in IR',
    year: 2009,
    docType: 'paper',
    score: 0.74,
    url: 'https://link.springer.com/book/10.1007/978-3-642-14267-3',
  },
  {
    id: 'd9',
    title: 'Overview of the TREC Deep Learning Track',
    snippet:
      'The track studies ad hoc ranking in a setting with large training sets, comparing traditional retrieval baselines against neural methods across document and passage ranking tasks.',
    authors: 'Craswell et al.',
    venue: 'TREC',
    year: 2022,
    docType: 'paper',
    score: 0.71,
    url: 'https://trec.nist.gov/',
  },
  {
    id: 'd10',
    title: 'Query Expansion Techniques for Sparse and Dense Retrieval: A Comparative Study',
    snippet:
      'We compare pseudo-relevance feedback against generative query expansion, measuring the effect of each on recall at the first stage and on end-to-end ranking quality.',
    authors: 'Nogueira and Cho',
    venue: 'ECIR',
    year: 2023,
    docType: 'paper',
    score: 0.68,
    url: 'https://arxiv.org/',
  },
  {
    id: 'd11',
    title: 'Evaluating Interactive Search Interfaces: A User Study Protocol',
    snippet:
      'A protocol for measuring task completion, query reformulation rate, and perceived control in interactive retrieval systems, with instruments for post-task questionnaires.',
    authors: 'Kelly',
    venue: 'CHIIR',
    year: 2019,
    docType: 'thesis',
    score: 0.64,
    url: 'https://dl.acm.org/',
  },
  {
    id: 'd12',
    title: 'Index Compression for Low-Latency Retrieval at Scale',
    snippet:
      'We describe posting list compression schemes and their effect on query latency, showing the tradeoff between decompression cost and reduced memory traffic.',
    authors: 'Moffat and Stuiver',
    venue: 'Information Retrieval Journal',
    year: 2018,
    docType: 'paper',
    score: 0.59,
    url: 'https://link.springer.com/journal/10791',
  },
];

/** Query suggestions offered as inline completions and in the dropdown. */
export const SUGGESTIONS = [
  'recent work in information retrieval',
  'dense retrieval vs bm25',
  'learned sparse retrieval',
  'retrieval augmented generation',
  'query expansion techniques',
  'index compression latency',
  'evaluating search interfaces',
  'passage ranking benchmarks',
  'late interaction models',
  'pseudo relevance feedback',
];

/** Naive term-overlap ranking. Good enough to make the UI feel alive. */
export function rankDocuments(query) {
  const terms = query.toLowerCase().split(/\W+/).filter((t) => t.length > 2);
  if (terms.length === 0) return [];

  return DOCUMENTS.map((doc) => {
    const haystack = `${doc.title} ${doc.snippet} ${doc.authors} ${doc.venue}`.toLowerCase();
    const hits = terms.filter((t) => haystack.includes(t)).length;
    const overlap = hits / terms.length;
    return { ...doc, score: Number((doc.score * (0.35 + 0.65 * overlap)).toFixed(3)) };
  })
    .filter((doc) => doc.score > 0.3)
    .sort((a, b) => b.score - a.score);
}
