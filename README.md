# Scientific RAG

A retrieval augmented question answering system for scientific papers, with citations, an evaluation suite, and a QLoRA fine tuned variant of Qwen3-8B. It runs locally on a single 12 GB GPU.

I built this to get hands on experience with the things I kept seeing in ML and LLM engineering roles: retrieval engineering, vector search, reranking, hallucination measurement, fine tuning, and orchestration. The goal was not to ship a product. It was to build each component properly, measure whether it actually helps, and be honest about what I found. A lot of the value here is in the evaluation, not just the pipeline.

## What it does

You give it scientific papers (PDFs) and ask questions. It retrieves the relevant passages, generates an answer grounded in them, and attaches citations that point back to the source so you can check the claims. Every retrieval and generation choice is exposed as a switch so I can turn it on and off and measure the difference.

Main features:

- PDF parsing that recovers sections and bibliographic metadata
- Hybrid retrieval: dense vectors (bge-m3) plus BM25, combined with Reciprocal Rank Fusion
- A cross encoder reranker (bge-reranker-v2-m3) for precision
- HyDE query expansion behind a switch
- Citation enforced answers from Qwen3-8B, with the citation tags checked against the retrieved context
- An evaluation suite: retrieval metrics, citation precision, faithfulness, and an LLM judge
- A QLoRA fine tuned adapter and a 2x2 experiment comparing fine tuning against retrieval
- A FastAPI service with `/chat`, `/ingest`, and `/health`

## Architecture

```
question
   -> HyDE expansion (optional, generates a hypothetical answer to search with)
   -> hybrid retrieval (vector + BM25)
   -> Reciprocal Rank Fusion
   -> cross encoder rerank
   -> prompt builder (context + citation contract)
   -> Qwen3-8B (4 bit, optional LoRA adapter)
   -> answer with checked citations
```

LangChain wires the pipeline together. To be precise about it, this is LLM pipeline orchestration, a fixed sequence of steps, not autonomous agents or tool calling. I left those out on purpose to keep the focus on retrieval quality.

## The stack, and why

These are the choices I made and the honest reasoning behind them.

**LLM: Qwen3-8B.** Strong open weights model that fits a 12 GB card in 4 bit (NF4) quantization, around 5.8 GB resident. It is a hybrid thinking model, but I run it with thinking off for clean, easy to benchmark answers. The original plan said "Qwen3-7B", which does not exist (the 7B was Qwen2.5), so I corrected it to Qwen3-8B.

**Embeddings: bge-m3.** A widely used, well tested retrieval model with good results on scientific text. It needs no special query prefix, which is one less thing to get wrong.

**Reranker: bge-reranker-v2-m3.** A cross encoder reads the query and a candidate together, so it scores relevance far better than the bi encoder embedder. Too slow to run over the whole corpus, so it only sees the fused top candidates.

**Vector store: ChromaDB.** Simple local setup and good enough for studying retrieval. I set it to cosine space explicitly, since Chroma defaults to L2 and the "cosine" claim would otherwise be wrong.

**Parsing: GROBID, with a PyMuPDF fallback.** This was the part I underestimated at first. Scientific PDFs are multi column with equations and reference lists, and raw text extraction reads in the wrong order and caps retrieval quality no matter how good the reranker is. GROBID turns a PDF into structured sections plus clean metadata (title, authors, year), which is also where the citation keys come from. When GROBID is not running, it falls back to PyMuPDF so nothing breaks.

**Fine tuning: QLoRA** (PEFT, TRL, BitsAndBytes 4 bit). The whole point is that it fits the 12 GB budget. Training peaks around 9.8 GB.

A few things I deliberately left out, since they did not serve the learning goals: vLLM/TGI, Elasticsearch, streaming, auth, and production monitoring. I treated the LLM judge as a local model too, so the project has no external API dependencies at all.

## Results

### Retrieval ablation

Run on 80 QASPER papers (1462 chunks, about 250 questions). Gold labels come from QASPER's evidence annotations, aligned to my chunks with word 5-gram overlap (100% of answerable questions got a gold chunk).

| config | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| vector | 0.116 | 0.271 | 0.366 | 0.270 | 0.259 |
| hybrid | 0.100 | 0.324 | 0.396 | 0.273 | 0.266 |
| hybrid + rerank | 0.116 | 0.342 | 0.421 | 0.299 | 0.295 |
| hybrid + rerank + HyDE | 0.116 | 0.347 | 0.432 | 0.299 | 0.297 |

Recall improves at every stage (recall@10 goes from 0.366 to 0.432). The reranker gives the biggest precision jump (MRR and nDCG), which is exactly what a cross encoder is for. HyDE adds a small recall gain and does not help MRR, which matches the honest expectation that it is not a guaranteed win. The absolute numbers are low because retrieving from 1462 chunks across many papers is genuinely hard, and that is the realistic setting.

### Fine tuning vs RAG (2x2)

Run on 25 QASPER questions. The judge is the base model scoring all cells.

| cell | judge overall | citation precision | NLI faithfulness | ROUGE-L |
|---|---|---|---|---|
| base | 4.96 | 0.00 | 0.00 | 0.041 |
| base + RAG | 4.54 | 1.00 | 0.28 | 0.101 |
| FT | 3.77 | 0.00 | 0.00 | 0.076 |
| FT + RAG | 3.18 | 1.00 | 0.13 | 0.142 |

At first glance this says the base model with no retrieval is the best, which is not what is actually going on, so it needs some explaining.

For the base and FT rows, the judge is mostly rewarding the model for not answering. With no context the model says it cannot answer the question, and the judge scores that highly because an honest refusal is technically faithful and well formatted. So those high numbers are not really about answer quality. It also does not help that the judge is the same base model, so it leans toward answers that sound like its own. That is the circularity I expected when I picked a local judge instead of an external one. Between those two things, I would not read the judge column as a ranking across the RAG and no RAG cells.

What I do trust here is the citation and grounding columns. Only the RAG cells have retrieved context to cite, and they go from nothing usable to perfect citation precision, with NLI faithfulness moving the same way. That is the part I actually care about and it is a clear point for retrieval.

The fine tuning result was a bit of a letdown. The adapter picked up the style of the QASPER reference answers, which are short and extractive, so its answers got terser and the ROUGE score went up because they look more like the references. But the shorter answers also scored lower on completeness and grounding, so it did not really make the answers better, it mostly made them shorter. That is roughly what I expected, that fine tuning on this kind of data teaches style and not facts, I just hoped it would help a bit more than it did.

So retrieval is doing the real work here and the fine tuning mostly changed how the answers read. It is not the tidy "everything I added helped" story I would have liked, but figuring out why the judge was misleading and watching the training data style leak straight into the answers taught me more than a clean result would have.

### A note on the faithfulness metric

The NLI based faithfulness is a strict lower bound. It checks each claim against each chunk and under credits true claims that combine facts from several chunks or paraphrase heavily, so it reads lower than the answers actually deserve. I kept it as a conservative signal and treat the LLM judge's faithfulness as the headline. A per claim LLM verifier would be more accurate if I revisit this.

### Hardware reality

On a 12 GB card with Windows already using a few GB, the LLM and the reranker do not both fit on the GPU at once. For evaluation this is a non issue because no LLM is loaded during retrieval, so the reranker runs on GPU and is fast. For live `/chat` the reranker falls back to CPU and a full answer takes a few minutes. I kept the reranker on for live chat and accept the wait. The evaluation runners are phased so only one heavy model is resident at a time.

## Running it

Setup. This assumes a working CUDA PyTorch install (I reused an existing torch 2.7.1+cu126, so it is intentionally not pinned in requirements).

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install -r requirements.txt
```

Start GROBID (Docker), then ingest some PDFs:

```bash
./scripts/start_grobid.sh
PYTHONPATH=. .venv/bin/python scripts/ingest.py data/raw_pdfs --reset
```

Serve the API:

```bash
PYTHONPATH=. .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
# docs at http://localhost:8000/docs
```

Ask a question:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "How does multi-head attention work?"}'
```

Evaluation (offline, no API needed):

```bash
PYTHONPATH=. .venv/bin/python evaluation/run_retrieval_eval.py --max-papers 80
PYTHONPATH=. .venv/bin/python evaluation/run_generation_eval.py --max-questions 30
PYTHONPATH=. .venv/bin/python evaluation/run_2x2_comparison.py --max-questions 25
```

Fine tuning:

```bash
.venv/bin/python -m pip install -r requirements-train.txt
PYTHONPATH=. .venv/bin/python training/qlora/train.py --smoke   # quick VRAM check
PYTHONPATH=. .venv/bin/python training/qlora/train.py           # full run
```

Tests:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

## Project layout

```
api/          FastAPI service (routes, schemas, deps)
rag/          ingestion, retrieval, reranking, generation, prompts, pipeline
training/     SFT dataset builders and the QLoRA trainer
evaluation/   QASPER loader, gold labels, metrics, eval runners
config/       central settings
scripts/      ingest, GROBID startup, model smoke tests
tests/        unit tests for the pure logic
```

## What I would do next

- A neutral judge model (not the model under test) to remove the self preference confound, and an evaluation that separates "correctly abstained" from "answered well".
- Better fine tuning targets. The terse QASPER references taught terseness. Self distilling from good base plus RAG answers would likely make fine tuning helpful instead of neutral.
- Treat parsing quality as its own measured component, since it sets the ceiling for everything downstream.

## What I took away from it

Honestly, the pipeline itself was the easy part. Hybrid retrieval, the reranker, HyDE, those are mostly a case of connecting the right pieces. What actually took thought was the evaluation. Almost all of the bugs and surprises came from trying to get numbers I could trust: the prompt issue where the model made up page numbers, the faithfulness metric being too strict, and the judge scoring a model that refused to answer above one that answered properly. I went in expecting to spend my time on retrieval and ended up spending most of it on measurement, and that is probably where I learned the most.
