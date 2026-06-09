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

The naive reading of this table is wrong, and understanding why is the actual result.

1. **RAG is the clear win for real question answering.** It is the only thing that lets the system cite (precision 0 to 1.00) and ground its answers (NLI faithfulness 0 to 0.28). Without retrieval the model can only abstain.

2. **The judge ranking is confounded, and that is the main lesson.** For a question with no context, the base model correctly refuses to answer, and the judge rewards the honest refusal with full marks. So the high "base" score is mostly reward for declining, not for good answers. On top of that the judge is the base model, so it has a self preference for base style outputs (the circularity I flagged when choosing a local judge). The judge column cannot be read as a quality ranking across RAG and non RAG cells.

3. **Fine tuning changed format, not knowledge.** The adapter was trained on terse, extractive QASPER reference answers, and it pushed outputs toward that style (ROUGE rises, highest at FT+RAG). But that terseness lowered judged completeness and grounding. So this adapter traded verbosity for reference similarity rather than improving answers. That lines up with the going in hypothesis: RAG carries the system, fine tuning mostly shifts style.

This is a more useful result than a clean "fine tuning plus RAG wins", because it surfaced two real evaluation pitfalls (abstention reward and judge self preference) and showed concretely that fine tuning data style propagates straight into the outputs.

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

The biggest lesson was that the evaluation is harder and more important than the pipeline. Building hybrid retrieval with reranking and HyDE was the easy part. Getting trustworthy numbers, catching the prompt bug where the model invented page numbers, realizing the faithfulness metric was too strict, and understanding why the judge ranked an abstaining model highest, that is where I actually learned how these systems should be measured.
