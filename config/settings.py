"""
Central configuration for the whole system.

Everything tunable lives here so experiments stay reproducible: change a
value, note it in the run log, compare metrics. Values can be overridden
with environment variables or a .env file (see .env.example), which is
handy for flipping things like USE_HYDE on and off during ablations.

A note on the GPU budget: this was built for a 12GB card (RTX 4070 Ti)
where Windows/WSL already eats a few GB, so the realistic budget is around
7.5 to 8GB. The LLM in 4-bit takes most of that, which means we can't
naively keep the embedder and reranker on the GPU at the same time as the
model. The *_device fields exist so we can place each component
deliberately, and low_vram_mode is a quick switch to push the retrieval
models onto CPU when the LLM is resident.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file (config/settings.py -> repo root).
ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----- Paths -----
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_pdf_dir: Path = ROOT_DIR / "data" / "raw_pdfs"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    chroma_dir: Path = ROOT_DIR / "chroma_db"
    log_dir: Path = ROOT_DIR / "logs"

    # ----- LLM -----
    # Qwen3-8B is a hybrid thinking model. enable_thinking controls whether
    # the chat template emits <think> blocks. We default it off for the QA
    # use case so answers are clean and easier to benchmark, but it's exposed
    # here because the fine-tuning data has to be formatted consistently with
    # whatever we pick.
    llm_model: str = "Qwen/Qwen3-8B"
    llm_device: str = "cuda"
    load_in_4bit: bool = True
    enable_thinking: bool = False
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9

    # Optional fine-tuned adapter. Empty means use the base model.
    lora_adapter_path: str = ""

    # ----- Embeddings -----
    # bge-m3 needs no special query prefix (unlike bge-large-en-v1.5), which
    # is one reason it's a nicer default. Keeping the field anyway in case we
    # swap models.
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_query_instruction: str = ""
    embedding_batch_size: int = 16
    # Chroma must be told to use cosine explicitly; its default is L2.
    distance_metric: str = "cosine"

    # ----- Reranker -----
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"

    # When the LLM is loaded for serving, flip this to keep the embedder and
    # reranker on CPU and avoid an out-of-memory on the 12GB card.
    low_vram_mode: bool = False

    # ----- Chunking -----
    # Sizes are in TOKENS, not characters. The original plan's "800" was
    # really characters (~200 tokens); measuring in tokens makes the
    # chunk-size sweep meaningful. Uses the embedding model's tokenizer.
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ----- Retrieval -----
    use_hyde: bool = True          # ablation switch, not assumed to help
    use_hybrid: bool = True        # vector + BM25, fused with RRF
    use_reranker: bool = True
    vector_top_k: int = 30         # recall-oriented first pass
    bm25_top_k: int = 30
    rrf_k: int = 60                # standard RRF constant
    rerank_pool_size: int = 20     # how many fused candidates the reranker scores
    rerank_max_length: int = 512   # cross-encoder truncation length
    rerank_top_n: int = 6          # precision-oriented final context

    # ----- GROBID (PDF parsing) -----
    grobid_url: str = "http://localhost:8070"
    grobid_timeout_s: int = 60
    # If GROBID is unreachable, fall back to PyMuPDF instead of failing.
    grobid_fallback_pymupdf: bool = True

    # ----- API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_question_chars: int = 2000

    # ----- LLM judge (offline eval only) -----
    judge_model: str = ""          # locked later; the one external API
    judge_api_key: str = Field(default="", repr=False)

    def ensure_dirs(self) -> None:
        """Create the local directories the app writes to."""
        for d in (self.data_dir, self.raw_pdf_dir, self.processed_dir,
                  self.chroma_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


# Single shared instance imported across the codebase.
settings = Settings()
