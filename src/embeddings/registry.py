# src/embeddings/registry.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Any, Callable

@dataclass
class EmbeddingConfig:
    """
    Configuration for embeddings.
    """
    provider: str = "openai"                 # fixed to 'openai' in this build
    model: str = "text-embedding-3-small"    # or 'text-embedding-3-large'
    batch_size: int = 512                    # tune if you hit 413/429
    normalize: bool = True                   # L2-normalize vectors
    truncate_tokens: Optional[int] = 8192    # safety cap; model supports up to 8192

@dataclass
class RunParams:
    """
    Parameters controlling a single embedding run.
    """
    limit: Optional[int] = None         # max number of articles
    since_hours: Optional[float] = None # restrict to recent articles
    progress_cb: Optional[Callable] = None  # callback for progress reporting

    # Input source: full article text or summary
    embed_source: str = "article"       # "article" or "summary"

    # How to handle long texts
    long_text_mode: str = "truncate"    # "truncate" or "chunk"
    max_tokens: int = 7500              # for truncate mode
    chunk_tokens: int = 2000            # for chunk mode
    overlap_tokens: int = 200           # overlap between chunks


def _l2_normalize(v):
    import numpy as np
    v = np.asarray(v, dtype="float32")
    n = (v ** 2).sum(axis=1, keepdims=True) ** 0.5
    n[n == 0] = 1.0
    return v / n


class OpenAIEmbedder:
    """
    Thin wrapper around OpenAI Embeddings API with:
      - optional token-based truncation
      - batching
      - simple retry/backoff for 429/5xx
      - optional L2 normalization
    """

    def __init__(self, cfg: EmbeddingConfig):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = cfg.model
        self.batch_size = max(1, int(cfg.batch_size))
        self.normalize = bool(cfg.normalize)
        self.truncate_tokens = cfg.truncate_tokens

        # Optional tokenizer for safe truncation
        try:
            import tiktoken  # type: ignore
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None  # truncation will be skipped

    def _truncate(self, texts: List[str]) -> List[str]:
        if not self.truncate_tokens or self._enc is None:
            return texts
        enc = self._enc
        limit = int(self.truncate_tokens)
        out: List[str] = []
        for t in texts:
            toks = enc.encode(t)
            out.append(t if len(toks) <= limit else enc.decode(toks[:limit]))
        return out

    def embed(self, texts: List[str]):
        """
        Returns an (N, D) numpy array of float32 embeddings.
        """
        import numpy as np

        if not texts:
            # Reasonable default D for text-embedding-3-small; avoids shape errors on empty calls
            return np.zeros((0, 1536), dtype="float32")

        texts = self._truncate(texts)

        out_chunks = []
        i = 0
        # Simple exponential backoff parameters
        backoff = 1.5
        delay = 1.0
        max_delay = 20.0

        while i < len(texts):
            chunk = texts[i : i + self.batch_size]
            try:
                resp = self.client.embeddings.create(model=self.model, input=chunk)
                # Infer dimension from first embedding to avoid hardcoding
                mat = np.array([d.embedding for d in resp.data], dtype="float32")
                out_chunks.append(mat)
                i += len(chunk)
                # reset delay on success
                delay = 1.0
            except Exception as e:
                msg = str(e).lower()
                # Retry on typical transient errors
                if any(key in msg for key in ("429", "rate limit", "temporarily unavailable", "500", "502", "503", "504")):
                    time.sleep(delay)
                    delay = min(max_delay, delay * backoff)
                    continue
                # Non-retryable
                raise

        import numpy as np
        embs = np.vstack(out_chunks) if out_chunks else np.zeros((0, 1536), dtype="float32")
        return _l2_normalize(embs) if self.normalize else embs


def get_embedder(cfg: EmbeddingConfig) -> OpenAIEmbedder:
    if cfg.provider != "openai":
        raise ValueError("This build only supports provider='openai'.")
    return OpenAIEmbedder(cfg)


# ---------------------------
# LangChain adapter helpers
# ---------------------------

def from_langchain_embedding(lc_embedding: Any) -> EmbeddingConfig:
    """
    Convert a LangChain embedding instance into our EmbeddingConfig.
    Currently supports langchain_openai.OpenAIEmbeddings.
    """
    cls = lc_embedding.__class__.__name__.lower()

    # langchain_openai.OpenAIEmbeddings
    if "openai" in cls and "embedding" in cls:
        # LC exposes `model`
        model = getattr(lc_embedding, "model", None) or "text-embedding-3-small"
        # LC usually doesn't have a standardized batch_size; keep our default (overridable via config)
        return EmbeddingConfig(
            provider="openai",
            model=model,
            batch_size=512,
            normalize=True,
        )

    raise ValueError(f"Unsupported LangChain embedding class: {lc_embedding.__class__.__name__}")


def to_langchain_embedding(cfg: EmbeddingConfig) -> Any:
    """
    Optional: Instantiate a LangChain embedding object from our EmbeddingConfig.
    Currently returns langchain_openai.OpenAIEmbeddings for provider='openai'.
    """
    if cfg.provider != "openai":
        raise ValueError("to_langchain_embedding only supports provider='openai' in this build.")
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as e:
        raise ImportError(
            "langchain-openai is not installed. Install with: `poetry add langchain-openai`."
        ) from e
    return OpenAIEmbeddings(model=cfg.model)


__all__ = [
    "EmbeddingConfig",
    "OpenAIEmbedder",
    "get_embedder",
    "from_langchain_embedding",
    "to_langchain_embedding",
]
