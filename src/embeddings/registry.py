# src/embeddings/registry.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EmbeddingConfig:
    """
    Configuration for OpenAI embeddings.
    """
    provider: str = "openai"                 # fixed to 'openai' in this build
    model: str = "text-embedding-3-small"    # or 'text-embedding-3-large'
    batch_size: int = 512                    # tune if you hit 413/429
    normalize: bool = True                   # L2-normalize vectors
    truncate_tokens: Optional[int] = 8192    # safety cap; model supports up to 8192


def _l2_normalize(v):
    import numpy as np
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
            if len(toks) <= limit:
                out.append(t)
            else:
                out.append(enc.decode(toks[:limit]))
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

        embs = np.vstack(out_chunks) if out_chunks else np.zeros((0, 1536), dtype="float32")
        return _l2_normalize(embs) if self.normalize else embs


def get_embedder(cfg: EmbeddingConfig) -> OpenAIEmbedder:
    if cfg.provider != "openai":
        raise ValueError("This build only supports provider='openai'.")
    return OpenAIEmbedder(cfg)
