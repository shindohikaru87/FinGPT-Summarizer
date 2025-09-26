# scripts/embed.py
import argparse, yaml, time, sys
from typing import Optional
from scripts._bootstrap_env import load_env

# LangChain OpenAI embeddings
from langchain_openai import OpenAIEmbeddings

# Your registry + service
from src.embeddings.registry import EmbeddingConfig, from_langchain_embedding
from src.embeddings.service import run_embeddings, RunParams

load_env()


class Progress:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.start = time.perf_counter()
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.last_render_len = 0
        self.latencies = []

    def tick(self, status: str, dt: float | None = None):
        self.done += 1
        if status == "ok":
            self.ok += 1
        elif status == "fail":
            self.failed += 1
        elif status == "skip":
            self.skipped += 1
        if dt is not None:
            self.latencies.append(dt)
        self.render()

    def render(self, final: bool = False):
        pct = (100.0 * self.done / self.total)
        avg = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        remaining = self.total - self.done
        eta = remaining * avg
        bar_len = 24
        filled = int(bar_len * self.done / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        line = (
            f"\r[{bar}] {pct:6.2f}%  "
            f"done={self.done}/{self.total}  ok={self.ok}  fail={self.failed}  skip={self.skipped}  "
            f"avg={avg:0.2f}s  ETA={eta:0.0f}s"
        )
        pad = max(0, self.last_render_len - len(line))
        sys.stdout.write(line + (" " * pad))
        sys.stdout.flush()
        self.last_render_len = len(line)
        if final:
            sys.stdout.write("\n")

    def finish(self):
        self.render(final=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--since-hours", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    e = (cfg or {}).get("embeddings", {}) or {}

    # 1) Build LangChain embedder from config
    #    (uses OPENAI_API_KEY from env; falls back to OpenAI default if not set)
    model_name = e.get("model", "text-embedding-3-small")
    lc = OpenAIEmbeddings(model=model_name)

    # 2) Convert LangChain embedder -> our EmbeddingConfig
    ecfg: EmbeddingConfig = from_langchain_embedding(lc)

    # 3) Allow config.yaml to override batch_size/normalize after conversion
    if "batch_size" in e:
        ecfg.batch_size = int(e["batch_size"])
    if "normalize" in e:
        ecfg.normalize = bool(e["normalize"])

    print(
        f"Running embeddings via LangChain(OpenAI) → EmbeddingConfig("
        f"provider={ecfg.provider}, model={ecfg.model}, "
        f"batch_size={ecfg.batch_size}, normalize={ecfg.normalize}) "
        f"for up to {args.limit} articles..."
    )

    prog = Progress(total=args.limit)

    def progress_cb(status: str, dt: float | None = None):
        prog.tick(status, dt)

    n = run_embeddings(
        ecfg,
        RunParams(limit=args.limit, since_hours=args.since_hours, progress_cb=progress_cb)
    )

    prog.finish()
    print(f"\n Embedded {n} articles.\n")


if __name__ == "__main__":
    main()
