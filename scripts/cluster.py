# scripts/cluster.py
import argparse, yaml
from src.clustering.clusterer import ClusterParams, run_clustering
from scripts._bootstrap_env import load_env

load_env()  # now OPENAI_API_KEY from .env is available

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--window-hours", type=int, default=24)
    ap.add_argument("--min-k", type=int, default=5)
    ap.add_argument("--max-k", type=int, default=25)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    c = cfg.get("clustering", {})
    p = ClusterParams(
        window_hours=args.window_hours or c.get("window_hours", 24),
        min_k=args.min_k or c.get("min_k", 5),
        max_k=args.max_k or c.get("max_k", 25),
    )
    run_id = run_clustering(p)
    print(f"✅ Cluster run complete. run_id={run_id}")

if __name__ == "__main__":
    main()
