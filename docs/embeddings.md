# 🧩 Embeddings in FinGPT Summarizer

This document explains:
1. What `scripts/embed.py` does.  
2. Why this embedding job is necessary.  
3. Why embeddings are important for search.  
4. A deep-dive on **embedding similarity scoring** with math and examples.

---

## 1. What `scripts/embed.py` Does (and Why It’s Needed)

`scripts/embed.py` is the **embedding pre-compute job**. It transforms raw article text into numeric vectors and stores them so search can be fast, cheap, and consistent.

### Responsibilities (mapped to code)
- **Bootstraps environment**  
  Loads API keys/DB URLs from `scripts/_bootstrap_env.py`.

- **CLI arguments**  
  ```bash
  --config        # path to config/config.yaml
  --limit         # cap how many articles to process in this run
  --since-hours   # only embed articles updated in the last N hours
  ```

- **Reads config**  
  Loads embedding settings from YAML (`model`, `batch_size`, `normalize`).

- **Progress tracking**  
  Displays a progress bar with counts of ok/fail/skip and average latency.

- **Run embedding service**  
  Calls `run_embeddings(...)` with:
  1. Candidate article selection.  
  2. Text assembly (title + summary).  
  3. Batch calls to the embedding model.  
  4. Normalization (optional).  
  5. Upsert into `embeddings` table.  
  6. Progress callback for live stats.  

### Why it’s necessary
- **Speed**: Search embeds the query only, not every document.  
- **Cost control**: Avoids recomputing unchanged docs.  
- **Consistency**: Ensures vectors are generated the same way.  
- **Re-embed support**: Allows safe upgrades to new models.  
- **Downstream reuse**: Same vectors also power clustering, deduplication, and recommendations.

---

## 2. Why Embeddings Matter

Embeddings convert text into vectors where **semantic similarity = geometric closeness**.

Benefits:
- **Synonyms/paraphrases**: “inflation cooling” ≈ “CPI growth slowed.”  
- **Context awareness**: “rate pause,” “dot plot,” and “FOMC guidance” cluster near “interest rate hikes.”  
- **Noise tolerance**: Articles with vague or clickbait titles can still match by summary content.  
- **Clustering & deduplication**: Group articles by topic, filter near-duplicates.  
- **Personalization**: User vectors (from history) can be compared against article vectors.  

> TL;DR: embeddings give **recall** and **semantic precision** that keyword search alone cannot.

---

## 3. Deep-Dive: Embedding Similarity Scoring

### 3.1 Query embedding
When a user searches “**interest rate hikes**”:  

```python
q_vector = openai.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
```
- Returns a **1536-dimensional vector** (for this model).  
- Think of it as the query’s “coordinates” in meaning-space.

### 3.2 Document embeddings
During ingestion, `embed.py` already computed and stored an embedding for each article (usually from **title + summary**).

If `normalize=True`, vectors are unit length (\|d\|=1), so cosine = dot product.

### 3.3 Cosine similarity
The similarity between query vector `q` and document vector `d` is:

\[
\text{cosine\_sim}(q, d) \;=\; \frac{q \cdot d}{\|q\| \; \|d\|}
\]

- Range: –1 to 1 (higher = more similar).  
- If normalized: simplifies to `q · d`.

### 3.4 Worked example (3D for clarity)
Query vector:  
\[
q = [0.2, \; 0.9, \; -0.1]
\]

Two docs:  
\[
d_1 = [0.3, \; 0.95, \; 0.0] \quad\text{(about rates)}
\]  
\[
d_2 = [-0.4, \; 0.1, \; 0.9] \quad\text{(about tech)}
\]

Dot products:  
- \( q \cdot d_1 = 0.915 \)  
- \( q \cdot d_2 = -0.08 \)

Norms:  
- \( \|q\| \approx 0.9274 \)  
- \( \|d_1\| \approx 0.9962 \)  
- \( \|d_2\| \approx 0.9899 \)

Cosine sims:  
- \( \text{cos}(q,d_1) \approx 0.990 \) ✅ high similarity  
- \( \text{cos}(q,d_2) \approx -0.087 \) ❌ dissimilar  

### 3.5 Where it fits
At search time:
1. Embed the **query** → \( q \).  
2. Fetch candidates with stored vectors \( d \).  
3. Compute cosine similarity for each.  
4. Combine with keyword score:  
   \[
   \text{final\_score} = \alpha \cdot \text{keyword} + \beta \cdot \text{semantic}
   \]  

---

## 4. Practical Notes
- Always embed **query and docs** with the same model/version.  
- Normalize vectors to make cosine = dot.  
- Tune α/β depending on whether you want more **precision** (keywords) or more **recall** (semantics).  
- For large corpora, consider **ANN indexes** (pgvector, FAISS).  

---

**Summary**  
- `scripts/embed.py` precomputes document embeddings for speed, cost, and consistency.  
- Embeddings are essential for matching **meaning**, not just words.  
- Embedding similarity scoring uses **cosine similarity** to measure how close a query is to each document in semantic space.  
