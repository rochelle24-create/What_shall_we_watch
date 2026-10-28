"""Recommend movies for a query, with a short "why you might like it" for each.

Pipeline for one query:
    1. Embed the query with sentence-transformers (all-MiniLM-L6-v2).
    2. Take the CANDIDATES movies whose search text is most similar to the query.
    3. Re-rank the candidates: score = (1 - SENT_WEIGHT) * similarity + SENT_WEIGHT * Fresh rate.
    4. For each of the top k movies, pick the reviews most similar to the query and
       ask flan-t5-small to write one or two sentences grounded in them.

Run this file to try a query from the terminal:
    python recommender.py "a cozy comedy for a rainy night"
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data import build_catalog, load_movies, load_reviews

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPLAIN_MODEL = "google/flan-t5-small"
SENT_WEIGHT = 0.3       # weight of the Fresh rate in the ranking (plan default)
CANDIDATES = 50         # movies re-ranked after the similarity search
SNIPPETS = 3            # review snippets given to the explanation model
SNIPPET_CHARS = 300     # each snippet is cut to this length
MIN_SNIPPET_SIM = 0.25  # drop review snippets less similar than this to the query


def _repeats(text: str) -> bool:
    """True if any word of four or more letters appears three or more times."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return any(count >= 3 for word, count in Counter(words).items() if len(word) >= 4)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MovieRecommender:
    def __init__(self, data_dir: Path = Path("data"), cache_dir: Path = Path(".data"), explain: bool = True):
        start = time.perf_counter()
        movies = load_movies(data_dir)
        reviews = load_reviews(data_dir)
        self.catalog = build_catalog(movies, reviews).reset_index(drop=True)
        self.reviews_by_link = {link: group for link, group in reviews.groupby("rotten_tomatoes_link")}

        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.embeddings = self._catalog_embeddings(cache_dir)

        self.explain_enabled = explain
        if explain:
            self.explain_tok = AutoTokenizer.from_pretrained(EXPLAIN_MODEL)
            self.explain_model = AutoModelForSeq2SeqLM.from_pretrained(EXPLAIN_MODEL).to(DEVICE).eval()
        self.load_seconds = time.perf_counter() - start

    def _catalog_embeddings(self, cache_dir: Path) -> np.ndarray:
        """Embed every catalog movie and cache the result.

        The cache is reused only when the search text is unchanged. A fingerprint of
        the search text is stored with the embeddings, so changing the catalog (for
        example adding synopses) rebuilds them automatically.
        """
        texts = self.catalog["search_text"].tolist()
        fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        cache = cache_dir / "catalog_embeddings.npz"
        if cache.exists():
            saved = np.load(cache)
            if str(saved["fingerprint"]) == fingerprint:
                return saved["embeddings"]
        emb = self.embedder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=emb, fingerprint=fingerprint)
        return emb

    def _snippets(self, query_vec: np.ndarray, link: str) -> list[str]:
        """Return the reviews of one movie that are most similar to the query."""
        group = self.reviews_by_link.get(link)
        if group is None or group.empty:
            return []
        texts = group["review_content"].tolist()
        emb = self.embedder.encode(texts, normalize_embeddings=True)
        sims = emb @ query_vec
        best = np.argsort(-sims)[:SNIPPETS]
        return [texts[i][:SNIPPET_CHARS] for i in best if sims[i] >= MIN_SNIPPET_SIM]

    def _explain(self, query: str, movie: dict, snippets: list[str]) -> str:
        """Ask flan-t5-small for a short reason, grounded in the snippets."""
        notes = "".join(f"Review note {i}: {s}\n" for i, s in enumerate(snippets, 1))
        prompt = (
            "Using only the facts in the notes below, write exactly one sentence telling a viewer "
            "why they might enjoy this movie. Do not repeat any word.\n"
            f"Viewer wants: {query}\n"
            f"Movie: {movie['movie_title']}\n"
            f"Genres: {movie['genres']}\n"
            f"Critics' consensus: {movie['critics_consensus']}\n"
            f"{notes}"
            "Answer:"
        )
        inputs = self.explain_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = self.explain_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
        text = self.explain_tok.decode(out[0], skip_special_tokens=True).strip()

        # Fall back to the consensus when the model returns almost nothing or repeats itself.
        if len(text) < 10 or _repeats(text):
            if movie["critics_consensus"]:
                return movie["critics_consensus"]
            return f"Critics rate it {round(movie['fresh_rate'] * 100)}% Fresh."
        return text

    def recommend(self, query: str, k: int = 5, sent_weight: float = SENT_WEIGHT, explain: bool | None = None):
        """Return (results, timing). Each result is a dict with title, genres, scores and the explanation."""
        if explain is None:
            explain = self.explain_enabled
        timing = {}

        start = time.perf_counter()
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_vec
        cand_idx = np.argsort(-sims)[:max(CANDIDATES, k)]
        timing["search"] = time.perf_counter() - start

        start = time.perf_counter()
        candidates = self.catalog.iloc[cand_idx].copy()
        candidates["similarity"] = sims[cand_idx]
        candidates["score"] = (
            (1 - sent_weight) * candidates["similarity"] + sent_weight * candidates["fresh_rate_smoothed"]
        )
        top = candidates.sort_values("score", ascending=False).head(k)
        results = [
            {
                "title": row["movie_title"],
                "genres": row["genres"],
                "fresh_rate": round(float(row["fresh_rate"]), 3),
                "similarity": round(float(row["similarity"]), 3),
                "score": round(float(row["score"]), 3),
                "why": "",
                "link": row["rotten_tomatoes_link"],
            }
            for _, row in top.iterrows()
        ]
        timing["rank"] = time.perf_counter() - start

        if explain:
            start = time.perf_counter()
            for result, (_, row) in zip(results, top.iterrows()):
                snippets = self._snippets(query_vec, row["rotten_tomatoes_link"])
                result["why"] = self._explain(query, row, snippets)
            timing["explain"] = time.perf_counter() - start

        timing["total"] = sum(timing.values())
        return results, timing


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python recommender.py "what you want to watch"')
        return
    query = " ".join(sys.argv[1:])
    recommender = MovieRecommender()
    print(f"Loaded in {recommender.load_seconds:.1f}s\n")
    results, timing = recommender.recommend(query)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']} | Fresh {round(r['fresh_rate'] * 100)}% | score {r['score']}")
        print(f"   {r['why']}\n")
    print("Timing (s):", {k: round(v, 2) for k, v in timing.items()})


if __name__ == "__main__":
    main()