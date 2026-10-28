"""Evaluate the recommender on the hand-written test queries in references.csv.

Metrics:
    Hit@1, Hit@5:  is the row's movie in the top 1 / top 5 results for its query?
                   Run for each weight in --weights (the plan asks for 0 and 0.3).
    BLEU, ROUGE:   the generated "why" reply for the row's own movie vs. the hand-written reply.
    Latency:       mean and median seconds for the search and for each explanation.
    Sentiment:     accuracy, precision and recall, read from outputs/sentiment_results.json
                   (written by sentiment.py).

Run from the movie-recommender folder:
    python evaluate.py
    python evaluate.py --weights 0 0.3 --no-explain
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from recommender import MovieRecommender, SENT_WEIGHT


def explain_for(rec: MovieRecommender, query: str, link: str):
    """Generate the reply for one specific movie, not whatever the ranking puts first.

    Returns (text, seconds), or (None, 0.0) if the movie is not in the catalog.
    """
    matches = rec.catalog[rec.catalog["rotten_tomatoes_link"] == link]
    if matches.empty:
        return None, 0.0
    movie = matches.iloc[0]
    start = time.perf_counter()
    query_vec = rec.embedder.encode([query], normalize_embeddings=True)[0]
    snippets = rec._snippets(query_vec, link)
    text = rec._explain(query, movie, snippets)
    return text, time.perf_counter() - start


def describe(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None}
    return {"mean": round(statistics.mean(values), 4), "median": round(statistics.median(values), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the recommender on references.csv.")
    parser.add_argument("--references", type=Path, default=Path("references.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, SENT_WEIGHT],
                        help="sentiment weights to compare for Hit@k")
    parser.add_argument("--no-explain", action="store_true",
                        help="skip reply generation, BLEU and ROUGE")
    args = parser.parse_args()

    refs = pd.read_csv(args.references, dtype=str).fillna("")
    refs = refs[refs["query"].str.strip() != ""].reset_index(drop=True)
    print(f"Loaded {len(refs)} test queries from {args.references}")

    rec = MovieRecommender(data_dir=args.data_dir, explain=not args.no_explain)
    print(f"Recommender loaded in {rec.load_seconds:.1f}s\n")

    # Hit@1 and Hit@5 for each weight.
    hit_table = []
    search_times = []
    for weight in args.weights:
        hit1 = hit5 = 0
        for _, row in refs.iterrows():
            results, timing = rec.recommend(row["query"], k=5, sent_weight=weight, explain=False)
            links = [r["link"] for r in results]
            hit1 += row["rotten_tomatoes_link"] in links[:1]
            hit5 += row["rotten_tomatoes_link"] in links[:5]
            search_times.append(timing["total"])
        n = len(refs)
        hit_table.append({"sent_weight": weight, "hit@1": round(hit1 / n, 4), "hit@5": round(hit5 / n, 4)})
        print(f"sent_weight={weight}: Hit@1 {hit1}/{n} ({hit1 / n:.2%}), Hit@5 {hit5}/{n} ({hit5 / n:.2%})")

    results_out = {
        "queries": len(refs),
        "hit_at_k": hit_table,
        "search_latency_s": describe(search_times),
    }

    per_query = []
    if not args.no_explain:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        hyps, golds, explain_times, rouge1, rougel = [], [], [], [], []
        for _, row in refs.iterrows():
            reference = row["reference_reply"].strip()
            if not reference:
                print(f"Skipped (no reference reply): {row['movie_title']}")
                continue
            text, seconds = explain_for(rec, row["query"], row["rotten_tomatoes_link"])
            if text is None:
                print(f"Skipped (movie not in catalog): {row['movie_title']}")
                continue
            scores = scorer.score(reference, text)
            hyps.append(text)
            golds.append(reference)
            explain_times.append(seconds)
            rouge1.append(scores["rouge1"].fmeasure)
            rougel.append(scores["rougeL"].fmeasure)
            per_query.append({
                "query": row["query"],
                "movie_title": row["movie_title"],
                "generated": text,
                "reference": reference,
                "rouge1_f": round(scores["rouge1"].fmeasure, 4),
                "rougeL_f": round(scores["rougeL"].fmeasure, 4),
                "explain_seconds": round(seconds, 3),
            })

        if hyps:
            bleu = sacrebleu.corpus_bleu(hyps, [golds]).score
            results_out["reply_quality"] = {
                "bleu": round(bleu, 2),
                "rouge1_f_mean": round(statistics.mean(rouge1), 4),
                "rougeL_f_mean": round(statistics.mean(rougel), 4),
                "replies_scored": len(hyps),
            }
            results_out["explain_latency_s"] = describe(explain_times)
            print(f"\nBLEU {bleu:.2f} | ROUGE-1 F {statistics.mean(rouge1):.3f} | ROUGE-L F {statistics.mean(rougel):.3f}"
                  f" over {len(hyps)} replies")

    sentiment_file = args.out_dir / "sentiment_results.json"
    if sentiment_file.exists():
        results_out["sentiment"] = json.loads(sentiment_file.read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_results.json").write_text(json.dumps(results_out, indent=2))
    if per_query:
        pd.DataFrame(per_query).to_csv(args.out_dir / "evaluation_per_query.csv", index=False)
    print(f"\nSaved to {args.out_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()