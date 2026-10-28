"""Load, clean, and subsample the Rotten Tomatoes data.

Expects two CSV files in the data folder:
    rotten_tomatoes_movies.csv
    rotten_tomatoes_critic_reviews.csv

Run this file directly to check that both files load and print their row counts:
    python data.py --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MOVIES_FILE = "rotten_tomatoes_movies.csv"
REVIEWS_FILE = "rotten_tomatoes_critic_reviews.csv"
SAMPLE_FILE = "critic_sample_10k.csv"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# Train/validation/test split of the sample. The folder name starts with a dot,
# so it is hidden on macOS and Linux. It is listed in .gitignore.
SPLIT_DIR = Path(".data")
TEST_N = 1000
VAL_N = 500
TRAIN_N = 2000
SPLIT_SEED = 42

# Smoothing for the Fresh rate used in ranking. A movie's rate is pulled toward the
# overall Fresh share by this many "virtual" reviews, so movies with only a few reviews
# don't rank at the top just because all of them are Fresh.
SMOOTHING_M = 10


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Load the movie file, keeping the columns the app uses."""
    movies = pd.read_csv(data_dir / MOVIES_FILE)
    movies = movies.drop_duplicates(subset="rotten_tomatoes_link")
    movies["movie_title"] = movies["movie_title"].fillna("Untitled")
    for col in ["movie_info", "critics_consensus", "genres", "directors"]:
        movies[col] = movies[col].fillna("")
    return movies


def load_reviews(data_dir: Path) -> pd.DataFrame:
    """Load critic reviews and keep only rows with text and a Fresh/Rotten label."""
    reviews = pd.read_csv(data_dir / REVIEWS_FILE)
    reviews = reviews.dropna(subset=["review_content", "review_type"])
    reviews = reviews[reviews["review_type"].isin(["Fresh", "Rotten"])]
    reviews = reviews.reset_index(drop=True)
    return reviews


def build_catalog(movies: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per movie that has at least one critic review.

    Adds:
        review_count: number of labeled critic reviews
        fresh_rate:   share of reviews labeled Fresh, from 0.0 to 1.0
        fresh_rate_smoothed: Fresh rate adjusted toward the overall share
        search_text:  text the query is matched against (title, genres, consensus, synopsis)
    Movies with no reviews are removed, as the project decisions require.
    """
    stats = (
        reviews.groupby("rotten_tomatoes_link")["review_type"]
        .agg(review_count="size", fresh_count=lambda s: (s == "Fresh").sum())
        .reset_index()
    )
    stats["fresh_rate"] = stats["fresh_count"] / stats["review_count"]
    overall_fresh = (reviews["review_type"] == "Fresh").mean()
    stats["fresh_rate_smoothed"] = (
        (stats["fresh_count"] + SMOOTHING_M * overall_fresh)
        / (stats["review_count"] + SMOOTHING_M)
    )
    stats = stats.drop(columns="fresh_count")

    catalog = movies.merge(stats, on="rotten_tomatoes_link", how="inner")
    catalog["search_text"] = (
        catalog["movie_title"]
        + ". Genres: " + catalog["genres"]
        + ". " + catalog["critics_consensus"]
        + " " + catalog["movie_info"]
    ).str.strip()
    return catalog.reset_index(drop=True)


def sample_reviews(reviews: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Random subsample of labeled reviews, saved as critic_sample_10k.csv."""
    n = min(n, len(reviews))
    return reviews.sample(n=n, random_state=seed).reset_index(drop=True)


def split_sample(sample: pd.DataFrame, seed: int = SPLIT_SEED):
    """Split the sample into (train, val, test) with no overlap. Adds a label column (Fresh = 1)."""
    shuffled = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["label"] = (shuffled["review_type"] == "Fresh").astype(int)
    test = shuffled.iloc[:TEST_N]
    val = shuffled.iloc[TEST_N:TEST_N + VAL_N]
    train = shuffled.iloc[TEST_N + VAL_N:TEST_N + VAL_N + TRAIN_N]
    return train, val, test


def save_splits(train, val, test, split_dir: Path = SPLIT_DIR) -> Path:
    """Write train.csv, val.csv and test.csv so every script uses the same reviews."""
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(split_dir / f"{name}.csv", index=False)
    return split_dir


def load_splits(split_dir: Path = SPLIT_DIR):
    """Read the saved split files back as (train, val, test)."""
    return tuple(pd.read_csv(split_dir / f"{name}.csv") for name in ["train", "val", "test"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Rotten Tomatoes data and print row counts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--write-sample", action="store_true",
                        help="also write critic_sample_10k.csv to the data folder")
    parser.add_argument("--write-splits", action="store_true",
                        help="split the sample and write train/val/test CSVs to the hidden .data folder")
    args = parser.parse_args()

    movies = load_movies(args.data_dir)
    reviews = load_reviews(args.data_dir)
    catalog = build_catalog(movies, reviews)

    print(f"Movies loaded:              {len(movies):>10,}")
    print(f"Labeled critic reviews:     {len(reviews):>10,}")
    print(f"Movies in catalog:          {len(catalog):>10,}")
    print(f"Movies removed (no reviews): {len(movies) - len(catalog):>9,}")

    if args.write_sample or args.write_splits:
        sample = sample_reviews(reviews)

    if args.write_sample:
        out = args.data_dir / SAMPLE_FILE
        sample.to_csv(out, index=False)
        print(f"Wrote {len(sample):,} reviews to {out}")

    if args.write_splits:
        train, val, test = split_sample(sample)
        folder = save_splits(train, val, test)
        print(f"Wrote splits to {folder}: train {len(train):,}, validation {len(val):,}, test {len(test):,}")

if __name__ == "__main__":
    main()
