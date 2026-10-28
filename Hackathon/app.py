"""Streamlit app: type what you want to watch and get suggestions with a reason for each.

Run from the movie-recommender folder:
    streamlit run app.py

Yes / No / Try again are shown on the page only. Answers are not saved.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from recommender import MovieRecommender

DATA_DIR = Path("data")
PAGE_SIZE = 5          # suggestions shown at a time
MAX_SUGGESTIONS = 50   # suggestions searched per query; "Try again" pages through them

state = st.session_state
state.setdefault("query", "")
state.setdefault("ranking", [])
state.setdefault("page", 1)
state.setdefault("answer", None)
state.setdefault("explanations", {})
state.setdefault("timing", {})


@st.cache_resource(show_spinner="Loading the models and movie catalog (the first time takes a few minutes)...")
def load_recommender() -> MovieRecommender:
    return MovieRecommender(data_dir=DATA_DIR)


def explain_movie(rec: MovieRecommender, query: str, link: str) -> str:
    """Write the reason for one movie, using reviews that match the query."""
    movie = rec.catalog[rec.catalog["rotten_tomatoes_link"] == link].iloc[0]
    query_vec = rec.embedder.encode([query], normalize_embeddings=True)[0]
    snippets = rec._snippets(query_vec, link)
    return rec._explain(query, movie, snippets)


def set_answer(answer: str) -> None:
    state.answer = answer


def try_again() -> None:
    if state.page * PAGE_SIZE < len(state.ranking):
        state.page += 1
        state.answer = None
    else:
        state.answer = "There are no more suggestions for this search."


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("What shall we watch?")
st.write(
    "Describe the kind of movie you want. We'll suggest movies based on Rotten Tomatoes "
    "critic reviews, and explain why you might like each one."
)

try:
    rec = load_recommender()
except FileNotFoundError:
    st.error("The data folder was not found. Run `python data.py --data-dir data --write-splits` first.")
    st.stop()

with st.form("search_form"):
    query = st.text_input(
        "What kind of movie would you like to watch today?",
        placeholder="e.g., a cozy comedy for a rainy night",
    )
    go = st.form_submit_button("Go", type="primary")

if go:
    text = query.strip()
    if not text:
        st.warning("Please type what you want to watch.")
    else:
        with st.spinner("Finding movies..."):
            ranking, timing = rec.recommend(text, k=MAX_SUGGESTIONS, explain=False)
        state.query = text
        state.ranking = ranking
        state.timing = timing
        state.page = 1
        state.answer = None

if state.ranking:
    start = (state.page - 1) * PAGE_SIZE
    batch = state.ranking[start:start + PAGE_SIZE]

    st.subheader(f"Suggestions for: {state.query}")
    for position, movie in enumerate(batch, start=start + 1):
        key = (state.query, movie["link"])
        if key not in state.explanations:
            with st.spinner("Writing the reason..."):
                state.explanations[key] = explain_movie(rec, state.query, movie["link"])
        with st.container(border=True):
            st.markdown(f"**{position}. {movie['title']}**")
            st.caption(f"{movie['genres']} · {round(movie['fresh_rate'] * 100)}% of critics say Fresh")
            st.write(state.explanations[key])

    st.write("Was that a good movie suggestion for you?")
    col_yes, col_no, col_again = st.columns(3)
    col_yes.button("Yes", on_click=set_answer, args=("Yes",), use_container_width=True)
    col_no.button("No", on_click=set_answer, args=("No",), use_container_width=True)
    col_again.button("Try again", on_click=try_again, use_container_width=True)

    if state.answer in ("Yes", "No"):
        st.info(f"Thanks! You answered {state.answer}. Answers are not saved in this app.")
    elif state.answer:
        st.info(state.answer)

    with st.expander("Timing"):
        st.write({name: f"{seconds:.3f}s" for name, seconds in state.timing.items()})