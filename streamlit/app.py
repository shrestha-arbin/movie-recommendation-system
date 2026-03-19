# src/app/app.py
import streamlit as st
import pandas as pd
from recommenders import (
    load_data_and_models,
    search_title,
    recommend_by_text,
    recommend_by_genome,
    recommend_by_cf,
    recommend_hybrid_from_movie
)

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

@st.cache_resource
def load_all():
    return load_data_and_models()

(
    movie_content_df,
    ratings_df,
    movie_map,
    tfidf_vectorizer,
    tfidf_matrix,
    genome_nn,
    genome_indices_map,
    svd,
    user_rated_movies,
    all_movie_ids,
    meta
) = load_all()

st.title("🎬 Movie Recommendation System")
with st.expander("Model info"):
    st.write("Loaded SVD model:", meta.get("svd_path"))
    if meta.get("svd_best_params"):
        st.json(meta["svd_best_params"])

tab1, tab2 = st.tabs(["Movie-based (Hybrid)", "User-based (CF)"])

# -------------------------
# TAB 1: movie-based hybrid
# -------------------------
with tab1:
    st.subheader("Find similar movies (Text / Genome / Hybrid)")
    query = st.text_input("Search a movie title", value="Toy Story")

    results = search_title(movie_content_df, query, top_k=30)
    if len(results) == 0:
        st.info("No matches. Try a different keyword.")
    else:
        choice = st.selectbox("Select a movie", results["title_unique"].tolist())

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("**Text-based (TF-IDF)**")
            rec_text = recommend_by_text(movie_content_df, tfidf_matrix, choice, top_n=10)
            st.dataframe(rec_text, use_container_width=True)

        with colB:
            st.markdown("**Genome-based (NN)**")
            rec_genome = recommend_by_genome(movie_content_df, genome_nn, genome_indices_map, choice, top_n=10)
            if rec_genome is None:
                if genome_nn is None or genome_indices_map is None:
                    st.info("Genome recommender is unavailable because genome model artifacts are missing.")
                else:
                    st.info("This selected movie has no usable genome signal. Try another movie for genome-based recommendations.")
            else:
                st.dataframe(rec_genome, use_container_width=True)

        with colC:
            st.markdown("**Hybrid (Text + Genome + optional CF)**")
            user_for_hybrid = st.text_input("Optional userId for CF boost", value="")
            user_id_val = int(user_for_hybrid) if user_for_hybrid.strip().isdigit() else None

            rec_hybrid = recommend_hybrid_from_movie(
                movie_content_df=movie_content_df,
                movie_map=movie_map,
                tfidf_matrix=tfidf_matrix,
                genome_nn=genome_nn,
                genome_indices_map=genome_indices_map,
                svd=svd,
                user_rated_movies=user_rated_movies,
                all_movie_ids=all_movie_ids,
                title_unique=choice,
                user_id=user_id_val,
                top_n=10
            )
            st.dataframe(rec_hybrid, use_container_width=True)

# -------------------------
# TAB 2: user-based CF
# -------------------------
with tab2:
    st.subheader("Personalized recommendations for a user (Tuned SVD)")
    user_id = st.text_input("Enter userId", "42")

    if user_id.strip().isdigit():
        user_id_val = int(user_id)
        rec_cf = recommend_by_cf(
            movie_map=movie_map,
            svd=svd,
            user_rated_movies=user_rated_movies,
            all_movie_ids=all_movie_ids,
            user_id=user_id_val,
            top_n=10
        )
        st.dataframe(rec_cf, use_container_width=True)
    else:
        st.info("Please enter a numeric userId.")
