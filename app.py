# app.py
import streamlit as st
from recommend import Recommender

st.set_page_config(page_title="Multi-Domain Recommender", layout="wide")

@st.cache_resource
def get_recommender():
    return Recommender()

recommender = get_recommender()

st.title("🎯 Multi-Domain AI Recommender — Movies & Books")
st.markdown("Select mode, then choose a title or type a text query. Results are semantic recommendations using sentence embeddings.")

col1, col2 = st.columns([1, 2])

with col1:
    mode = st.selectbox("Mode", options=["movie", "book"], format_func=lambda x: "🎬 Movies" if x=="movie" else "📚 Books")
    st.write("Dataset size:")
    if mode == "movie":
        st.write(f"{len(recommender.movies_df)} movies")
        titles = recommender.movies_df['title'].tolist()
    else:
        st.write(f"{len(recommender.books_df)} books")
        titles = recommender.books_df['title'].tolist()

    st.write("---")
    input_mode = st.radio("Input type", options=["Choose a title", "Free text query"])
    if input_mode == "Choose a title":
        selected_title = st.selectbox("Select title", options=titles)
    else:
        selected_title = st.text_area("Enter text to find similar items", value="")

    top_k = st.slider("Number of recommendations", 1, 10, 5)
    run = st.button("Recommend")

with col2:
    st.header("Recommendations")
    placeholder = st.empty()

if run:
    try:
        if input_mode == "Choose a title":
            if not selected_title:
                st.warning("Please select a title.")
            else:
                results = recommender.recommend_by_title(selected_title, mode=mode, top_k=top_k)
        else:
            if not selected_title.strip():
                st.warning("Please type a query.")
                results = []
            else:
                results = recommender.recommend_by_text(selected_title, mode=mode, top_k=top_k)

        if not results:
            placeholder.info("No recommendations found.")
        else:
            # show results
            for r in results:
                st.subheader(r["title"])
                st.write(r["overview"])
                st.caption(f"Similarity score: {r['score']:.4f}")
                st.write("---")
    except Exception as e:
        st.error(f"Error: {e}")
