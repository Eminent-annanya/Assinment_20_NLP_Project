
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="centered"
)

# Load saved components
@st.cache_resource
def load_components():
    data = joblib.load("artifacts/book_recommender.joblib")

    return (
        data["books"],
        data["similarity_matrix"]
    )

books, similarity_matrix = load_components()

# Application title
st.title("📚 Book Recommendation System")
st.write("Discover books similar to your favourite books!")

st.divider()

# Book selection dropdown
book_list = books["title"].dropna().unique().tolist()

selected_book = st.selectbox(
    "Select a book:",
    book_list
)

# Recommendation button
if st.button("Get Recommendations", type="primary"):

    # Find selected book index
    book_index = books.index[
        books["title"] == selected_book
    ][0]

    # Calculate similarity scores
    scores = similarity_matrix.getrow(
        book_index
    ).toarray().ravel()

    # Sort scores in descending order
    sorted_indices = scores.argsort()[::-1]

    # Generate recommendations
    recommendations = []

    for index in sorted_indices:

        if books.iloc[index]["title"] == selected_book:
            continue

        recommendations.append({
            "Book Title": books.iloc[index]["title"],
            "Author": books.iloc[index]["authors"],
            "Similarity Score": round(float(scores[index]), 4)
        })

        if len(recommendations) == 5:
            break

    # Display recommendations
    st.subheader("Recommended Books")

    if recommendations:
        result = pd.DataFrame(recommendations)
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No recommendations found.")
