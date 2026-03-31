import pandas as pd
import numpy as np
import gradio as gr
import traceback

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ----------------------------------------------------
# LOAD BOOK DATA
# ----------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

# IMPORTANT FIX: Chroma returns ISBNs as strings, so match format
books["isbn13"] = books["isbn13"].astype(str)

# Fix thumbnails
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)


# ----------------------------------------------------
# LOAD DOCUMENTS FOR SEMANTIC SEARCH
# ----------------------------------------------------
documents = []
with open("tagged_description.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(Document(page_content=line))


# ----------------------------------------------------
# EMBEDDINGS + CHROMA VECTOR DB
# ----------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="./db_books"
)


# ----------------------------------------------------
# RECOMMENDATION PIPELINE
# ----------------------------------------------------
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Semantic search (returns documents)
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract top ISBNs from recs
    books_list = []
    for rec in recs:
        raw = rec.page_content.strip().strip('"')
        isbn = raw.split()[0]  # first token = ISBN (string)
        books_list.append(isbn)

    # Map to actual books
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Tone filter
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            # Short description
            words = row["description"].split()
            truncated = " ".join(words[:30]) + "..."

            # Pretty author formatting
            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            caption = f"{row['title']} by {authors_str}: {truncated}"
            results.append((row["large_thumbnail"], caption))

        return results

    except Exception:
        traceback.print_exc()
        return []


# ----------------------------------------------------
# USER INTERFACE (GRADIO)
# ----------------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# 📚 N.E.X.T T.E.X.T — Your Next Book")
    gr.Markdown("### Describe what kind of book you feel like reading")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book:",
            placeholder="e.g., A fantasy adventure about friendship"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Tone:",
            value="All"
        )
        submit_button = gr.Button("Recommend Books")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch(share=True, server_port=7861)
