import requests
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import clean_text
from config import BASE_URL, MODEL_NAME, DB_PATH, TOP_K

# 🔹 Load embedding model
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

db = None


def build_index():
    global db

   
    if os.path.exists(DB_PATH):
        print("📦 Loading existing DB...")
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        print("✅ DB Loaded!")
        return

    print("📥 Fetching books...")

    res = requests.get(f"{BASE_URL}/api/books/public")
    books = res.json()["books"]

    print("TOTAL BOOKS:", len(books))

    documents = []

    for book in books:
        title = clean_text(book.get("title", ""))
        authors = clean_text(book.get("authors", ""))
        categories = clean_text(book.get("categories", ""))

        description = clean_text(book.get("description", ""))

        if not description:
            description = f"This is a book titled {title} written by {authors} in category {categories}."

        
        text = f"""
        Title: {title}
        Title: {title}
        Title: {title}
        Author: {authors}
        Category: {categories}
        Description: {description}
        Keywords: {title} {description[:100]}
        """

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": book.get("_id"),
                    "title": book.get("title"),
                    "authors": book.get("authors"),
                    "thumbnail": book.get("thumbnail"),
                    "categories": book.get("categories"),
                    "description": book.get("description"),
                },
            )
        )

    print("🧠 Creating embeddings...")

    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    db.persist()  # ✅ save DB

    print("✅ AI Search Ready!")


# 🔍 SEARCH FUNCTION
def search_books(query):
    global db

    docs_with_scores = db.similarity_search_with_score(query, k=TOP_K*6)

    results = []
    seen_titles = set()

    for doc, score in docs_with_scores:
        title = doc.metadata.get("title")

       
        if title in seen_titles:
            continue
        seen_titles.add(title)

        results.append({
            "id": doc.metadata.get("id"),
            "title": title,
            "authors": doc.metadata.get("authors"),
            "thumbnail": doc.metadata.get("thumbnail"),
            "categories": doc.metadata.get("categories"),
            "description": doc.metadata.get("description"),
            "score": float(score)
        })

    
    results = sorted(results, key=lambda x: x["score"])

    print("RESULT COUNT:", len(results))  # debug

    return results