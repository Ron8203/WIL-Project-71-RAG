from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


CHUNKS_FILE = Path("data/processed/chunks.json")
VECTORSTORE_DIR = Path("data/processed/faiss_index")


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            },
        )

        documents.append(doc)

    return documents


def main():
    print("Loading processed chunks...")

    documents = load_chunks()

    print(f"Loaded {len(documents)} chunks.")

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    print("\nVector store created successfully.")
    print(f"Saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()