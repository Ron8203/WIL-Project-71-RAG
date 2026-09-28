from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = "data/processed/faiss_index"


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    question = "How can a small business protect itself from ransomware?"

    results = vectorstore.similarity_search(
        question,
        k=4,
    )

    print("\nQUESTION:")
    print(question)

    print("\nTOP RESULTS:\n")

    for i, doc in enumerate(results, start=1):
        print("=" * 70)
        print(f"RESULT {i}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Page: {doc.metadata['page']}")
        print(f"Chunk ID: {doc.metadata['chunk_id']}")
        print()
        print(doc.page_content[:700])
        print()


if __name__ == "__main__":
    main()