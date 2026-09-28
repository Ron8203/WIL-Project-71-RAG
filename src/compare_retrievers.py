import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from bm25_retriever import BM25Retriever


VECTORSTORE_DIR = "data/processed/faiss_index"


TEST_QUESTIONS = [
    "How can a small business protect itself from ransomware?",
    "How does multi-factor authentication protect an email account?",
    "What should a business do after an email account has been compromised?",
    "How should a small business back up important information?",
    "How can employees recognise suspicious phishing emails?",
]


def load_faiss():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def print_faiss_results(vectorstore, question):

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    print("\nFAISS RESULTS")

    for i, doc in enumerate(results, start=1):
        print(
            f"{i}. {doc.metadata['source']} "
            f"- Page {doc.metadata['page']}"
        )

        print(
            doc.page_content[:250]
            .replace("\n", " ")
        )

        print()


def print_bm25_results(retriever, question):

    results = retriever.search(
        question,
        top_k=3
    )

    print("\nBM25 RESULTS")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['source']} "
            f"- Page {result['page']} "
            f"(score={result['score']:.3f})"
        )

        print(
            result["text"][:250]
            .replace("\n", " ")
        )

        print()


def main():

    print("Loading retrieval systems...")

    vectorstore = load_faiss()
    bm25 = BM25Retriever()

    for question_number, question in enumerate(
        TEST_QUESTIONS,
        start=1
    ):

        print("\n")
        print("#" * 80)
        print(f"QUESTION {question_number}")
        print(question)
        print("#" * 80)

        print_faiss_results(
            vectorstore,
            question
        )

        print_bm25_results(
            bm25,
            question
        )


if __name__ == "__main__":
    main()