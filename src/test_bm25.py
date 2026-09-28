from bm25_retriever import BM25Retriever


def main():

    retriever = BM25Retriever()

    question = (
        "How can a small business protect itself "
        "from ransomware?"
    )

    results = retriever.search(
        question,
        top_k=4
    )

    print("\nQUESTION:")
    print(question)

    print("\nBM25 TOP RESULTS:\n")

    for i, result in enumerate(results, start=1):

        print("=" * 70)
        print(f"RESULT {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print()

        print(result["text"][:700])

        print()


if __name__ == "__main__":
    main()