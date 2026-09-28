from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/chunks.json")


def tokenize(text):
    """
    Simple tokenizer for BM25.
    Converts text to lowercase and keeps words/numbers.
    """
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:

    def __init__(self):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(
            f"BM25 index created using "
            f"{len(self.chunks)} chunks."
        )

    def search(self, query, top_k=4):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_results = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]

        results = []

        for index, score in ranked_results:

            chunk = self.chunks[index]

            results.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )

        return results