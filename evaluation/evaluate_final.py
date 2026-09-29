from pathlib import Path
import sys
import math
import pandas as pd

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from bm25_retriever import BM25Retriever
from rag import CyberSecurityRAG, FALLBACK_RESPONSE
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


TEST_FILE = PROJECT_ROOT / "evaluation" / "test_questions_40.csv"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "processed" / "faiss_index"

RETRIEVAL_RESULTS_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_results_30.csv"
)

RETRIEVAL_SUMMARY_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_summary_30.csv"
)

REFUSAL_RESULTS_FILE = (
    PROJECT_ROOT / "evaluation" / "refusal_results_10.csv"
)

REFUSAL_SUMMARY_FILE = (
    PROJECT_ROOT / "evaluation" / "refusal_summary_10.csv"
)

TOP_K_DOCS = 3
RETRIEVAL_POOL = 10


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def parse_relevant_sources(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def unique_sources_from_faiss(
    vectorstore,
    question,
    top_k=TOP_K_DOCS
):
    docs = vectorstore.similarity_search(
        question,
        k=RETRIEVAL_POOL,
    )

    ranked_sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        if source not in seen:
            seen.add(source)
            ranked_sources.append(source)

        if len(ranked_sources) >= top_k:
            break

    return ranked_sources


def unique_sources_from_bm25(
    retriever,
    question,
    top_k=TOP_K_DOCS
):
    results = retriever.search(
        question,
        top_k=RETRIEVAL_POOL,
    )

    ranked_sources = []
    seen = set()

    for result in results:
        source = result["source"]

        if source not in seen:
            seen.add(source)
            ranked_sources.append(source)

        if len(ranked_sources) >= top_k:
            break

    return ranked_sources


def hit_at_k(
    ranked_sources,
    relevant_sources,
    k
):
    return int(
        any(
            source in relevant_sources
            for source in ranked_sources[:k]
        )
    )


def reciprocal_rank(
    ranked_sources,
    relevant_sources
):
    for rank, source in enumerate(
        ranked_sources,
        start=1
    ):
        if source in relevant_sources:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_sources,
    relevant_sources,
    k=3
):
    gains = [
        1 if source in relevant_sources else 0
        for source in ranked_sources[:k]
    ]

    dcg = 0.0

    for index, gain in enumerate(
        gains,
        start=1
    ):
        dcg += (
            gain /
            math.log2(index + 1)
        )

    ideal_relevant = min(
        len(relevant_sources),
        k,
    )

    if ideal_relevant == 0:
        return 0.0

    idcg = 0.0

    for index in range(
        1,
        ideal_relevant + 1
    ):
        idcg += (
            1 /
            math.log2(index + 1)
        )

    return dcg / idcg


# ---------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------

def evaluate_retrieval_method(
    method_name,
    questions_df,
    search_function
):
    rows = []

    for _, row in questions_df.iterrows():

        question = row["question"]
        category = row["category"]

        relevant_sources = parse_relevant_sources(
            row["relevant_sources"]
        )

        ranked_sources = search_function(
            question,
            TOP_K_DOCS,
        )

        padded = ranked_sources + [""] * (
            TOP_K_DOCS -
            len(ranked_sources)
        )

        rows.append(
            {
                "method": method_name,
                "question": question,
                "category": category,
                "relevant_sources": ";".join(
                    sorted(relevant_sources)
                ),
                "rank_1_source": padded[0],
                "rank_2_source": padded[1],
                "rank_3_source": padded[2],
                "hit_at_1": hit_at_k(
                    ranked_sources,
                    relevant_sources,
                    1,
                ),
                "hit_at_3": hit_at_k(
                    ranked_sources,
                    relevant_sources,
                    3,
                ),
                "reciprocal_rank": reciprocal_rank(
                    ranked_sources,
                    relevant_sources,
                ),
                "ndcg_at_3": ndcg_at_k(
                    ranked_sources,
                    relevant_sources,
                    3,
                ),
            }
        )

    return rows


# ---------------------------------------------------------
# Unsupported-question refusal evaluation
# ---------------------------------------------------------

def evaluate_refusals(
    rag,
    unsupported_df
):
    rows = []

    total = len(unsupported_df)

    for number, (_, row) in enumerate(
        unsupported_df.iterrows(),
        start=1
    ):

        question = row["question"]

        print(
            f"Testing unsupported question "
            f"{number}/{total}..."
        )

        result = rag.answer(question)

        answer = result["answer"].strip()

        refused_correctly = int(
            answer == FALLBACK_RESPONSE
        )

        rows.append(
            {
                "question": question,
                "category": row["category"],
                "answer": answer,
                "correct_refusal": refused_correctly,
                "incorrect_answer": (
                    1 - refused_correctly
                ),
            }
        )

    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Could not find: {TEST_FILE}"
        )

    print(
        "Loading 40-question evaluation set..."
    )

    questions_df = pd.read_csv(
        TEST_FILE
    )

    answerable_df = questions_df[
        questions_df["answerable"]
        .astype(str)
        .str.lower()
        .eq("yes")
    ].copy()

    unsupported_df = questions_df[
        questions_df["answerable"]
        .astype(str)
        .str.lower()
        .eq("no")
    ].copy()

    print(
        f"Answerable questions: "
        f"{len(answerable_df)}"
    )

    print(
        f"Unsupported questions: "
        f"{len(unsupported_df)}"
    )

    # -----------------------------------------------------
    # Load retrieval systems
    # -----------------------------------------------------

    print("\nLoading FAISS...")

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("Loading BM25...")

    bm25 = BM25Retriever()

    # -----------------------------------------------------
    # Retrieval metrics on 30 answerable questions
    # -----------------------------------------------------

    print(
        "\nEvaluating FAISS on "
        "30 answerable questions..."
    )

    faiss_rows = evaluate_retrieval_method(
        "FAISS",
        answerable_df,
        lambda question, k:
        unique_sources_from_faiss(
            vectorstore,
            question,
            k,
        ),
    )

    print(
        "Evaluating BM25 on "
        "30 answerable questions..."
    )

    bm25_rows = evaluate_retrieval_method(
        "BM25",
        answerable_df,
        lambda question, k:
        unique_sources_from_bm25(
            bm25,
            question,
            k,
        ),
    )

    retrieval_results_df = pd.DataFrame(
        faiss_rows + bm25_rows
    )

    retrieval_results_df.to_csv(
        RETRIEVAL_RESULTS_FILE,
        index=False,
    )

    retrieval_summary_df = (
        retrieval_results_df
        .groupby(
            "method",
            as_index=False
        )
        .agg(
            questions=(
                "question",
                "count"
            ),
            hit_at_1=(
                "hit_at_1",
                "mean"
            ),
            hit_at_3=(
                "hit_at_3",
                "mean"
            ),
            mrr_at_3=(
                "reciprocal_rank",
                "mean"
            ),
            ndcg_at_3=(
                "ndcg_at_3",
                "mean"
            ),
        )
    )

    for metric in [
        "hit_at_1",
        "hit_at_3",
        "mrr_at_3",
        "ndcg_at_3",
    ]:
        retrieval_summary_df[
            metric
        ] = retrieval_summary_df[
            metric
        ].round(4)

    retrieval_summary_df.to_csv(
        RETRIEVAL_SUMMARY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Refusal test on 10 unsupported questions
    # -----------------------------------------------------

    print(
        "\nLoading RAG system for "
        "unsupported-question testing..."
    )

    rag = CyberSecurityRAG()

    print(
        "\nEvaluating refusal behaviour..."
    )

    refusal_rows = evaluate_refusals(
        rag,
        unsupported_df,
    )

    refusal_results_df = pd.DataFrame(
        refusal_rows
    )

    refusal_results_df.to_csv(
        REFUSAL_RESULTS_FILE,
        index=False,
    )

    refusal_accuracy = (
        refusal_results_df[
            "correct_refusal"
        ].mean()
    )

    false_answer_rate = (
        refusal_results_df[
            "incorrect_answer"
        ].mean()
    )

    refusal_summary_df = pd.DataFrame(
        [
            {
                "unsupported_questions":
                    len(refusal_results_df),
                "correct_refusals":
                    int(
                        refusal_results_df[
                            "correct_refusal"
                        ].sum()
                    ),
                "refusal_accuracy":
                    round(
                        refusal_accuracy,
                        4
                    ),
                "false_answer_rate":
                    round(
                        false_answer_rate,
                        4
                    ),
            }
        ]
    )

    refusal_summary_df.to_csv(
        REFUSAL_SUMMARY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Print final results
    # -----------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "FINAL RETRIEVAL "
        "EVALUATION - "
        "30 ANSWERABLE QUESTIONS"
    )

    print(
        "=" * 70
    )

    print(
        retrieval_summary_df.to_string(
            index=False
        )
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "UNSUPPORTED-QUESTION "
        "EVALUATION - "
        "10 QUESTIONS"
    )

    print(
        "=" * 70
    )

    print(
        refusal_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nFiles saved:"
    )

    print(
        RETRIEVAL_RESULTS_FILE
    )

    print(
        RETRIEVAL_SUMMARY_FILE
    )

    print(
        REFUSAL_RESULTS_FILE
    )

    print(
        REFUSAL_SUMMARY_FILE
    )


if __name__ == "__main__":
    main()