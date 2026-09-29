from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_summary_30.csv"
)

REFUSAL_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "refusal_summary_10.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def retrieval_chart():

    df = pd.read_csv(
        SUMMARY_FILE
    )

    metrics = [
        "hit_at_1",
        "hit_at_3",
        "mrr_at_3",
        "ndcg_at_3",
    ]

    display_names = [
        "Hit@1",
        "Hit@3",
        "MRR@3",
        "NDCG@3",
    ]

    chart_data = (
        df
        .set_index("method")[metrics]
        .T
    )

    chart_data.index = display_names

    ax = chart_data.plot(
        kind="bar",
        figsize=(9, 6)
    )

    ax.set_title(
        "FAISS vs BM25 Retrieval Performance"
    )

    ax.set_xlabel(
        "Evaluation Metric"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.1
    )

    ax.legend(
        title="Retriever"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "retrieval_comparison.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def refusal_chart():

    df = pd.read_csv(
        REFUSAL_FILE
    )

    refusal_accuracy = (
        df.loc[
            0,
            "refusal_accuracy"
        ]
    )

    false_answer_rate = (
        df.loc[
            0,
            "false_answer_rate"
        ]
    )

    values = [
        refusal_accuracy,
        false_answer_rate,
    ]

    labels = [
        "Correct Refusal",
        "False Answer",
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Unsupported Question Handling"
    )

    ax.set_ylabel(
        "Rate"
    )

    ax.set_ylim(
        0,
        1.1
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.03,
            f"{value * 100:.0f}%",
            ha="center"
        )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "unsupported_question_results.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def main():

    retrieval_chart()
    refusal_chart()

    print(
        "\nEvaluation charts created successfully."
    )


if __name__ == "__main__":
    main()