import sys
from pathlib import Path

import streamlit as st


# Allow imports from the src directory
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from rag import CyberSecurityRAG, FALLBACK_RESPONSE


st.set_page_config(
    page_title="Small Business Cybersecurity Assistant",
    page_icon="🔐",
    layout="centered"
)


st.title("🔐 Small Business Cybersecurity Assistant")

st.write(
    """
    Ask questions about cybersecurity for Australian small businesses.

    Answers are generated using guidance retrieved from
    Australian Cyber Security Centre documents.
    """
)

st.info(
    "This prototype provides general cybersecurity information "
    "and does not replace professional cybersecurity, legal, "
    "or incident-response advice."
)


@st.cache_resource
def load_rag():
    return CyberSecurityRAG()


with st.spinner("Loading cybersecurity knowledge base..."):
    rag = load_rag()


question = st.text_input(
    "Ask a cybersecurity question",
    placeholder=(
        "Example: How can I protect my business from ransomware?"
    )
)


if st.button("Get Answer"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching trusted cybersecurity guidance..."
        ):

            try:

                result = rag.answer(
                    question.strip()
                )

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                if (
                    result["answer"] != FALLBACK_RESPONSE
                    and result["sources"]
                ):

                    st.subheader("Sources")

                    for source in result["sources"]:

                        st.write(
                            f"• {source['source']} "
                            f"— Page {source['page']}"
                        )

            except Exception as error:

                st.error(
                    "The assistant could not generate an answer."
                )

                st.exception(error)


st.divider()

st.caption(
    "Test-Driven Retrieval-Augmented Generation prototype | "
    "RMIT WIL Project 71"
)