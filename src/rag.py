from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


VECTORSTORE_DIR = "data/processed/faiss_index"

FALLBACK_RESPONSE = (
    "I cannot answer this question from the available "
    "cybersecurity documents."
)


class CyberSecurityRAG:

    def __init__(self):

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Loading FAISS vector store...")

        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Loading Llama 3.2 through Ollama...")

        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

        print("RAG system ready.\n")

    def retrieve(self, question, top_k=6):

        return self.vectorstore.similarity_search(
            question,
            k=top_k,
        )

    def build_context(self, documents):

        context_parts = []

        for doc in documents:

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                "Unknown page"
            )

            context_parts.append(
                f"""
DOCUMENT: {source}
PAGE: {page}

{doc.page_content}
"""
            )

        return "\n".join(context_parts)

    def get_unique_sources(self, documents):

        sources = []
        seen = set()

        for doc in documents:

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                "Unknown page"
            )

            key = (source, page)

            if key not in seen:
                seen.add(key)

                sources.append(
                    {
                        "source": source,
                        "page": page,
                    }
                )

        return sources

    def answer(self, question):

        documents = self.retrieve(
            question,
            top_k=6
        )

        context = self.build_context(
            documents
        )

        prompt = f"""
You are a cybersecurity information assistant for
Australian small businesses.

Answer the user's question ONLY using information
explicitly stated in the supplied context.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent facts or recommendations.

3. Only include recommendations clearly supported by
   the supplied context.

4. Ignore incomplete sentence fragments.

5. Do not create citations or source names yourself.
   The application will display sources separately.

6. If the supplied context does not contain enough
   relevant information to answer the question,
   respond exactly with:

"{FALLBACK_RESPONSE}"

7. Keep the answer concise, clear and practical.

8. If the question asks about prevention, prioritise
   prevention guidance.

9. If the question asks what to do after an incident,
   prioritise incident-response guidance.

10. Do not repeat the same recommendation.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

        response = self.llm.invoke(prompt)

        answer_text = response.content.strip()

        if answer_text == FALLBACK_RESPONSE:
            sources = []
        else:
            sources = self.get_unique_sources(
                documents
            )

        return {
            "answer": answer_text,
            "sources": sources,
        }


if __name__ == "__main__":

    rag = CyberSecurityRAG()

    question = input(
        "Ask a cybersecurity question: "
    )

    result = rag.answer(question)

    print("\nANSWER:\n")
    print(result["answer"])

    if result["sources"]:

        print("\nSOURCES:\n")

        for i, source in enumerate(
            result["sources"],
            start=1
        ):
            print(
                f"{i}. "
                f"{source['source']} "
                f"- Page {source['page']}"
            )