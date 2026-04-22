import sys
from src.ingestion import run_ingestion
from src.retrieval import load_vectorstore, build_retriever
from src.generation import build_rag_chain, ask


def main():
    print("=== Production RAG ===\n")

    vectorstore = load_vectorstore()
    retriever   = build_retriever(vectorstore)
    chain       = build_rag_chain(retriever)

    print("\nAsk questions about your documents.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("❓ > ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        ask(question, chain)


if __name__ == "__main__":
    main()