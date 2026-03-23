"""
main.py
────────────────────────────────────────────────────────────
Entry point — wires retriever + chatbot and runs a simple
REPL so you can test everything from the terminal.
────────────────────────────────────────────────────────────
"""

import os
from langchain_core.messages import HumanMessage
from langsmith import traceable
from retriever import create_retriever_from_url
from chatbot import ChatbotService
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tubetalk-ai"
RAG_VERSION = "multilingual"

#@traceable(name = "test-2-2")
def main():
    # ── Config ──────────────────────────────────────────────────
    GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")
    YOUTUBE_URL    ="https://youtu.be/ba-HMvDn_vU"
    THREAD_ID      = "MIT-test"

    # ── Step 1 : Build Hybrid Retriever ─────────────────────────
    retriever = create_retriever_from_url(YOUTUBE_URL , doc_language="english")
    if retriever is None:
        print(" Could not build retriever. Exiting.")
        return

    # ── Step 2 : Build LangGraph Chatbot ────────────────────────
    service = ChatbotService(api_key=GOOGLE_API_KEY)
    app     = service.build_chatbot(retriever)          # EnsembleRetriever injected here

    config = {
    "configurable": {
        "thread_id": THREAD_ID
    },
    "run_name": f"tubetalk-{RAG_VERSION}"
    }
    
    # ── Step 3 : Chat REPL ──────────────────────────────────────
    print("\n Chatbot ready!  Type 'quit' to exit.\n")
    while True:
        user_input = "What specific behavioral evidence did the speaker use to distinguish between the patient's impairment in navigating environments versus reproducing multi-part objects?"
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        last_msg = result["messages"][-1].content
        print(f"\n Bot:\n{last_msg}\n")
        break


if __name__ == "__main__":
    main()