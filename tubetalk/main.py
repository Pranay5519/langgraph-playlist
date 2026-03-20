"""
main.py
────────────────────────────────────────────────────────────
Entry point — wires retriever + chatbot and runs a simple
REPL so you can test everything from the terminal.
────────────────────────────────────────────────────────────
"""

import os
from langchain_core.messages import HumanMessage

from retriever import create_retriever_from_url
from chatbot import ChatbotService
from dotenv import load_dotenv
load_dotenv()

def main():
    # ── Config ──────────────────────────────────────────────────
    GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")
    YOUTUBE_URL    ="https://youtu.be/WzvURhaDZqI"
    THREAD_ID      = "session-1"

    # ── Step 1 : Build Hybrid Retriever ─────────────────────────
    retriever = create_retriever_from_url(YOUTUBE_URL)
    if retriever is None:
        print("❌ Could not build retriever. Exiting.")
        return

    # ── Step 2 : Build LangGraph Chatbot ────────────────────────
    service = ChatbotService(api_key=GOOGLE_API_KEY)
    app     = service.build_chatbot(retriever)          # EnsembleRetriever injected here

    config  = {"configurable": {"thread_id": THREAD_ID}}
    
    # ── Step 3 : Chat REPL ──────────────────────────────────────
    print("\n✅ Chatbot ready!  Type 'quit' to exit.\n")
    while True:
        user_input = "how many layers are present in this architecture?"
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        last_msg = result["messages"][-1].content
        print(f"\n🤖 Bot:\n{last_msg}\n")
        break


if __name__ == "__main__":
    main()