import os
from langchain_core.messages import HumanMessage
from langsmith import traceable
from simple_retriever import create_retriever_from_url
from chatbot import ChatbotService
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tubetalk-ai"
RAG_VERSION = "cosine-retriever(g-embed)-hindi-query"

def main():
    # ── Config ──────────────────────────────────────────────────
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    YOUTUBE_URL = "https://youtu.be/WzvURhaDZqI"
    THREAD_ID = "genai"

    # ── Step 1 : Build Hybrid Retriever ─────────────────────────
    retriever = create_retriever_from_url(YOUTUBE_URL, doc_language="hindi",thread_id=THREAD_ID)
    if retriever is None:
        print(" Could not build retriever. Exiting.")
        return

    # ── Step 2 : Build LangGraph Chatbot ────────────────────────
    service = ChatbotService(api_key=GOOGLE_API_KEY)
    app = service.build_chatbot(retriever)

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        },
        "run_name": f"tubetalk-{RAG_VERSION}"
    }
    
    # ── Step 3 : Single Execution ───────────────────────────────
    user_input = "How many Layers are present in this Architecture (genai)"
    
    print(f"\n🚀 Sending Query: {user_input}\n")

    # Invoke the graph once
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # Extract and print the final AI message
    last_msg = result["messages"][-1].content
    print(f"Bot:\n{last_msg}\n")

if __name__ == "__main__":
    main()