
import os
import sys
import uuid
from langchain_core.messages import HumanMessage
from chatbot import ChatbotService
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tubetalk-ai"

# ──────────────────────────────────────────────
# Defaults  (press Enter at any prompt to use)
# ──────────────────────────────────────────────
DEFAULT_TECHNIQUE  = "1"                                   # simple
DEFAULT_URL        = "https://youtu.be/WzvURhaDZqI"        # GenAI Map video
DEFAULT_LANGUAGE   = "1"                                   # English
DEFAULT_THREAD_ID  = "genai_mul_em_002"                    # reuses cached FAISS index

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

TECHNIQUES = {
    "1": ("simple",  "Simple Cosine Similarity (FAISS) + CrossEncoder Reranker"),
    "2": ("hybrid",  "Hybrid Retriever (FAISS + BM25 + MultiQuery)"),
    "3": ("crag",    "Corrective RAG (FAISS + LLM Grader + BM25 Fallback)"),
}

def pick_technique() -> tuple[str, str]:
    """Prompt user to choose a retrieval technique. Returns (key, label)."""
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│              TubeTalk — Choose Retriever                  │")
    print("├──────────────────────────────────────────────────────────┤")
    for k, (_, label) in TECHNIQUES.items():
        print(f"│  [{k}] {label:<54}│")
    print("└──────────────────────────────────────────────────────────┘")
    print(f"   (Press Enter to use default: [{DEFAULT_TECHNIQUE}] {TECHNIQUES[DEFAULT_TECHNIQUE][1]})")

    while True:
        choice = input("Enter choice (1 / 2 / 3): ").strip() or DEFAULT_TECHNIQUE
        if choice in TECHNIQUES:
            key, label = TECHNIQUES[choice]
            print(f"✅ Selected: {label}\n")
            return key, label
        print("  ⚠️  Invalid choice. Please enter 1, 2, or 3.")


def pick_language() -> str:
    """Prompt user for document language."""
    print("🌐 What language is the video in?")
    print(f"   [1] English   [2] Hindi   (Press Enter for default: [{DEFAULT_LANGUAGE}] English)")
    while True:
        choice = input("Enter choice (1 / 2): ").strip() or DEFAULT_LANGUAGE
        if choice == "1":
            return "english"
        elif choice == "2":
            return "hindi"
        print("  ⚠️  Please enter 1 or 2.")


def build_retriever(technique: str, youtube_url: str, doc_language: str, thread_id: str):
    """Import and call the correct create_retriever_from_url based on technique."""
    if technique == "simple":
        from simple_retriever import create_retriever_from_url
    elif technique == "hybrid":
        from hybrid_retriever import create_retriever_from_url
    elif technique == "crag":
        from crag import create_retriever_from_url
    else:
        raise ValueError(f"Unknown technique: {technique}")

    return create_retriever_from_url(youtube_url, doc_language=doc_language, thread_id=thread_id)


def print_banner(technique_label: str, youtube_url: str):
    print("\n" + "═" * 52)
    print(f"  🎬  TubeTalk Chat")
    print(f"  🔧  Retriever  : {technique_label}")
    print(f"  🔗  Video      : {youtube_url}")
    print(f"  💬  Type your question. Type 'exit' to quit.")
    print("═" * 52 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found in .env. Exiting.")
        sys.exit(1)

    # ── Step 1: Choose technique ─────────────────────────────────
    technique_key, technique_label = pick_technique()

    # ── Step 2: YouTube URL ──────────────────────────────────────
    youtube_url = input(f"🔗 Paste the YouTube URL (Enter for default): ").strip() or DEFAULT_URL
    print(f"   ▶ Using URL: {youtube_url}")

    # ── Step 3: Language ─────────────────────────────────────────
    doc_language = pick_language()

    # ── Step 4: Build retriever ──────────────────────────────────
    thread_id = DEFAULT_THREAD_ID
    print(f"\n🔧 Building retriever (thread: {thread_id})...")

    retriever = build_retriever(technique_key, youtube_url, doc_language, thread_id)
    if retriever is None:
        print("❌ Could not build retriever. Check the URL and try again.")
        sys.exit(1)

    # ── Step 5: Build chatbot ────────────────────────────────────
    service = ChatbotService(api_key=GOOGLE_API_KEY)
    app     = service.build_chatbot(retriever)

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"tubetalk-{technique_key}",
    }

    print_banner(technique_label, youtube_url)

    # ── Step 6: Chat loop ────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting TubeTalk. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print("👋 Exiting TubeTalk. Goodbye!")
            break

        # ── Special commands ─────────────────────────────────────
        if user_input.lower() == "!switch":
            print("\n🔄 Switching retriever technique...")
            technique_key, technique_label = pick_technique()
            thread_id = f"{technique_key}_{uuid.uuid4().hex[:8]}"
            print(f"🔧 Rebuilding retriever (thread: {thread_id})...")
            retriever = build_retriever(technique_key, youtube_url, doc_language, thread_id)
            if retriever is None:
                print("❌ Could not rebuild retriever.")
                continue
            app    = service.build_chatbot(retriever)
            config = {
                "configurable": {"thread_id": thread_id},
                "run_name": f"tubetalk-{technique_key}",
            }
            print(f"✅ Switched to: {technique_label}\n")
            continue

        if user_input.lower() == "!help":
            print("\n📖 Commands:")
            print("   !switch  — switch retrieval technique mid-session")
            print("   !info    — show current session info")
            print("   exit     — quit the chat\n")
            continue

        if user_input.lower() == "!info":
            print(f"\n  🔧 Technique : {technique_label}")
            print(f"  🔗 Video     : {youtube_url}")
            print(f"  🌐 Language  : {doc_language}")
            print(f"  🧵 Thread ID : {thread_id}\n")
            continue

        # ── Invoke chatbot ────────────────────────────────────────
        print("⏳ Thinking...\n")
        try:
            result   = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            last_msg = result["messages"][-1].content
            print(f"🤖 Bot:\n{last_msg}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
