"""
retriever.py
────────────────────────────────────────────────────────────
Hybrid Retriever  =  FAISS (dense) + BM25 (sparse)
No custom class. Just fetch from both, deduplicate, return.
────────────────────────────────────────────────────────────
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


# ──────────────────────────────────────────────
# 1. Transcript Loader  (unchanged)
# ──────────────────────────────────────────────

def load_transcript(url: str) -> str | None:

    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    if match:
        video_id = match.group(1)
        try:
            captions = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'hi']).snippets
            data = [f"{item.text} ({item.start})" for item in captions]
            return " ".join(data)
        except Exception as e:
            print(f"❌ Error fetching transcript: {e}")
            return None


# ──────────────────────────────────────────────
# 2. Text Splitter
# ──────────────────────────────────────────────

def text_splitter(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])


# ──────────────────────────────────────────────
# 3. Build Hybrid Retriever
#    Returns a simple callable: query → list[Document]
# ──────────────────────────────────────────────

def build_hybrid_retriever(chunks, *, faiss_k: int = 4, bm25_k: int = 4, top_n: int = 5):
    """
    Builds and returns hybrid_retrieve(query) callable.

    Parameters
    ----------
    chunks  : list[Document]  from text_splitter()
    faiss_k : top-k docs fetched from FAISS
    bm25_k  : top-k docs fetched from BM25
    top_n   : final docs returned after merge
    """

    # ── Dense retriever (FAISS) ──────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store    = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": faiss_k},
    )

    # ── Sparse retriever (BM25) ──────────────────────────────
    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k

    # ── Hybrid callable ──────────────────────────────────────
    def hybrid_retrieve(query: str):

        faiss_docs = faiss_retriever.invoke(query)
        bm25_docs  = bm25_retriever.invoke(query)

        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        print(f"\n📦 FAISS docs ({len(faiss_docs)}):")
        for i, doc in enumerate(faiss_docs, 1):
            print(f"  [{i}] {doc.page_content[:120]}...")

        print(f"\n📦 BM25 docs ({len(bm25_docs)}):")
        for i, doc in enumerate(bm25_docs, 1):
            print(f"  [{i}] {doc.page_content[:120]}...")

        # Merge — deduplicate by content, FAISS first then BM25
        seen   = set()
        merged = []
        for doc in faiss_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        final = merged[:top_n]

        print(f"\n✅ Final merged docs ({len(final)}):")
        for i, doc in enumerate(final, 1):
            print(f"  [{i}] {doc.page_content[:120]}...")
        print(f"{'='*60}\n")

        return final

    return hybrid_retrieve


# ──────────────────────────────────────────────
# 4. One-shot pipeline helper
# ──────────────────────────────────────────────

def create_retriever_from_url(youtube_url: str):
    """URL → transcript → chunks → hybrid_retrieve callable"""

    print("📥 Fetching transcript …")
    transcript = load_transcript(youtube_url)
    if not transcript:
        print("❌ Could not load transcript.")
        return None

    print("✂️  Splitting into chunks …")
    chunks = text_splitter(transcript)

    print("🔍 Building hybrid retriever (FAISS + BM25) …")
    retriever = build_hybrid_retriever(chunks)

    print("✅ Hybrid retriever ready.")
    return retriever