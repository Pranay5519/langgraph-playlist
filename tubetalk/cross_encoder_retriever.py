import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

# ──────────────────────────────────────────────
# 1. Transcript Loader
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
            print(f" Error fetching transcript: {e}")
            return None


# ──────────────────────────────────────────────
# 2. Text Splitter
# ──────────────────────────────────────────────
def text_splitter(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

# ──────────────────────────────────────────────
# 3. Build Retriever with Cross-Encoder Reranking
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"

# ──────────────────────────────────────────────
# Cross-Encoder model for reranking
# Fetches a wide candidate pool from FAISS, then
# rescores every (query, chunk) pair for precision.
# ──────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL_SIZE = 10   # How many docs FAISS fetches before reranking

def build_retriever(chunks, thread_id, *, top_n: int = 5, doc_language: str = "English"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Load cross-encoder once — reused for every query in this session
    print(f"⚖️  Loading Cross-Encoder: {CROSS_ENCODER_MODEL} ...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    base_dir = os.path.join("tubetalk", "faiss_indexes")
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"faiss_index_{thread_id}")

    def build_vector_store(docs):
        if os.path.exists(index_path):
            print(f"📁 Loading existing FAISS index from: {index_path}")
            return FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            ), "loaded_from_cache"
        else:
            print("🧠 Requesting Gemini to create new embeddings...")
            vs = FAISS.from_documents(docs, embeddings)
            vs.save_local(index_path)
            print(f"💾 FAISS index saved to: {index_path}")
            return vs, "newly_created"

    vector_store, _ = build_vector_store(chunks)

    @traceable(name="CrossEncoder_Reranker", metadata={"cross_encoder_model": CROSS_ENCODER_MODEL, "candidate_pool_size": CANDIDATE_POOL_SIZE})
    def rerank(query: str, candidate_docs: list, top_n: int):
        """Stage 2 — Scores every (query, chunk) pair and returns the top_n."""
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)          # numpy array of floats

        scored_docs = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True
        )
        reranked = [doc for _, doc in scored_docs[:top_n]]
        top_scores = [round(float(s), 4) for s, _ in scored_docs[:top_n]]
        print(f"✅ Cross-Encoder reranked → top {top_n} docs | scores: {top_scores}")
        return reranked

    @traceable(name="CrossEncoder_Cosine_Retriever", metadata={"embedding_model": EMBEDDING_MODEL, "cross_encoder_model": CROSS_ENCODER_MODEL, "index_path": index_path})
    def retrieve(query: str):
        search_query = query

        if doc_language.lower() == "hindi":
            print(f"🇮🇳 Hindi Document Detected: Optimizing query...")

            translation_prompt = ChatPromptTemplate.from_template(
                "Rewrite the following question into a single optimized search query in Hindi. "
                "Use transliteration for technical terms. Output ONLY the query.\n\n"
                "Original Question: {question}"
            )

            chain = translation_prompt | llm | StrOutputParser()

            try:
                response = chain.invoke({"question": query}).strip()
                if response:
                    search_query = response
                    print(f"🔍 Optimized Hindi Query: {search_query}")
                else:
                    print("⚠️ LLM returned empty string. Falling back.")
            except Exception as e:
                print(f"⚠️ Translation failed: {e}. Using original query.")

        # Final Guardrail: Ensure search_query is NOT empty
        if not search_query or not search_query.strip():
            search_query = query

        # ── Stage 1: Bi-Encoder (FAISS) — fetch wider candidate pool ──
        candidate_docs = vector_store.similarity_search(
            search_query, k=CANDIDATE_POOL_SIZE
        )
        print(f"📦 FAISS returned {len(candidate_docs)} candidates for reranking")

        # ── Stage 2: Cross-Encoder Reranking (traced separately in LangSmith) ──
        reranked_docs = rerank(search_query, candidate_docs, top_n)
        return reranked_docs

    return retrieve

# ──────────────────────────────────────────────
# 4. One-shot pipeline helper
# ──────────────────────────────────────────────
def create_retriever_from_url(youtube_url: str, doc_language: str, thread_id: str):
    """URL → transcript → chunks → retrieve callable"""

    print("📥 Fetching transcript …")
    transcript = load_transcript(youtube_url)
    if not transcript:
        return None

    print("✂️  Splitting into chunks …")
    chunks = text_splitter(transcript)

    print(f"🔍 Building CrossEncoder Retriever ({doc_language}) …")
    retriever = build_retriever(chunks, thread_id=thread_id, doc_language=doc_language)

    print("✅ Retriever ready.")
    return retriever
