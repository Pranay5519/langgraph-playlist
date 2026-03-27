import os
import re
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
# Pydantic model for score-based doc evaluation
# ──────────────────────────────────────────────

class DocEvalOutput(BaseModel):
    """Score-based relevance evaluation for a retrieved chunk."""
    score: float = Field(..., description="Relevance score in [0.0, 1.0].")
    reason: str = Field(..., description="Short reason for the score.")

EMBEDDING_MODEL = "gemini-embedding-001"
UPPER_TH = 0.7   # at least one doc above this → CORRECT
LOWER_TH = 0.3   # all docs below this → INCORRECT; anything in between → AMBIGUOUS

# ──────────────────────────────────────────────
# 1. Transcript Loader  (same as other retrievers)
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
# 2. Text Splitter  (same as other retrievers)
# ──────────────────────────────────────────────

def text_splitter(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

# ──────────────────────────────────────────────
# 3. CRAG Core Logic
#
#  Flow per query:
#    retrieve → score each doc →
#      CORRECT   (any score > 0.7) → keep docs scored > 0.3
#      INCORRECT (all scores < 0.3) → rewrite query → BM25 fallback
#      AMBIGUOUS (in between)       → keep docs scored > 0.3 + BM25 supplement
#    → return final docs
# ──────────────────────────────────────────────

def build_crag_retriever(chunks, thread_id, *, top_n: int = 5, final_k: int = 2, doc_language: str = "English"):
    """
    Builds and returns a crag_retrieve(query) → list[Document] callable.

    CRAG Steps:
      1. Dense retrieval (FAISS cosine similarity)
      2. Score each retrieved doc with an LLM in [0.0, 1.0]
      3. Determine verdict:
           CORRECT   → any score > UPPER_TH (0.7)  → use good docs (score > LOWER_TH)
           INCORRECT → all scores < LOWER_TH (0.3) → rewrite query + BM25 fallback
           AMBIGUOUS → mixed signals               → good docs + BM25 supplement
      4. Return final documents
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0 , api_key = os.getenv("GOOGLE_API_KEY_2"))
    grader_llm = llm.with_structured_output(DocEvalOutput)

    # ── Dense retriever (FAISS) ──────────────────────────────────
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    base_dir = os.path.join("tubetalk", "faiss_indexes")
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"faiss_index_{thread_id}")

    if os.path.exists(index_path):
        print(f"📁 Loading existing FAISS index from: {index_path}")
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("🧠 Requesting Gemini to create new embeddings...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        print(f"💾 FAISS index saved to: {index_path}")

    # ── Sparse retriever (BM25) – used as fallback / supplement ──
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_n

    # ── Grader prompt (score-based) ───────────────────────────────
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
        "You will be given ONE retrieved chunk and a question.\n"
        "Return a relevance score in [0.0, 1.0].\n"
        "- 1.0: this chunk ALONE is fully sufficient to answer the question completely. Reserve this only for the single best possible chunk.\n"
        "- 0.7–0.9: chunk is highly relevant but not perfectly self-contained.\n"
        "- 0.4–0.6: chunk is partially relevant, contains related info but not a direct answer.\n"
        "- 0.0–0.3: chunk is irrelevant or only tangentially related.\n"
        "IMPORTANT: Be very conservative. In most cases, at most ONE chunk should score above 0.8.\n"
        "Also return a short reason.\n"
        "Output JSON only."),
        ("human",
         "Question: {question}\n\nChunk:\n{chunk}")
    ])
    doc_eval_chain = grade_prompt | grader_llm

    # ── Query rewriter prompt ─────────────────────────────────────
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a question re-writer that optimizes a query for document retrieval. "
         "Look at the input question and try to reason about the underlying semantic intent. "
         "Output ONLY the improved question, no explanation."),
        ("human", "Here is the initial question:\n{question}")
    ])
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()

    # ── CRAG callable ─────────────────────────────────────────────
    @traceable(name="CRAG_Retriever", metadata={"embedding_model": EMBEDDING_MODEL, "index_path": index_path})
    def crag_retrieve(query: str):

        @traceable(name="CRAG_Dense_Retrieval")
        def dense_search(q):
            return vector_store.similarity_search(q, k=top_n)

        @traceable(name="CRAG_Score_Documents")
        def score_documents(docs, q) -> tuple[List[float], List[str], list, list]:
            """Score each doc; collect 'good' docs (score > LOWER_TH) and sorted pairs."""
            scores: List[float] = []
            reasons: List[str] = []
            good = []
            pairs = []  # (score, doc)

            for doc in docs:
                out: DocEvalOutput = doc_eval_chain.invoke({
                    "question": q,
                    "chunk": doc.page_content
                })
                scores.append(out.score)
                reasons.append(out.reason)
                pairs.append((out.score, doc))
                print(f"  📊 Score: {out.score:.2f} | Reason: {out.reason[:60]}...")
                if out.score > LOWER_TH:
                    good.append(doc)

            # Sort by score descending so we can easily pick top-k
            scored_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            return scores, reasons, good, scored_pairs

        @traceable(name="CRAG_Query_Rewrite")
        def rewrite_query(q):
            rewritten = rewriter_chain.invoke({"question": q}).strip()
            print(f"  ✏️  Rewritten query: {rewritten}")
            return rewritten if rewritten else q

        @traceable(name="CRAG_BM25_Fallback")
        def bm25_fallback(q):
            return bm25_retriever.invoke(q)

        def dedup(docs):
            """Remove duplicate chunks (same page_content)."""
            seen = set()
            out = []
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    out.append(d)
            return out

        # -- Hindi optimisation (mirrors simple_retriever pattern) --
        search_query = query
        if doc_language.lower() == "hindi":
            print("🇮🇳 Hindi Document Detected: Optimising query...")
            translation_prompt = ChatPromptTemplate.from_template(
                "Rewrite the following question into a single optimized search query in Hindi. "
                "Use transliteration for technical terms. Output ONLY the query.\n\n"
                "Original Question: {question}"
            )
            try:
                search_query = (translation_prompt | llm | StrOutputParser()).invoke(
                    {"question": query}
                ).strip() or query
                print(f"🔍 Optimised Hindi query: {search_query}")
            except Exception as e:
                print(f"⚠️  Hindi optimisation failed: {e}. Using original.")

        if not search_query.strip():
            search_query = query

        # ── Step 1: Dense retrieval ───────────────────────────────
        print("🔎 Step 1: Dense retrieval (FAISS)...")
        retrieved_docs = dense_search(search_query)

        # ── Step 2: Score documents ──────────────────────────────
        print("⚖️  Step 2: Scoring documents...")
        scores, reasons, good_docs, scored_pairs = score_documents(retrieved_docs, query)

        # ── Step 3: Determine verdict & corrective action ─────────
        if any(s > UPPER_TH for s in scores):
            # ✅ CORRECT — at least one highly relevant chunk found
            # Pick only the top final_k docs that scored above UPPER_TH (best first)
            verdict = "CORRECT"
            above_upper = [(s, d) for s, d in scored_pairs if s > UPPER_TH]
            final_docs = [d for _, d in above_upper[:final_k]]
            print(f"✅ Step 3 [{verdict}]: {len(above_upper)} chunk(s) scored > {UPPER_TH}. Returning top {len(final_docs)}.")

        elif len(scores) > 0 and all(s < LOWER_TH for s in scores):
            # ❌ INCORRECT — no chunk is even partially useful
            verdict = "INCORRECT"
            print(f"❌ Step 3 [{verdict}]: All chunks scored < {LOWER_TH}. Rewriting query + BM25 fallback...")
            rewritten_q = rewrite_query(query)
            final_docs = bm25_fallback(rewritten_q)[:final_k]

        else:
            # 🟡 AMBIGUOUS — mixed signals, supplement with BM25
            verdict = "AMBIGUOUS"
            print(f"🟡 Step 3 [{verdict}]: Mixed signals. Keeping good docs + BM25 supplement...")
            rewritten_q = rewrite_query(query)
            bm25_docs = bm25_fallback(rewritten_q)
            final_docs = dedup(good_docs + bm25_docs)[:final_k]

        print(f"  📦 Returning {len(final_docs)} final docs.")
        return final_docs

    return crag_retrieve

# ──────────────────────────────────────────────
# 4. One-shot pipeline helper  (same pattern as other retrievers)
# ──────────────────────────────────────────────

def create_retriever_from_url(youtube_url: str, doc_language: str, thread_id: str):
    """URL → transcript → chunks → crag_retrieve callable"""

    print("📥 Fetching transcript …")
    transcript = load_transcript(youtube_url)
    if not transcript:
        print("❌ Could not load transcript.")
        return None

    print("✂️  Splitting into chunks …")
    chunks = text_splitter(transcript)

    print(f"🔍 Building CRAG retriever ({doc_language}) …")
    retriever = build_crag_retriever(chunks, thread_id=thread_id, doc_language=doc_language)

    print("✅ CRAG retriever ready.")
    return retriever
