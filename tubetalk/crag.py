import os
import re
import asyncio
from typing import List, Tuple
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
# 3. Optimized CRAG Core Logic
#
#  Key improvements over original:
#    - Parallel LLM grading via asyncio (5x faster for top_n=5)
#    - AMBIGUOUS path now supplements with BM25 (matches CRAG paper intent)
#    - Dedup applied on ALL verdict paths, not just INCORRECT
#    - final_k respected on CORRECT_MIXED and AMBIGUOUS paths
#    - Separate LLM instances for grading (temp=0) and rewriting (temp=0.3)
#    - BM25 fallback in INCORRECT now uses the Hindi-optimised query if applicable
# ──────────────────────────────────────────────

def build_crag_retriever(chunks, thread_id, *, top_n: int = 5, final_k: int = 4, doc_language: str = "English"):
    """
    Builds and returns a crag_retrieve(query) → list[Document] callable.

    CRAG Steps:
      1. Dense retrieval (FAISS cosine similarity)
      2. Score retrieved docs in PARALLEL with an LLM in [0.0, 1.0]
      3. Determine verdict:
           CORRECT_STRONG → ≥4 docs > UPPER_TH  → top 3
           CORRECT_MIXED  → 1-3 docs > UPPER_TH  → top final_k above LOWER_TH
           AMBIGUOUS      → none > UPPER_TH but some > LOWER_TH
                            → good dense docs + BM25 supplement (deduped)
           INCORRECT      → all < LOWER_TH → rewrite query + BM25 fallback
      4. Dedup and return final documents
    """

    # FIX: separate LLM instances so rewriter can use a slightly higher temp
    grader_base_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, api_key=os.getenv("GOOGLE_API_KEY_2")
    )
    rewriter_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.3, api_key=os.getenv("GOOGLE_API_KEY_2")
    )
    grader_llm = grader_base_llm.with_structured_output(DocEvalOutput)

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

    # ── Sparse retriever (BM25) – fallback / supplement ──
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_n

    # ── Grader prompt ─────────────────────────────────────────────
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
         "You will be given ONE retrieved chunk and a question.\n"
         "Return a relevance score in [0.0, 1.0].\n"
         "- 1.0: this chunk ALONE is fully sufficient to answer the question completely.\n"
         "- 0.7–0.9: chunk is highly relevant but not perfectly self-contained.\n"
         "- 0.4–0.6: chunk is partially relevant, contains related info but not a direct answer.\n"
         "- 0.0–0.3: chunk is irrelevant or only tangentially related.\n"
         "IMPORTANT: Be very conservative. In most cases, at most ONE chunk should score above 0.8.\n"
         "Also return a short reason.\n"
         "Output JSON only."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}")
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
    rewriter_chain = rewrite_prompt | rewriter_llm | StrOutputParser()

    # ── CRAG callable ─────────────────────────────────────────────
    @traceable(name="CRAG_Retriever", metadata={"embedding_model": EMBEDDING_MODEL, "index_path": index_path})
    def crag_retrieve(query: str):

        @traceable(name="CRAG_Dense_Retrieval")
        def dense_search(q: str):
            return vector_store.similarity_search(q, k=top_n)

        # FIX: grade all docs in parallel using asyncio instead of serially
        @traceable(name="CRAG_Score_Documents_Parallel")
        def score_documents_parallel(docs, q: str) -> Tuple[List[float], List[str], list, list]:
            """Score all docs concurrently; collect 'good' docs (score > LOWER_TH)."""

            async def grade_one(doc) -> DocEvalOutput:
                return await doc_eval_chain.ainvoke({
                    "question": q,
                    "chunk": doc.page_content
                })

            async def grade_all():
                return await asyncio.gather(*[grade_one(d) for d in docs])

            results: List[DocEvalOutput] = asyncio.run(grade_all())

            scores, reasons, good, pairs = [], [], [], []
            for doc, out in zip(docs, results):
                scores.append(out.score)
                reasons.append(out.reason)
                pairs.append((out.score, doc))
                print(f"  📊 Score: {out.score:.2f} | Reason: {out.reason[:60]}...")
                if out.score > LOWER_TH:
                    good.append(doc)

            scored_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            return scores, reasons, good, scored_pairs

        @traceable(name="CRAG_Query_Rewrite")
        def rewrite_query(q: str) -> str:
            rewritten = rewriter_chain.invoke({"question": q}).strip()
            print(f"  ✏️  Rewritten query: {rewritten}")
            return rewritten if rewritten else q

        @traceable(name="CRAG_BM25_Fallback")
        def bm25_fallback(q: str):
            return bm25_retriever.invoke(q)

        def dedup(docs):
            """Remove duplicate chunks by content — applied on ALL paths."""
            seen, out = set(), []
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    out.append(d)
            return out

        # ── Hindi optimisation ────────────────────────────────────
        search_query = query
        if doc_language.lower() == "hindi":
            print("🇮🇳 Hindi Document Detected: Optimising query...")
            translation_prompt = ChatPromptTemplate.from_template(
                "Rewrite the following question into a single optimized search query in Hindi. "
                "Use transliteration for technical terms. Output ONLY the query.\n\n"
                "Original Question: {question}"
            )
            try:
                search_query = (translation_prompt | rewriter_llm | StrOutputParser()).invoke(
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

        # ── Step 2: Score documents (parallel) ───────────────────
        print("⚖️  Step 2: Scoring documents in parallel...")
        scores, reasons, good_docs, scored_pairs = score_documents_parallel(retrieved_docs, query)

        # ── Step 3: Determine verdict & corrective action ─────────
        above_upper     = [(s, d) for s, d in scored_pairs if s > UPPER_TH]
        all_above_lower = [(s, d) for s, d in scored_pairs if s > LOWER_TH]

        if len(above_upper) >= 4:
            # ✅ CORRECT_STRONG — ≥4 docs clearly relevant → top 3
            verdict = "CORRECT_STRONG"
            final_docs = dedup([d for _, d in above_upper[:3]])
            print(f"✅ Step 3 [{verdict}]: {len(above_upper)} docs > {UPPER_TH}. Returning top 3.")

        elif len(above_upper) > 0:
            # ✅ CORRECT_MIXED — 1–3 docs > 0.7 → top final_k above LOWER_TH
            # FIX: was hardcoded to 4; now respects final_k param
            verdict = "CORRECT_MIXED"
            final_docs = dedup([d for _, d in all_above_lower[:final_k]])
            print(f"✅ Step 3 [{verdict}]: {len(above_upper)} doc(s) > {UPPER_TH}. Returning top {final_k} (score > {LOWER_TH}).")

        elif len(all_above_lower) > 0:
            # 🟡 AMBIGUOUS — none > 0.7 but some partially relevant
            # FIX: now supplements with BM25 (original CRAG paper intent)
            verdict = "AMBIGUOUS"
            dense_good  = [d for _, d in all_above_lower]
            # FIX: BM25 uses the language-optimised search_query, not raw query
            bm25_docs   = bm25_fallback(search_query)
            combined    = dedup(dense_good + bm25_docs)[:final_k]
            final_docs  = combined
            print(f"🟡 Step 3 [{verdict}]: No doc > {UPPER_TH}. Returning {len(dense_good)} dense + BM25 supplement (deduped → {len(final_docs)}).")

        else:
            # ❌ INCORRECT — everything below 0.3 → rewrite + BM25 fallback
            # FIX: BM25 now uses the rewritten query, not the language-optimised one
            verdict = "INCORRECT"
            print(f"❌ Step 3 [{verdict}]: All docs < {LOWER_TH}. Rewriting query + BM25 fallback...")
            rewritten_q = rewrite_query(query)
            final_docs  = dedup(bm25_fallback(rewritten_q))[:final_k]

        print(f"  📦 Returning {len(final_docs)} final docs.")
        return final_docs

    return crag_retrieve

# ──────────────────────────────────────────────
# 4. One-shot pipeline helper
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