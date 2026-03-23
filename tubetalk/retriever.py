import re
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI 
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()
import os
from pydantic import BaseModel, Field
from typing import List

class SingleQueryOutput(BaseModel):
    """A single optimized search query for keyword matching."""
    query: str = Field(..., description="A single search query optimized for BM25.")

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
            print(f" Error fetching transcript: {e}")
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

def build_hybrid_retriever(chunks,thread_id, *, faiss_k: int = 4, bm25_k: int = 4, top_n: int = 5 , doc_language :str = "English"):
    """
    Builds and returns hybrid_retrieve(query) callable.

    Parameters
    ----------
    chunks  : list[Document]  from text_splitter()
    faiss_k : top-k docs fetched from FAISS
    bm25_k  : top-k docs fetched from BM25
    top_n   : final docs returned after merge
    """
    # define Structured LLMS
    llm = ChatOllama(model="qwen3:latest")

    bm25_structured_llm = llm.with_structured_output(SingleQueryOutput)
    
    # ── Dense retriever (FAISS) ──────────────────────────────
    embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-base",
                    model_kwargs={"device": "cpu"}
                )

    base_dir = os.path.join("tubetalk", "faiss_indexes")
    
    # 2. Automatically create the folders if they are missing
    os.makedirs(base_dir, exist_ok=True)

    # 3. Set the final path for this specific video
    index_path = os.path.join(base_dir, f"faiss_index_{thread_id}")

    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {index_path}")
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True # Necessary for local loading
        )
    else:
        print("🧠 Creating new embeddings and FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        print(f"💾 FAISS index saved to: {index_path}")
    
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question", "language"], 
    template="""You are an AI language model assistant. 
    Your task is to generate exactly five (5) different versions of the user question 
    to retrieve relevant documents from a vector database. 
    
    STRICT RULES:
    1. Output ONLY the questions.
    2. One question per line.
    3. DO NOT include any introductory text, concluding text, or explanations.
    4. DO NOT use numbering (1, 2, 3) or bullet points.
    5. DO NOT use bold text (**).
    6. All generated queries MUST be in {language}.
    
    Original question: {question}""",
)
    custom_prompt = QUERY_PROMPT.partial(language = doc_language)
    multiquery_retriever = MultiQueryRetriever.from_llm(
                        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
                        llm=ChatOllama(model="qwen3:latest"),
                        prompt=custom_prompt ,
                        parser_key="lines"  # ensures that any accidental whitespace or empty lines are cleaned up before the search
                    )


    # ── Sparse retriever (BM25) ──────────────────────────────
    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k
    
    # ── Hybrid callable ──────────────────────────────────────
    @traceable(name="Hybrid_Retriever")
    def hybrid_retrieve(query: str):
        @traceable(name="embeddings")
        def dense_search(q):
            return multiquery_retriever.invoke(q)
        @traceable(name="BM25 Sparse Retrieval")
        def bm25_search(q):
            return bm25_retriever.invoke(q)
        
        faiss_docs = dense_search(query)
        
        if doc_language=='hindi':
            BM25_HINDI_PROMPT = """
            Rewrite the following question into a single optimized search query in Hindi.
            
            CRITICAL: 
            - Use transliteration for technical terms (e.g. 'layers' -> 'लेयर्स', 'filter' -> 'फ़िल्टर').
            - DO NOT use formal dictionary translations (e.g. avoid 'परतें').
            - Focus on matching keywords as they are likely spoken in the video.
            
            Original Question: {question}
            """
            formatted_prompt = BM25_HINDI_PROMPT.format(question=query)
            
            # 3. Invoke the structured LLM and extract the string
            hindi_query_obj: SingleQueryOutput = bm25_structured_llm.invoke(formatted_prompt)
            bm25_docs = bm25_search(hindi_query_obj.query)
            
        else:
            bm25_docs  = bm25_search(query)
            
        @traceable(name="Hybrid Merge")
        def merge_results(f_docs, b_docs):
            seen = set()
            merged = []
            for doc in f_docs + b_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
            return merged[:top_n]

        final = merge_results(faiss_docs, bm25_docs)
        return final

    return hybrid_retrieve


# ──────────────────────────────────────────────
# 4. One-shot pipeline helper
# ──────────────────────────────────────────────

def create_retriever_from_url(youtube_url: str , doc_language,thread_id):
    """URL → transcript → chunks → hybrid_retrieve callable"""

    print("📥 Fetching transcript …")
    transcript = load_transcript(youtube_url)
    if not transcript:
        print("❌ Could not load transcript.")
        return None

    print("✂️  Splitting into chunks …")
    chunks = text_splitter(transcript)

    print("🔍 Building hybrid retriever (FAISS + BM25) …")
    retriever = build_hybrid_retriever(chunks,thread_id=thread_id , doc_language=doc_language)

    print("✅ Hybrid retriever ready.")
    return retriever