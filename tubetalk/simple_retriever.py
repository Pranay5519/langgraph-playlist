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
# 3. Build Simple Cosine Similarity Retriever
# ──────────────────────────────────────────────
def build_retriever(chunks, thread_id, *, top_n: int = 5, doc_language: str = "English"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    base_dir = os.path.join("tubetalk", "faiss_indexes")
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"faiss_index_{thread_id}")

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True 
        )
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)

    @traceable(name="Translated_Cosine_Retriever")
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
            
            # Use safety checks for the LLM response
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

        docs = vector_store.similarity_search(search_query, k=top_n)
        return docs

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

    print(f"🔍 Building Cosine Similarity retriever ({doc_language}) …")
    retriever = build_retriever(chunks, thread_id=thread_id, doc_language=doc_language)

    print("✅ Retriever ready.")
    return retriever