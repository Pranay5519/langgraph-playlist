"""
chatbot.py
────────────────────────────────────────────────────────────
LangGraph chatbot that uses the Hybrid Retriever.
Only the retriever wiring changes from your original code.
────────────────────────────────────────────────────────────
"""

import sqlite3
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable
from pydantic import BaseModel , Field
from typing import List, Optional
from langchain_ollama import ChatOllama

# ──────────────────────────────────────────────
# Pydantic structured-output model
# (replaces your app.pydantic_models.chatbot_model.AnsandTime)
# ──────────────────────────────────────────────

class AnsandTime(BaseModel):
    # Compulsory answer, strictly in English
    answer: str = Field(
        ..., 
        description="The detailed answer to the user's question. MUST be written in English only."
    )
    # Single timestamp instead of a list
    timestamp: Optional[float] = Field(
        None, 
        description="The specific start time (in seconds) where this information appears in the video."
    )
    # Optional code
    code: Optional[str] = Field(
        None, 
        description="Any code snippets mentioned in the transcript. Leave null if no code is present."
    )

# ──────────────────────────────────────────────
# Chat State
# ──────────────────────────────────────────────

class ChatState(TypedDict):
    # add_messages appends each turn instead of overwriting
    messages: Annotated[list[BaseMessage], add_messages]

# ──────────────────────────────────────────────
# Chatbot Service
# ──────────────────────────────────────────────
class ChatbotService:
    """
    LangGraph chatbot that answers questions about a YouTube video.
    Uses a Hybrid Retriever (FAISS + BM25) for context retrieval.
    Persists conversation history in SQLite.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0,
        db_path: str = "tubetalk.db",
    ):
        # ── LLM ────────────────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
        self.structured_model = self.llm.with_structured_output(AnsandTime)

        # ── System prompt ───────────────────────────────────────
        self.system_message = SystemMessage(
        content=(
            "You are the YouTuber from the video. "
            "1. Answer ONLY using the transcript context provided. "
            "2. Your answer must be written in ENGLISH, even if the transcript is in another language. "
            "3. Provide exactly one relevant timestamp if available."
        )
    )

        # ── Persistent memory (SQLite) ──────────────────────────
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn=conn)

    # ── Graph node ──────────────────────────────────────────────
    @traceable(name="Chat Node")
    def _chat_node(self, state: ChatState, retriever):
        """
        Single graph node:
          1. Extract last user question from state
          2. Retrieve relevant chunks via Hybrid Retriever
          3. Build prompt = system + context + full history
          4. Call structured LLM
          5. Return AI message
        """
        user_question = state["messages"][-1].content

        # Hybrid retrieval  ← only change from original
        retrieved_chunks = retriever(user_question)
        #print(retrieved_chunks)
        context = "\n\n".join(doc.page_content for doc in retrieved_chunks)

        messages = (
            [
                self.system_message,
                SystemMessage(content=f"Transcript context:\n{context}"),
            ]
            + state["messages"]
        )

        response: AnsandTime = self.structured_model.invoke(messages)

        ai_text = (
            f"Answer: {response.answer}\n"
            f"Timestamp: {response.timestamp}s\n"
        )
        
        # Only add code block if it exists
        if response.code:
            ai_text += f"\nCode:\n```python\n{response.code}\n```"

        return {"messages": [AIMessage(content=ai_text)]}

    # ── Graph builder ────────────────────────────────────────────
    def build_chatbot(self, retriever):
        """
        Compile the LangGraph with the hybrid retriever baked in.

        Parameters
        ----------
        retriever : EnsembleRetriever   (from retriever.py)

        Returns
        -------
        Compiled LangGraph app  (call with .invoke() or .stream())
        """
        graph = StateGraph(ChatState)

        graph.add_node(
            "chat_node",
            lambda state: self._chat_node(state, retriever),
        )
        graph.add_edge(START, "chat_node")
        graph.add_edge("chat_node", END)

        return graph.compile(checkpointer=self.checkpointer)