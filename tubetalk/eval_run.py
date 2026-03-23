import os
import uuid
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langsmith import traceable
# Import your custom modules
from retriever import create_retriever_from_url
from chatbot import ChatbotService
from eval_metrics import correctness

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "TubeTalk-Production-Eval"
# ── CONFIG ──────────────────────────────────────────────────
# Ensure these match your LangSmith setup
YOUTUBE_URL = "https://youtu.be/WzvURhaDZqI" 
DATASET_NAME = "TubeTalk.ai EVAL"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

THREAD_ID = "genai"

retriever_result = create_retriever_from_url(YOUTUBE_URL, doc_language="hindi" , thread_id=THREAD_ID)
if isinstance(retriever_result, tuple):
    retriever_obj = retriever_result[0]
else:
    retriever_obj = retriever_result

# Initialize the chatbot service
service = ChatbotService(api_key=GOOGLE_API_KEY)
app = service.build_chatbot(retriever_obj)

# ── TARGET FUNCTION ──────────────────────────────────────────
@traceable(name="TT-Eval-TargetFunc")
def predict_chatbot(inputs: dict):
    user_q = inputs["question"]
    
    # Generate a unique thread ID for a clean state
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Invoke the graph to get the answer
    result = app.invoke(
        {"messages": [HumanMessage(content=user_q)]},
        config=config
    )
    
    # Only return the messages; ignoring 'documents' for now
    return {
        "messages": result["messages"]
    }
# ── RUN EVALUATION ───────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting evaluation on dataset: {DATASET_NAME}")
    
    try:
        results = evaluate(
            predict_chatbot,
            data=DATASET_NAME,
            evaluators=[correctness],
            experiment_prefix="qwen-2.5-Hybrid-RAG-Hindi-Doc",
            metadata={
        "model": "ollama-qwen3:latest",
        "retriever": "hybrid-ensemble",
        "language": "hindi",
        "evaluator" : "ollama-qwen3:latest"
    }
        )
        print("Evaluation complete! View results in LangSmith.")
    except Exception as e:
        print(f"Evaluation failed: {e}")