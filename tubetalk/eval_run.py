import os
import uuid
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langsmith import traceable
# Import your custom modules
from simple_retriever import create_retriever_from_url
from eval_metrics import ragas_faithfulness_evaluator, ragas_context_recall_evaluator
from chatbot import ChatbotService
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "TubeTalk-Production-Eval"
# ── CONFIG ──────────────────────────────────────────────────
# Ensure these match your LangSmith setup
YOUTUBE_URL = "https://youtu.be/WzvURhaDZqI" 
DATASET_NAME = "TubeTalk.ai EVAL 2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EXPERIMENT_PREFIX = "simple-retriever"
THREAD_ID = "genai_mul_em_002"

retriever_result = create_retriever_from_url(YOUTUBE_URL, doc_language="english" , thread_id=THREAD_ID)
if isinstance(retriever_result, tuple):
    retriever_obj = retriever_result[0]
else:
    retriever_obj = retriever_result

# Initialize the chaStbot service
service = ChatbotService(api_key=GOOGLE_API_KEY)
app = service.build_chatbot(retriever_obj)

# ── TARGET FUNCTION ──────────────────────────────────────────
@traceable(name="TT-Eval-TargetFunc")
def predict_chatbot(inputs: dict):
    user_q = inputs["question"]
    
    # Generate a unique thread ID for a clean state
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Invoke the graph to get the model's answer
    result = app.invoke(
        {"messages": [HumanMessage(content=user_q)]},
        config=config
    )
    
    # Manually fetch the documents here using the global retriever object!
    retrieved_docs = retriever_obj(user_q)
    
    # Return both the generated messages and the fetched documents
    return {
        "messages": result["messages"],
        "documents": retrieved_docs
    }

# ── RUN EVALUATION ───────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting evaluation on dataset: {DATASET_NAME}")
    
    try:
        results = evaluate(
            predict_chatbot,
            data=DATASET_NAME,
            evaluators=[ragas_faithfulness_evaluator, ragas_context_recall_evaluator],
            experiment_prefix=EXPERIMENT_PREFIX,
            metadata={
        "model": "gemini-2.5-flash",
        "retriever": EXPERIMENT_PREFIX,           
        "language": "english",
        "evaluator" : "gemini-2.5-flash"
    }
        )
        print("Evaluation complete! View results in LangSmith.")
    except Exception as e:
        print(f"Evaluation failed: {e}")