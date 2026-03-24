import os
import uuid
from dotenv import load_dotenv
from langsmith import Client, traceable
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage

# Custom imports
from retriever import create_retriever_from_url
from chatbot import ChatbotService
from eval_metrics import run_ragas_evaluation  # <--- Our new module

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "TubeTalk-Ragas-Production"

# --- CONFIG ---
YOUTUBE_URL = "https://youtu.be/WzvURhaDZqI" 
DATASET_NAME = "TubeTalk.ai EVAL"
THREAD_ID = "genai"

# Initialize Services
retriever_result = create_retriever_from_url(YOUTUBE_URL, doc_language="hindi", thread_id=THREAD_ID)
retriever_obj = retriever_result[0] if isinstance(retriever_result, tuple) else retriever_result
service = ChatbotService(api_key=os.getenv("GOOGLE_API_KEY"))
app = service.build_chatbot(retriever_obj)

@traceable(name="TT-Eval-TargetFunc")
def predict_chatbot(inputs: dict):
    # Standard Chatbot execution
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = app.invoke({"messages": [HumanMessage(content=inputs["question"])]}, config=config)
    
    return {
        "answer": result["messages"][-1].content,
        "contexts": [doc.page_content for doc in result.get("documents", [])]
    }

if __name__ == "__main__":
    client = Client()
    examples = list(client.list_examples(dataset_name=DATASET_NAME))
    
    results_for_ragas = []
    print(f"🚀 Running evaluation on {len(examples)} test cases...")

    for ex in examples:
        # 1. Run the Chatbot
        outputs = predict_chatbot(ex.inputs)
        
        # 2. Collect for the Metrics Engine
        results_for_ragas.append({
            "question": ex.inputs["question"],
            "answer": outputs["answer"],
            "contexts": outputs["contexts"],
            "ground_truth": ex.outputs["answer"]
        })

    # 3. Call the external metrics file
    tracer = LangChainTracer(project_name="TubeTalk-Ragas-Run")
    final_scores = run_ragas_evaluation(results_for_ragas, callbacks=[tracer])

    print("\n✅ Ragas Evaluation Complete!")
    print(final_scores)