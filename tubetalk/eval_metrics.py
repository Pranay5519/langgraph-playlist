from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from ragas.dataset_schema import SingleTurnSample
# 1. Initialize your Gemini Models
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. Wrap them for Ragas
ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

# 3. override the default LLMs/Embeddings inside the Ragas metrics
faithfulness.llm = ragas_llm

answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings

context_precision.llm = ragas_llm
context_recall.llm = ragas_llm



def ragas_faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    question = inputs["question"]
    answer = outputs["messages"][-1].content
    contexts = [doc.page_content for doc in outputs["documents"]]
    
    # Create a SingleTurnSample object
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    # Use single_turn_score instead of score_single
    score = faithfulness.single_turn_score(sample)
    
    return {"key": "ragas_faithfulness", "score": score}


def ragas_context_recall_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    question = inputs["question"]
    answer = outputs["messages"][-1].content
    contexts = [doc.page_content for doc in outputs["documents"]]
    ground_truth = reference_outputs["answer"] 
    
    # Create a SingleTurnSample with reference data
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    
    # Use single_turn_score instead of score_single
    score = context_recall.single_turn_score(sample)
    
    return {"key": "ragas_context_recall", "score": score}



def ragas_answer_relevancy_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    question = inputs["question"]
    answer = outputs["messages"][-1].content
    contexts = [doc.page_content for doc in outputs["documents"]]
    
    # Create a SingleTurnSample object
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    score = answer_relevancy.single_turn_score(sample)
    
    return {"key": "ragas_answer_relevancy", "score": score}
def ragas_context_precision_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    question = inputs["question"]
    answer = outputs["messages"][-1].content
    contexts = [doc.page_content for doc in outputs["documents"]]
    ground_truth = reference_outputs["answer"] 
    
    # Precision relies heavily on the 'reference' (Ground Truth) to check if ranked correctly
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    
    score = context_precision.single_turn_score(sample)
    
    return {"key": "ragas_context_precision", "score": score}
def ragas_answer_correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    question = inputs["question"]
    answer = outputs["messages"][-1].content
    contexts = [doc.page_content for doc in outputs["documents"]]
    ground_truth = reference_outputs["answer"] 
    
    # Answer correctness compares the response to the reference
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    
    score = answer_correctness.single_turn_score(sample)
    
    return {"key": "ragas_answer_correctness", "score": score}