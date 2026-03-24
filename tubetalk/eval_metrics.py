import os
from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Judge LLM
llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
evaluator_llm = LangchainLLMWrapper(llm_instance)

def run_ragas_evaluation(results_list, callbacks=None):
    """
    Core evaluation logic.
    Expects a list of dicts: [{'question': str, 'answer': str, 'contexts': list, 'ground_truth': str}]
    """
    # 1. Prepare the Ragas-specific Dataset
    prepared_data = []
    for item in results_list:
        prepared_data.append({
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item["contexts"],
            "reference": item["ground_truth"]
        })

    eval_ds = EvaluationDataset.from_list(prepared_data)

    # 2. Execute Ragas evaluation
    result = evaluate(
        dataset=eval_ds,
        metrics=[
            faithfulness, 
            answer_relevancy, 
            context_precision, 
            context_recall
        ],
        llm=evaluator_llm,
        callbacks=callbacks
    )
    
    return result