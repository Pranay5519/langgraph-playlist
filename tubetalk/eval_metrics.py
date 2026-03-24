from typing import Annotated, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# ── GRADER MODELS ──────────────────────────────────────────

class GroundedGrade(BaseModel):
    explanation: str = Field(description="Step-by-step reasoning for the score")
    grounded: bool = Field(description="True if the answer is ONLY based on facts, False if it hallucinates")

class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(description="Reasoning for the relevance score")
    relevant: bool = Field(description="True if facts are related to the question, False otherwise")

class CorrectnessGrade(BaseModel):
    explanation: str = Field(description="Reasoning for comparing prediction to reference")
    correct: bool = Field(description="True if the prediction matches the reference answer's meaning")

# ── INITIALIZE GRADER LLM ──────────────────────────────────
# Using Gemini 2.5 Flash as the "Judge"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ── EVALUATOR FUNCTIONS ────────────────────────────────────

def groundedness(inputs: dict, outputs: dict) -> bool:
    """Checks if the answer is based ONLY on the retrieved documents (Hallucination check)."""
    grader = llm.with_structured_output(GroundedGrade)
    
    # Extract docs and answer from the chatbot output
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = outputs["messages"][-1].content
    
    prompt = f"FACTS: {doc_string}\n\nSTUDENT ANSWER: {answer}"
    
    result = grader.invoke([
        SystemMessage(content="Determine if the answer is grounded in the facts provided. No outside info."),
        HumanMessage(content=prompt)
    ])
    return result.grounded

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Checks if the retrieved chunks a I re actually related to the user's question."""
    grader = llm.with_structured_output(RetrievalRelevanceGrade)
    
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    prompt = f"QUESTION: {inputs['question']}\n\nFACTS: {doc_string}"
    
    result = grader.invoke([
        SystemMessage(content="Determine if the retrieved facts are relevant to the question."),
        HumanMessage(content=prompt)
    ])
    return result.relevant

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Checks if the chatbot's answer matches your manual 'Ground Truth' answer."""
    grader = llm.with_structured_output(CorrectnessGrade)
    
    prediction = outputs["messages"][-1].content
    reference = reference_outputs["answer"]
    
    prompt = f"QUESTION: {inputs['question']}\nREFERENCE: {reference}\nPREDICTION: {prediction}"
    
    result = grader.invoke([
        SystemMessage(content="Act as a strict professor grading an exam based on a reference key."),
        HumanMessage(content=prompt)
    ])
    return result.correct



