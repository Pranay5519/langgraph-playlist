import os
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# --- CONFIG ---
DATASET_NAME = "TubeTalk.ai EVAL"

def check_langsmith_connection():
    client = Client()
    
    print(f"🔍 Accessing LangSmith Dataset: '{DATASET_NAME}'...")
    
    try:
        # 1. Fetch examples
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        
        if not examples:
            print("⚠️ Warning: Connection successful, but the dataset appears to be empty.")
            return

        print(f"✅ Success! Found {len(examples)} test cases.\n")
        print(f"{'#':<3} | {'Question Summary':<50} | {'Ground Truth Status'}")
        print("-" * 85)

        results_for_ragas = []

        # 2. Improved Loop with validation
        for i, ex in enumerate(examples, 1):
            print("question")
            # Extract data with safety defaults
            question = ex.inputs.get("question", "N/A")
            ground_truth = ex.outputs.get("answer", "N/A")

            # Basic Validation: Check if data is missing
            status = "✅ OK" if ground_truth != "N/A" else "❌ Missing Output"

            # Print a snippet for visual confirmation
            print(f"{i:<3} | {question[:47] + '...':<50} | {status}")

            results_for_ragas.append({
                "question": question,
                "ground_truth": ground_truth
            })

        print(f"\n🚀 Client check complete. Total records prepared: {len(results_for_ragas)}")
        return results_for_ragas

    except Exception as e:
        print(f"❌ LangSmith Client Error: {e}")
        print("Check your LANGCHAIN_API_KEY and internet connection.")

if __name__ == "__main__":
    check_langsmith_connection()