import random
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()
client = Client()

dataset_name = "TubeTalk.ai EVAL 2"
dataset = client.create_dataset(dataset_name)

# 1. Your fixed question with all layers joined into a single sentence string
fixed_example = {
    "inputs": {"question": "How many layers are present in this architecture and what are they?"},
    "outputs": {
        "answer": "Based on the 'Map of Generative AI' presented in the video, there are 8 layers that organize the entire AI world: Research Layer (0:15:24) which is the birthplace of core AI innovations, Foundation Layer (0:35:57) where research ideas are converted into large-scale AI models, Platform Layer (0:51:45) providing scalable access via APIs, Builder Layer (1:12:09) where models and logic are combined into workflows, Application Layer (1:30:02) being the final software product, Operation Layer (1:41:23) handling deployment and reliability, Distribution Layer (2:03:20) focusing on business and marketing, and the User Layer (2:11:03) for the end-users."
    },
}

# 2. The pool for the random second question
pool = [
    {
        "inputs": {"question": "What is the main goal of the Research Layer in the GenAI Map?"},
        "outputs": {"answer": "The Research Layer is where core AI innovation is born, focusing on developing new model architectures and learning algorithms (0:15:24)."},
    },

    {
        "inputs": {"question": "why is the Builder Layer considered crucial for bridging raw model intelligence from the Platform Layer with the practical functionality needed in the Application Layer, and what role does Context Engineering play in this process?"},
        "outputs": {"answer": "The Builder Layer takes raw model intelligence and shapes it for specific use cases. It converts LLMs from simple 'next-token predictors' into actionable systems by utilizing tools like RAG and frameworks like LangChain to create functional products. Context Engineering is critical here because it allows developers to manage what information is sent to the model in real-time, switching contexts between different tools (like GitHub or Slack) to enhance the model's capability to execute complex tasks."},
    }
]

# Pick one at random
random_second_example = random.choice(pool)

# Combine and upload
final_examples = [fixed_example, random_second_example]

client.create_examples(
    dataset_id=dataset.id,
    inputs=[e["inputs"] for e in final_examples],
    outputs=[e["outputs"] for e in final_examples]
)

print(f"Dataset '{dataset_name}' created with 2 examples.")