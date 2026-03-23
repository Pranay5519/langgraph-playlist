from langsmith import Client
from dotenv import load_dotenv
load_dotenv()
client = Client()

# Define dataset: these are your test cases
dataset_name = "TubeTalk.ai EVAL"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    dataset_id=dataset.id,
examples = [
{
"inputs": {"question": "What is the main goal of the Research Layer in the GenAI Map?"},
"outputs": {"answer": "The Research Layer is where core AI innovation is born, focusing on developing new model architectures (like Transformers or Diffusion models) and learning algorithms (0:15:24)."},
},
{
"inputs": {"question": "How are ideas from the Research Layer converted into working models?"},
"outputs": {"answer": "In the Foundation Layer, research ideas are implemented into code and trained on massive datasets using huge compute resources to create large-scale foundation models (0:35:57)."},
},
{
"inputs": {"question": "how many layers are present in this GenAi archtiecture"},
"outputs": {"answer": "Generative AI architecture is organized into 8 distinct layers"}
},
{
"inputs": {"question": "What is the specific purpose of 'Context Engineering' within the Builder Layer, and how does it differ from prompt engineering?"},
"outputs": {"answer": "Context Engineering involves managing the external data and information passed to the model in real-time, such as switching between GitHub codebases, Jira tickets, or Slack discussions depending on the task, to ensure the model has the correct information (1:24:15)."}
},
{
"inputs": {"question": "According to the map, what is the role of the Operations Layer in ensuring AI application reliability?"},
"outputs": {"answer": "The Operations Layer focuses on deploying and running the software reliably. This includes packaging the application (e.g., Docker), setting up infrastructure (e.g., Kubernetes), implementing deployment strategies like canary or blue-green, managing versions, and handling scaling and load management (1:41:23)."}
},
{
"inputs": {"question": "Explain the feedback loop mechanism described between the User Layer and the Research Layer."},
"outputs": {"answer": "Users provide feedback on AI products (e.g., citing hallucinations). This feedback goes to the Builder Layer, then to the Platform Layer to analyze API metrics, and finally back to the Foundation or Research Layers to improve model alignment, factual accuracy, or core architecture (2:20:19)."}
}
 ]
)