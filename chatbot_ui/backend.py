from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

#load api key
load_dotenv()
# load model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#create a new state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
CONFIG = {'configurable': {'thread_id': 'thread-id'}}

chatbot.invoke(
                {'messages': [HumanMessage(content='user_input')]},
                config=CONFIG,
                stream_mode='messages'
            )
print(chatbot.get_state(config=CONFIG).values['messages'])