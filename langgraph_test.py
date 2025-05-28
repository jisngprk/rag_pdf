import operator
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
from langchain.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import isTyping, readStreamMsg
import Istype
load_dotenv()
chat = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_tool = Tool(
    name="Wikipedia Search",
    func=wiki.run,
    description="Useful for searching Wikipedia articles."
)

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper)

# Define the search tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Use this tool to search for real-time information from DuckDuckGo."
)


tools = [search_tool, wiki_tool]
tool_node = ToolNode(tools)



class ChatAgentState(TypedDict): 
    messages: Sequence[BaseMessage]


def should_continue(state: ChatAgentState) -> bool:
    messages = state["messages"] 
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:                
        return "continue"
    
def call_model(state: ChatAgentState, config):
    question = state["messages"]

    system = (
        "Assistant는 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
        "Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
      
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="messages"),
    ])

    
    model = chat.bind_tools(tools)
    chain = prompt | model
                
    response = chain.invoke(question)
        
    return {"messages": [response]}

def buildChatAgent():
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile()

chat_app = buildChatAgent()

def run_agent_executor(connectionId, requestId, app, query):
    isTyping(connectionId, requestId)
    
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    message = ""
    for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
        # print('event: ', event)
        
        message = event["messages"][-1]
        # print('message: ', message)

    msg = readStreamMsg(connectionId, requestId, message.content)

    return msg