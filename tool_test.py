from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent   
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
load_dotenv()


from langchain_groq import ChatGroq

def create_agent(llm, tools, system_prompt):
    system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ]
    )

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent



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

# List of tools the agent can use
tools = [search_tool, wiki_tool]

# List of tools the agent can use
# tools = [wiki_tool]
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

agent = initialize_agent(
    tools=tools,
    llm=chatmodel,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # Allows the agent to retry when parsing fails
)

ret = agent.invoke("in 2 lines tell me about Italy")
print(ret)
print(type(ret))
