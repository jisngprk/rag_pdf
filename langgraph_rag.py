# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from src.core.embedder import Embedder
from langchain_groq import ChatGroq
from src.core.rag import RAG
from src.core.loader import PdfLoader
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig
import uuid
def random_uuid():
    return str(uuid.uuid4())


file_path = "./data/SPRi AI Brief 5월호 산업동향.pdf"

# API 키 정보 로드
load_dotenv()

loader = PdfLoader(clean_model_name="llama-3.1-8b-instant")
embedder = Embedder(data_path='./data')
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

documents = loader.run(file_path, is_clean=False)
vectorDB = embedder.run(documents)
rag = RAG(vectorDB, chatmodel)
# 직접 retriever 사용
# docs = rag.kb_retriever.invoke("What is the main topic of the document?")
# print(docs)
retriever_tool = create_retriever_tool(
    rag.kb_retriever, 
    name="rag_retriever",
    description="A tool for retrieving information from the RAG")

# ret = retriever_tool.invoke({
#     "query": "What is the main topic of the document?",
#     "chat_history": []
# })

tools = [retriever_tool]
# 에이전트 상태를 정의하는 타입 딕셔너리, 메시지 시퀀스를 관리하고 추가 동작 정의
class AgentState(TypedDict):
    # add_messages reducer 함수를 사용하여 메시지 시퀀스를 관리
    messages: Annotated[Sequence[BaseMessage], add_messages]

class grade(BaseModel):
    """A binary score for relevance checks"""

    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the question or 'no' if it is not."
    )

def grade_documents(state) -> Literal["generate", "rewrite"]:
    chatmodel_with_grade = chatmodel.with_structured_output(grade)
    prompt = PromptTemplate.from_template(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question
        """
    )
    chain = prompt | chatmodel_with_grade
    messages = state["messages"]
    # 가장 마지막 메시지 추출
    last_message = messages[-1]

    # 원래 질문 추출
    question = messages[0].content

    # 검색된 문서 추출
    retrieved_docs = last_message.content
    print(retrieved_docs)
    print('-'*100)
    scored_result = chain.invoke({"question": question, "context": retrieved_docs})
    print(scored_result)
    score = scored_result.binary_score

    # 관련성 여부에 따른 결정
    if score == "yes":
        print("==== [DECISION: DOCS RELEVANT] ====")
        return "generate"
    else:
        print("==== [DECISION: DOCS NOT RELEVANT] ====")
        print(score)
        return "rewrite"


def agent(state):
    # 현재 상태에서 메시지 추출
    messages = state["messages"]

    # retriever tool 바인딩
    chatmodel_with_tools = chatmodel.bind_tools(tools)

    # 에이전트 응답 생성
    response = chatmodel_with_tools.invoke(messages)
    print('-'*100)
    print(response)
    print('-'*100)
    # 기존 리스트에 추가되므로 리스트 형태로 반환
    return {"messages": [response]}

def rewrite(state):
    print("==== [QUERY REWRITE] ====")
    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    # 원래 질문 추출
    question = messages[0].content

    # 질문 개선을 위한 프롬프트 구성
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # LLM 모델로 질문 개선
    # model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    # Query-Transform 체인 실행
    response = chatmodel.invoke(msg)

    # 재작성된 질문 반환
    return {"messages": [response]}


def generate(state):
    # 현재 상태에서 메시지 추출
    messages = state["messages"]

    # 원래 질문 추출
    question = messages[0].content

    # 가장 마지막 메시지 추출
    docs = messages[-1].content

    # RAG 프롬프트 템플릿 가져오기
    prompt = hub.pull("teddynote/rag-prompt")
    print(prompt)
    print('-'*100)

    # LLM 모델 초기화
    # llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)

    # RAG 체인 구성
    rag_chain = prompt | chatmodel | StrOutputParser()

    # 답변 생성 실행
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# AgentState 기반 상태 그래프 워크플로우 초기화
workflow = StateGraph(AgentState)

# 노드 정의
workflow.add_node("agent", agent)  # 에이전트 노드
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # 검색 노드
workflow.add_node("rewrite", rewrite)  # 질문 재작성 노드
workflow.add_node("generate", generate)  # 관련 문서 확인 후 응답 생성 노드

# 엣지 연결
workflow.add_edge(START, "agent")

# 검색 여부 결정을 위한 조건부 엣지 추가
workflow.add_conditional_edges(
    "agent",
    # 에이전트 결정 평가
    tools_condition,
    {
        # 조건 출력을 그래프 노드에 매핑
        "tools": "retrieve",
        END: END,
    },
)

# 액션 노드 실행 후 처리될 엣지 정의
workflow.add_conditional_edges(
    "retrieve",
    # 문서 품질 평가
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# 그래프 컴파일
graph = workflow.compile(checkpointer=MemorySaver())

# obj = graph.get_graph().draw_mermaid_png()
# with open("test.png", "wb") as png:
#     png.write(obj)


inputs = {
    "messages": [
        ("user", "인공지능 산업 동향에 대해 설명하세요."),
    ]
}

config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})
ret = graph.invoke(inputs, config)
print(ret)