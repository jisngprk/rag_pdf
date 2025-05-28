import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embedder import Embedder
from langchain_groq import ChatGroq
from src.core.rag import RAG
from src.core.loader import PdfLoader

import streamlit as st

data_path = "./data"

# 전역 리소스 초기화 (파일 경로와 무관하게 재사용)
@st.cache_resource
def init_resources():
    loader = PdfLoader(clean_model_name="llama-3.1-8b-instant")
    embedder = Embedder(data_path=data_path)
    chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)
    return {
        'loader': loader, 
        'embedder': embedder, 
        'chatmodel': chatmodel
    }

# 파일별 파이프라인 생성 (파일이 변경될 때만 재생성)
@st.cache_resource
def create_pipeline(file_path: str):
    if file_path is None:
        return None
    
    resources = init_resources()
    documents = resources["loader"].run(file_path)
    vectorDB = resources["embedder"].run(documents)
    return RAG(vectorDB, resources["chatmodel"])

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "file_info" not in st.session_state:
        st.session_state["file_info"] = None
    
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = None

def draw_sidebar():
    with st.sidebar:
        st.title("Resume Chat")
        st.write("This is a chatbot that can help you chat with your resume.")
        st.write("You can ask questions about your resume, and the chatbot will answer you.")
        file_info = st.file_uploader("Upload your resume", type="pdf")
        if file_info:
            bytes_data = file_info.read()
            st.write("filename:", file_info.name)
            file_path = os.path.join(data_path, file_info.name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            st.write("File uploaded successfully")
            
            # 파일이 변경되면 파이프라인 업데이트
            if st.session_state["file_info"] != file_path:
                st.session_state["file_info"] = file_path
                st.session_state["pipeline"] = create_pipeline(file_path)
                # 파일이 변경되면 대화 기록 초기화
                st.session_state["messages"] = []

def main():
    # 세션 상태 초기화
    init_session_state()
    
    st.title("Resume Chat")
    draw_sidebar()
    
    # 채팅 히스토리 표시
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    user_query = st.chat_input("Ask me anything ..")
    if user_query:
        # 사용자 메시지 추가
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            if st.session_state["file_info"] is None:
                st.write("Please upload a resume first.")
            else:
                st.write("Thinking...")
                # 캐시된 파이프라인 사용
                pipeline = st.session_state["pipeline"]
                response = pipeline.invoke(user_query, chat_history=st.session_state["messages"])
                st.write(response['answer'])
                # 어시스턴트 메시지 추가
                st.session_state["messages"].append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()


