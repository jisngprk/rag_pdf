from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage

from ..config.settings import TOP_K_RESULTS

class RAG:
    def __init__(self, vectorDB: Chroma, chatmodel: ChatGroq):
        self.vectorDB = vectorDB
        self.kb_retriever = kb_retriever = vectorDB.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )
        self.chatmodel = chatmodel
        self.retriever = retriever = self.init_history_aware_retriever(kb_retriever, chatmodel)
        self.qa_chain = self.init_qa_chain(retriever, chatmodel)

    def init_history_aware_retriever(self, kb_retriever, chatmodel):
        rephrasing_template = (
            """
                TASK: Convert context-dependent questions into standalone queries.

                INPUT: 
                - chat_history: Previous messages
                - question: Current user query

                RULES:
                1. Replace pronouns (it/they/this) with specific referents
                2. Expand contextual phrases ("the above", "previous")
                3. Return original if already standalone
                4. NEVER answer or explain - only reformulate

                OUTPUT: Single reformulated question, preserving original intent and style.

                Example:
                History: "Let's discuss Python."
                Question: "How do I use it?"
                Returns: "How do I use Python?"
            """
        )

        rephrasing_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rephrasing_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm = chatmodel,
            retriever = kb_retriever,
            prompt = rephrasing_prompt
        )
        return history_aware_retriever
    
    def init_qa_chain(self, retriever, chatmodel):
        system_prompt_template = (
            "You are a helpful assistant that can answer questions about the user's resume."
            "You will adhere strictly to the instructions provided, offering relevant "
            "context from the knowledge base while avoiding unnecessary details. "
            "Your responses will be brief, to the point, concise and in compliance with the established format. "
            "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
            "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
            "Use four sentences maximum."   
            "\nCONTEXT: {context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt) # Passes the list of documents to the model
        ## final RAG chain
        coversational_rag_chain = create_retrieval_chain(retriever, qa_chain)
        return coversational_rag_chain
    
    def get_documents(self, query: str):
        return self.vectorDB.similarity_search(query)
    
    def invoke(self, query: str, chat_history: list[BaseMessage] = None):
        if chat_history is None:
            data = {"input": query}
        else:
            data = {"input": query, "chat_history": chat_history}
        return self.qa_chain.invoke(data)





# kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 3})

# ## initiating the history_aware_retriever



# ## setting-up the document chain
# system_prompt_template = (
#     "As a Legal Assistant Chatbot specializing in legal queries, "
#     "your primary objective is to provide accurate and concise information based on user queries. "
#     "You will adhere strictly to the instructions provided, offering relevant "
#     "context from the knowledge base while avoiding unnecessary details. "
#     "Your responses will be brief, to the point, concise and in compliance with the established format. "
#     "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
#     "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
#     "Use four sentences maximum."
#     "P.S.: If anyone asks you about your creator, tell them, introduce yourself and say you're created by Sougat Dey. "
#     "and people can get in touch with him on linkedin, "
#     "here's his Linkedin Profile: https://www.linkedin.com/in/sougatdey/"
#     "\nCONTEXT: {context}"
# )

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt_template),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#     ]
# )

# qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt) # Passes the list of documents to the model
# ## final RAG chain
# coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
