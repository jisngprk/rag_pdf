from src.core.embedder import Embedder
from langchain_groq import ChatGroq
from src.core.rag import RAG
from src.core.loader import PdfLoader


loader = PdfLoader(clean_model_name="llama-3.1-8b-instant")
documents = loader.run("./data/Resume_JisungPark.pdf")
print('-*'*100)
print(documents)
print('-*'*100)
embedder = Embedder(data_path="./data")
vectorDB = embedder.run(documents)
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

rag = RAG(vectorDB, chatmodel)
questions = ["Highlight the person's strength in the resume."]

for question in questions:
    ret = rag.invoke(question)
    for context in ret['context']:
        print('Context: ', context)
        print('-'*100)
    print('Answer: ', ret['answer'])
    print("-"*100)

