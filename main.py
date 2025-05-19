## functional dependencies
import time
## settings up the env
import os
from dotenv import load_dotenv
load_dotenv()

## langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir_path, "data")
persistent_directory = os.path.join(current_dir_path, "data-ingestion-local") ## creating a directory to save the vector store locally

if not os.path.exists(persistent_directory):
    print("[INFO] Initiating the build of Vector Database .. 📌📌", end="\n\n")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
        )

    pdfs = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.pdf')]

    doc_container = []

    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        for doc in docs:
            doc_container.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs_split = text_splitter.split_documents(doc_container)

    ## displaying information about the split documents
    print("\n--- Document Chunks Information ---", end="\n")
    print(f"Number of document chunks: {len(docs_split)}", end="\n\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    print("[INFO] Started embedding", end="\n")
    start_time = time.time()
    vectorDB = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persistent_directory
    )

    end_time = time.time()
    print("[INFO] Finished embedding", end="\n")
    print(f"[ADD. INFO] Time taken: {end_time - start_time}")

else:
    print("[ALERT] Vector Database already exist. ️⚠️")




