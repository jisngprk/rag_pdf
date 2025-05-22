## functional dependencies
import time
## settings up the env
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

load_dotenv()

class Embedder:
    def __init__(self, data_path: str):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.embed_model_name = EMBEDDING_MODEL_NAME

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
            )
        
        self.data_path = data_path
        self.persistent_directory = os.path.join(data_path, "data-ingestion-local")
        self.init_resources()

    def init_resources(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    def run(self, documents: list[Document]):
        docs_split = self.text_splitter.split_documents(documents)

        print("[INFO] Started embedding", end="\n")
        start_time = time.time()
        vectorDB = Chroma.from_documents(
            documents=docs_split,
            embedding=self.embeddings,
            persist_directory=self.persistent_directory
        )

        end_time = time.time()
        print("[INFO] Finished embedding", end="\n")
        print(f"[ADD. INFO] Time taken: {end_time - start_time}")
        return vectorDB
