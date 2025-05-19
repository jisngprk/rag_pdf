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

