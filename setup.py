from setuptools import setup, find_packages

setup(
    name="rag_pdf",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-groq",
        "langchain-huggingface",
        "langchain-chroma",
        "python-dotenv",
    ],
) 