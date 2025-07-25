from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class Preprocessor:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.chain = self.get_clean_sequential_chain()

    def get_recovery_prompt(self):
        template = """
        You are a helpful assistant that cleans text.
        You will be given a text and you will need to recover the original text.
        If the word looks like broken, you should recover it. 
        ONLY RECOVER THE TEXT. DO NOT SUMMARIZE THE TEXT.

        Please recover the original text:

        {text}

        Cleaned text:
        """

        clean_prompt = PromptTemplate(input_variables=["text"], template=template)
        return clean_prompt
    
    def get_formatting_prompt(self):
        template = """
        You are a helpful assistant that formats text.
        You will be given a text and you will need to format it.
        Follow the following format:
        Please read the following text and convert it into the following format:
        DO NOT SUMMARIZE THE TEXT. JUST FORMAT IT.
        ---
        Summary:
        [One or two sentences summarizing the person's background.]

        Work Experience:
        - [Job Title], [Company], [Location], [Dates]
        [Key achievements or responsibilities]
        - ...

        Skills:
        - [Skill 1]
        - [Skill 2]
        - ...

        Education:
        - [Degree], [Institution], [Dates]
        [Additional info if available]

        Publications:
        - "[Title]", [Journal or Conference], [Year]
        ---

        Input text:
        {cleaned_text}

        Output (formatted resume):

        """
        format_prompt = PromptTemplate(input_variables=["cleaned_text"], template=template)
        return format_prompt

    def get_recovery_chain(self):
        recovery_prompt = self.get_recovery_prompt()
        recovery_chain = LLMChain(llm=self.llm, prompt=recovery_prompt, output_key='cleaned_text')
        return recovery_chain

    def get_formatting_chain(self):
        formatting_prompt = self.get_formatting_prompt()
        formatting_chain = LLMChain(llm=self.llm, prompt=formatting_prompt, output_key='formatted_text')
        return formatting_chain

    def get_clean_reformat_sequential_cahin(self):
        recovery_chain = self.get_recovery_chain()
        formatting_chain = self.get_formatting_chain()
        sequential_chain = SequentialChain(chains=[recovery_chain, formatting_chain], input_variables=["text"], output_variables=["cleaned_text", "formatted_text"])
        return sequential_chain
    
    def get_clean_sequential_chain(self):
        recovery_chain = self.get_recovery_chain()
        sequential_chain = SequentialChain(chains=[recovery_chain], input_variables=["text"], output_variables=["cleaned_text"])
        return sequential_chain
    
    def run(self, text: str):
        return self.chain.invoke({"text": text})
    

class PdfLoader:
    def __init__(self, clean_model_name: str):
        self.clean_model_name = clean_model_name
        self.llm = llm = ChatGroq(model=clean_model_name)
        self.preprocessor = Preprocessor(llm=llm)

    def parse_pdf(self, pdf_path: str) -> list[Document]:
        doc_container = []

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc_container.append(doc)
        return doc_container
    
    def run(self, pdf_path: str, is_clean: bool = True) -> list[Document]:
        doc = self.parse_pdf(pdf_path)
        if not is_clean:
            print(doc[0].page_content)
            return doc
        
        ret = []
        for document in doc:
            updated_document = self.preprocessor.run({"text": document.page_content})
            document.page_content = updated_document['cleaned_text']
            ret.append(document)
        
        print("-"*100)
        print(ret[0].page_content)
        print("-"*100)
        return ret