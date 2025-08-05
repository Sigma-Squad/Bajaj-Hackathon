from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever


class AI_model:
    def __init__(self):
        self.vector_db = None
        self.retriever = None
        self.current_doc = ""

    def upload_docs(self):
        self.current_doc = "documents/temp_document.pdf"
        loader = PyPDFLoader(file_path=self.current_doc)
        data = loader.load()

        print("LOADED")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 7500, chunk_overlap = 100)
        chunks = text_splitter.split_documents(data)

        print(f"Split into {len(chunks)} chunks.")

        #Add to vector database
        self.vector_db = Chroma.from_documents(
            documents = chunks,
            embedding = OllamaEmbeddings(model = "mxbai-embed-large", show_progress = True),
            collection_name = "local-rag"
            # persist_directory="./chroma_db" 
        )

        #self.retriever = self.vector_db.as_retriever()

        return f"uploaded"

    def run_model(self, query):
        #LLM from Ollama
        local_model = "llama3.2"
        llm = ChatOllama(model=local_model)

        print("llm loaded")
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant who only answers based on the given context. You are supposed to identify the validity of an insurance policy given these conditions, give answer in no more than 100 characters: 
            Original question: {question}"""
        )

        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        print("retriever loaded")
        #RAG prompt
        template = """
        You are an insurance assistant. Based only on the context below, answer the user's question in **exactly one short sentence**:

        Context:
        {context}

        Question: {question}
        """


        prompt = ChatPromptTemplate.from_template(template)
        print("prompt loaded")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("chain loaded")
        prompt = query #"46M, knee surgery, Pune, 3-month policy"

        back_prompt = "Output only one line, which starts with yes or no, which is the verdict. Explain your reasoning"
        result = chain.invoke(prompt + back_prompt)
        print("result:", result)
        return result