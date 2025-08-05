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
    def _init_(self):
        self.vector_db = None
        self.retriever = None
        self.current_doc = ""

    def upload_docs(self):
        self.current_doc = "documents/temp_document.pdf"
        loader = PyPDFLoader(file_path=self.current_doc)
        data = loader.load()

        print("LOADED")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 300)
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
            template="""You are an insurance policy analyst answering questions based only on the content of a health insurance policy document provided to you. Answer each question strictly based on the document and do not use prior knowledge or assumptions. Limit each answer to no more than 100 character.

            Question:
            {question}"""
        )

        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        print("retriever loaded")
        #RAG prompt
        template = """
        Answer the following question based only on the provided context. Do not use any external knowledge. Answer in no more than 100 characters.

        Context:
        {context}

        Question:
        {question}
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