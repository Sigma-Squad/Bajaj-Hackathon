from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from google import genai
import dotenv, os, json

dotenv.load_dotenv()

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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 240)
        chunks = text_splitter.split_documents(data)

        print(f"Split into {len(chunks)} chunks.")

        # Add to vector database using Gemini embeddings
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name="local-rag"
            # persist_directory="./chroma_db"
        )

        #self.retriever = self.vector_db.as_retriever()

        return f"uploaded"

    def run_model(self, queries):
        # Use Gemini API for LLM
        contexts = []
        for query in queries:
            # Retrieve context from vector DB
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query)
            contexts.append("\n".join([doc.page_content for doc in docs]))

        # Prepare prompt
        prompts = []
        front_prompt = """
You are an insurance policy analyst. Answer each question based only on the provided context. Do not use prior knowledge and do not use previous chats.
For each question, your output should be a single line.
The line may start with either 'Yes' or 'No' only if it is a conditional question, followed by a detailed explanation of your reasoning.
Process all the questions provided below and present the answers in the same order.
Use the delimiter ||| strictly to delimit each of your final responses.

questions to answer:
"""
        index = 0
        for context, query in zip(contexts, queries):
            index += 1
            prompt = f"""<query_{index}>\n<context>{context}</context>\n<question>{query}</question>\n</query_{index}>\n\n"""
            prompts.append(prompt)

        # Gemini API call
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        print("started query")
        combined_prompt = front_prompt + "\n\n".join(prompts)
        response = llm.invoke(combined_prompt)
        with open("documents/combined_prompt.txt", "w") as f:
            f.write(combined_prompt)
        
        print(response.content)
        return response.content.split("|||")  # Split the response by the delimiter to get individual answers