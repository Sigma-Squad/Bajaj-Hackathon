from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


from google import genai
import dotenv, os

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

    def run_model(self, query):
        # Use Gemini API for LLM
        # Retrieve context from vector DB
        retriever = self.vector_db.as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Prepare prompt
        prompt = f"""
        You are an insurance policy analyst answering questions based only on the content of a health insurance policy document provided to you. Answer each question strictly based on the document and do not use prior knowledge or assumptions. Limit each answer to one or two lines.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nOutput only one line, which starts with yes or no, which is the verdict. Explain your reasoning."""

        # Gemini API call
        client = genai.Client(
            api_key= os.getenv("GEMINI_API_KEY"),
        )

        """
        result = client.models.embed_content(
                model="gemini-embedding-001",
                contents= [

                ])

        print(result.embeddings)"""
        
        print("started query")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )

        return response.text