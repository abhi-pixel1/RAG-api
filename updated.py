import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
import uvicorn

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Log startup
print("API server starting")

################################################################################################
# Server Startup Configuration
################################################################################################

# MongoDB Atlas connection
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
DB_NAME = 'RAG_Store'
db = client[DB_NAME]

# Collections for knowledge base and chat history
knowledge_base = db['knowledge_base']
chat_history = db['chat_history']

# Search indexes
ATLAS_VECTOR_SEARCH_INDEX_NAME_KB = "vector_index_kb"
ATLAS_VECTOR_SEARCH_INDEX_NAME_CH = "vector_index_ch"

# Hugging Face model setup
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
huggingfacehub_api_token = os.getenv("HUGGING_FACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.2,
    top_k=10,
    huggingfacehub_api_token=huggingfacehub_api_token
)

# Text splitter, embeddings, and prompt template setup
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
embeddings = HuggingFaceEmbeddings()

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate.from_template(template)

# Vector search setup for chat history and knowledge base
chat_history_lang = MongoDBAtlasVectorSearch.from_connection_string(
    uri,
    DB_NAME + "." + "chat_history",
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_CH,
)

knowledge_base_lang = MongoDBAtlasVectorSearch.from_connection_string(
    uri,
    DB_NAME + "." + "knowledge_base",
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB,
)

# Retriever and memory setup
knowledge_base_retriever = knowledge_base_lang.as_retriever(search_kwargs={"k": 1})
chat_history_retriever = chat_history_lang.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=chat_history_retriever, memory_key="chat_history", input_key="human_input")

chain = (prompt | llm)

# Log ready status
print("API server ready")

################################################################################################
# API Endpoints
################################################################################################

@app.post("/uploadfile/")
async def upload_text_file(file: UploadFile):
    """Handles the upload of text files."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        loader = TextLoader(temp_file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)

        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=knowledge_base,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
        )

        return {"message": f"Successfully processed and stored {len(docs)} document chunks."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles the upload of PDF files."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)

        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=knowledge_base,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
        )

        os.remove(temp_file_path)
        return {"message": "PDF uploaded and vector store created successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/uploadppt/")
async def upload_ppt(file: UploadFile = File(...)):
    """Handles the upload of PowerPoint files."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        loader = UnstructuredPowerPointLoader(temp_file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)

        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=knowledge_base,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
        )

        os.remove(temp_file_path)
        return {"message": "PPT uploaded and vector store created successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query/")
def query(human_input: str):
    """Handles search queries and returns relevant information."""
    try:
        page_contents_kb = [doc.page_content for doc in knowledge_base_retriever.invoke(human_input)]
        page_contents_ch = [doc.page_content for doc in chat_history_retriever.invoke(human_input)]

        response = chain.invoke({
            "context": page_contents_kb,
            "chat_history": page_contents_ch,
            "human_input": human_input
        })

        memory.save_context({"Human": human_input}, {"Chatbot": response})
        return response
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
