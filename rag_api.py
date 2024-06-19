import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.memory import VectorStoreRetrieverMemory
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import tempfile

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

print("API server starting")

################################################################################################
# This will be run on server startup
################################################################################################

# mongo atlas DB connection
uri =  os.getenv("MONGO_URI")

client = MongoClient(uri, server_api=ServerApi('1'))

DB_NAME = 'RAG_Store'
db = client[DB_NAME]

# Collections for knowledge base and chat history
knowledge_base = db['knowledge_base']
chat_history = db['chat_history']

# Search indexes
ATLAS_VECTOR_SEARCH_INDEX_NAME_KB = "vector_index_kb"
ATLAS_VECTOR_SEARCH_INDEX_NAME_CH = "vector_index_ch"

################################################################################################

# Getting the model from huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.2,
    top_k=10,
    huggingfacehub_api_token=huggingfacehub_api_token
)

################################################################################################

# Setting up text splitter, embeddings, and prompt template
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
embeddings = HuggingFaceEmbeddings()

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate.from_template(template)

################################################################################################

# Setting up vector search for chat history and knowledge base
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

# Setting up retrievers and memory for conversation history
knowledge_base_retriever = knowledge_base_lang.as_retriever(search_kwargs={"k": 1})
chat_history_retriever = chat_history_lang.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=chat_history_retriever, memory_key="chat_history", input_key="human_input")

chain = (prompt | llm)

################################################################################################
print("API server ready")
################################################################################################

# '/uploadfile/' endpoint handles file uploads for .txt files
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    file_path = file.filename

    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load the text from the saved file and perform chunking
    loader = TextLoader(file_path)
    doc = loader.load()
    docs = text_splitter.split_documents(doc)

    # Saving the chunk embeddings to MongoDB Atlas
    MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=knowledge_base,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
    )

    # Returning the number of documents stored in the database
    return len(docs)

# '/upload' endpoint handles file uploads for PDF files
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split documents into smaller chunks
        texts = text_splitter.split_documents(documents)

        # Add texts to vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=texts,
            embedding=embeddings,
            collection=knowledge_base,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
        )

        # Remove temporary file
        os.remove(temp_file_path)

        return {"message": "PDF uploaded and vector store created successfully."}
    except Exception as e:
        return {"error": str(e)}

################################################################################################

# '/query/' endpoint handles search queries
@app.post("/query/")
def query(human_input: str):
    try:
        # Perform similarity search on the knowledge base using the provided human input
        page_contents_kb = [doc.page_content for doc in knowledge_base_retriever.invoke(human_input)]
        page_contents_ch = [doc.page_content for doc in chat_history_retriever.invoke(human_input)]

        # Pass the context and history added query to the LLM model to get the result
        result = chain.invoke({"context": page_contents_kb, "chat_history": page_contents_ch, "human_input": human_input})

        # Save the current human input and chatbot response to memory
        memory.save_context({"Human": human_input}, {"Chatbot": result})

        # Return the LLM response
        return result
    except Exception as e:
        return {"error": str(e)}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)