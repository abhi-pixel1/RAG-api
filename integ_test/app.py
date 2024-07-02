import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader,PyPDFLoader,UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# import chromadb
from langchain.memory import VectorStoreRetrieverMemory
# from langchain.chains.question_answering import load_qa_chain
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import tempfile
from pydantic import BaseModel



# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables import RunnablePassthrough


load_dotenv()  # Load environment variables from .env file


app = FastAPI()


class StringConversionRequest(BaseModel):
    text: str


print("API server starting")

################################################################################################
# This will be run on server startup
################################################################################################

# mongo atlas DB connection
uri = os.getenv("MONGO_URI")

client = MongoClient(uri, server_api=ServerApi('1'))

DB_NAME = 'RAG_Store'
db = client[DB_NAME]

# Collections for knowledge base and chat history
knowledge_base = db['knowledge_base']
# chat_history = db['chat_history']

# Search indexes
ATLAS_VECTOR_SEARCH_INDEX_NAME_KB = "vector_index_kb"
ATLAS_VECTOR_SEARCH_INDEX_NAME_CH = "vector_index_ch"

################################################################################################

# Getting the model from huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256, #256
    temperature=0.2, #0.2
    top_k=10,
    huggingfacehub_api_token=huggingfacehub_api_token
)

###############################################################################################

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

# Setting up retrivers and memory for conversation history
knowledge_base_retriever = knowledge_base_lang.as_retriever(search_kwargs={"k": 1})
chat_history_retriever = chat_history_lang.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=chat_history_retriever, memory_key="chat_history", input_key="human_input")



chain = ( prompt | llm )

################################################################################################
print("API server ready")
################################################################################################

# methods to handle all the request

# '/uploadfile/' endpoint handles file uploads. Currently supports only .txt files.
# File received in POST request is saved to disk, then reloaded for chunking and adding to DB.
# Aim: Chunk the file directly without saving it to disk.
# Issue: Any non-.txt file received will fail during the chunking process.

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
     # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

    # Load the text from the saved file and perfornm chunking
        loader = TextLoader(temp_file_path)
        doc = loader.load()
        docs = text_splitter.split_documents(doc)

        # context_db = Chroma.from_documents(docs, embeddings, persist_directory="./context_db")
        # context_retriever = context_db.as_retriever()

        # Saving the chunk embeddings to mongo atlas.
        vector_search = MongoDBAtlasVectorSearch.from_documents(
                                documents=docs, 
                                embedding= embeddings, 
                                collection= knowledge_base,
                                index_name= ATLAS_VECTOR_SEARCH_INDEX_NAME_KB
                                                    )

        # Returning number of docs stored to DB
        return {temp_file_path : len(docs)}
        #return {"message": "Text file uploaded and vector store created successfully."}
    except Exception as e:
        return {"error": str(e)}

######################################################################################################################    

#'/upload' endpoint handles file uploads for PDF files
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

        return {temp_file_path : len(texts)}
    except Exception as e:
        return {"error": str(e)}


################################################################################################
@app.post("/uploadppt")
async def upload_ppt(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Load PPT
        loader = UnstructuredPowerPointLoader(temp_file_path)
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

        return {temp_file_path : len(texts)}
    except Exception as e:
        return {"error": str(e)}
# '/query/' endpoint handles search queries.
# It performs similarity search on the knowledge base using the provided human input.

################################################################################################

@app.post("/query/")
def query(human_input: StringConversionRequest):

    # Execute a processing chain on the search results and input query

    # Perform similarity search on the knowledge base using the provided human input.
    page_contents_kb = [doc.page_content for doc in knowledge_base_retriever.invoke(human_input.text)]
    print(page_contents_kb)
    page_contents_ch = [doc.page_content for doc in chat_history_retriever.invoke(human_input.text)]
    print(page_contents_ch)

    # pass the context and history added query to the llm model to get the result
    c = chain.invoke({"context": page_contents_kb, "chat_history": page_contents_ch, "human_input": human_input.text})

    # print(prompt.format(context=page_contents_kb, chat_history=page_contents_ch, human_input=human_input))
    # print("-----------------------------------------------------------------")
    # print(c)

    # Save the current human input and chatbot response to memory
    memory.save_context({"Human": human_input.text}, {"Chatbot": c})

    # Return the llm response
    return c

################################################################################################

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)