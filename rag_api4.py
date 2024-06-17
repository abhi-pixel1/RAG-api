from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# import chromadb
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains.question_answering import load_qa_chain
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.vectorstores import MongoDBAtlasVectorSearch




app = FastAPI()

print("API server starting")

################################################################################################
# This will be run on server startup
################################################################################################

# mongo atlas DB connection
uri = "mongodb+srv://abhinav:Password123@cluster0.jlclhef.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

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
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256, #256
    temperature=0.2, #0.2
    top_k=10,
    huggingfacehub_api_token="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"
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

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)

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

# Setting up memory for conversation history
history_retriever = chat_history_lang.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=history_retriever, memory_key="chat_history", input_key="human_input")

# Loading question answering chain with memory and prompt
chain = load_qa_chain(
    llm=llm, 
    chain_type="stuff", 
    memory=memory, 
    prompt=prompt
)

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
    file_path = file.filename

    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content) 

    # Load the text from the saved file and perfornm chunking
    loader = TextLoader(file_path)
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
    return len(docs)

################################################################################################

# '/query/' endpoint handles search queries.
# It performs similarity search on the knowledge base using the provided human input.
@app.post("/query/")
def query(human_input: str ):
    # Perform similarity search in the knowledge base using the input query
    docs = knowledge_base_lang.similarity_search(human_input)

    # Execute a processing chain on the search results and input query
    c = chain({"input_documents": docs, "human_input": human_input}, return_only_outputs=True)
    
    # print(docs)
    
    # Return the result of the processing chain
    return c















uvicorn.run(app, host='127.0.0.1', port=8000)