from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import chromadb
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains.question_answering import load_qa_chain


loader = TextLoader("horoscope.txt")
doc = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
docs = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()
context_db = Chroma.from_documents(docs, embeddings, persist_directory="./context_db")
context_retriever = context_db.as_retriever()

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=512, #256
    temperature=0.5, #0.2
    top_k=10,
    huggingfacehub_api_token="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"
)



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



persistent_client = chromadb.PersistentClient("./history_db")

history_db = Chroma(
    client=persistent_client,
    collection_name="chat_history",
    embedding_function=embeddings,
)
history_retriever = history_db.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=history_retriever, memory_key="chat_history", input_key="human_input")


chain = load_qa_chain(
    llm=llm, chain_type="stuff", memory=memory, prompt=prompt
)


query = "tell the fortune of Sagittarius"
docs = context_db.similarity_search(query)


c = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
print(c)