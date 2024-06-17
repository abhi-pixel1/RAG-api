
# RAG-Based Chatbot API

## Overview
This repository contains the code for a RAG-based (Retrieval-Augmented Generation) chatbot API. The chatbot uses MongoDB Atlas for storing and retrieving document embeddings, and integrates with Hugging Face's Mixtral-8x7B-Instruct-v0.1 model for generating responses. The API handles file uploads, performs text chunking, and processes user queries to provide contextually relevant answers.

## Features
* Handles file uploads, chunking, and storing document embeddings.
* Performs similarity search on knowledge base and chat history.
* Integrates with Hugging Face model for generating responses.
* Utilizes MongoDB Atlas for vector search and storage.

## Configuration
* MongoDB Atlas: Ensure you have a MongoDB Atlas account and a database named RAG_Store with two collections: knowledge_base and chat_history. These collections store the document embeddings and chat history, respectively.

* Hugging Face Model: The model used is mistralai/Mixtral-8x7B-Instruct-v0.1. The model configurations include max_new_tokens=256, temperature=0.2, and top_k=10.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`MONGO_URI`

`HUGGINGFACEHUB_API_TOKEN`

## API Endpoints
### Upload File
**`POST /uploadfile/`**

* **Description**: Uploads a file, chunks it, and stores the document embeddings in MongoDB Atlas.
* Parameters:
    * `file`: The file to be uploaded (currently supports only .txt files).
* Returns: The number of chunks stored in the database.

### Query
**`POST /query/`**

* **Description**: Handles search queries, performs similarity search on the knowledge base and chat history, and generates a response.
* Parameters:
    * `human_input`: The query input from the user.
* Returns: The generated response from the model.

## Notes
* IP Address and Port: Ensure the IP address and port are set correctly before deploying.
* Database Structure: The current setup uses a single collection for chat history and knowledge base. Future improvements may include a separate collection for each session or user.
* Model and Embeddings: Consider using a better embedding model and text splitter for improved performance.
* File Handling: The /uploadfile/ endpoint currently saves the file to disk before processing. Aim to process the file directly from memory to improve efficiency.

## Future Improvements
* Enhance file handling to support more file types and process files directly from memory.
* Optimize embeddings and text splitting methods for better performance.
* Improve database structure to handle multiple users and sessions more effectively.
