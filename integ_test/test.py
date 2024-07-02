import os
import pytest
from fastapi.testclient import TestClient
from fastapi import status
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from app import app
from dotenv import load_dotenv


client = TestClient(app=app)

load_dotenv()


"""
    Pytest fixture to set up and tear down the MongoDB connection.

    This fixture:
    1. Creates a connection to the MongoDB database before running the tests
    2. Yields the knowledge_base_collection for use in the tests
    3. Closes the MongoDB connection after all tests are completed

    The 'scope="module"' parameter ensures that this fixture is created once
    for the entire test module, rather than for each individual test function.
    This helps to reduce overhead by reusing the same database connection
    across multiple tests.

    Yields:
        pymongo.collection.Collection: The knowledge_base collection from the database.
"""
@pytest.fixture(scope="module")
def knowledge_base_collection():
    uri = os.getenv("MONGO_URI")
    mongo_client = MongoClient(uri, server_api=ServerApi('1'))
    db = mongo_client["RAG_Store"]
    knowledge_base_collection = db["knowledge_base"]
    yield knowledge_base_collection
    mongo_client.close()


# Test case to verify the upload of a text file
# It checks if the file is successfully uploaded and the correct number of documents are added to the database
def test_upload_txt_file(knowledge_base_collection):    
    with open("spider.txt", "rb") as file:
        response = client.post("/uploadfile/", files={"file": file})
    
    assert response.status_code == 200
    response_data = response.json()
    file_path = {'source':list(response_data.keys())[0]}
    assert knowledge_base_collection.count_documents(file_path) == list(response_data.values())[0]


# Test case to verify the upload of a PDF file
# It checks if the file is successfully uploaded and the correct number of documents are added to the database
def test_upload_pdf_file(knowledge_base_collection):
    with open("atten.pdf", "rb") as file:
        response = client.post("/upload", files={"file": file})
    
    assert response.status_code == 200
    response_data = response.json()
    file_path = {'source':list(response_data.keys())[0]}
    assert knowledge_base_collection.count_documents(file_path) == list(response_data.values())[0]


# Test case to verify the upload of a PowerPoint file
# It checks if the file is successfully uploaded and the correct number of documents are added to the database
def test_upload_ppt_file(knowledge_base_collection):
    with open("cap.pptx", "rb") as file:
        response = client.post("/uploadppt", files={"file": file})
    
    assert response.status_code == 200
    response_data = response.json()
    file_path = {'source':list(response_data.keys())[0]}
    assert knowledge_base_collection.count_documents(file_path) == list(response_data.values())[0]


# Test case to verify the query endpoint
# It checks if the query endpoint responds successfully to a sample question
def test_query():
    response = client.post("/query", json={"text": "What is the content of the document?"})
    assert response.status_code == 200