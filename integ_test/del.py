import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection URI
uri = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(uri, server_api=ServerApi('1'))

# Database and collection names
DB_NAME = 'RAG_Store'
COLLECTION_NAME = 'knowledge_base'

# Access the knowledge_base collection
db = client[DB_NAME]
knowledge_base = db[COLLECTION_NAME]

# Delete all documents in the knowledge_base collection
result = knowledge_base.delete_many({})

# Print the number of documents deleted
print(f"Successfully deleted {result.deleted_count} documents from the knowledge_base collection.")
