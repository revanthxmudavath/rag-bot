from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://revanthnaik12_db_user:mudavatr@cluster-rag-bot-backend.okflkbo.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
print(client.admin.command("ping"))
