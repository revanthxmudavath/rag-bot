from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb-uri-here"
client = MongoClient(uri, server_api=ServerApi('1'))
print(client.admin.command("ping"))
