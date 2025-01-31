from pymongo import MongoClient
from contextlib import contextmanager

from config import get_settings


settings = get_settings()

_mongo_client = None


def get_connection_string():
    # if settings.mongo_host == "localhost:27017":
    return "mongodb://localhost:27017"

    # return f"mongodb+srv://{settings.mongo_user}:{settings.mongo_password}@{settings.mongo_host}{settings.mongo_read_preference}"


def create_session():
    """Retorna uma instância única do MongoClient."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(get_connection_string())

    return _mongo_client


@contextmanager
def get_database():
    """Retorna o banco de dados da aplicação usando um cliente singleton."""
    client = create_session()
    database = client[settings.database_mongodb]
    try:
        yield database
    except Exception as ex:
        client.close()
        raise ex
        
