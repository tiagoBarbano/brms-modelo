from pymongo import MongoClient
from contextlib import contextmanager

from config import get_settings


settings = get_settings()

_mongo_client = None


def create_session():
    """Retorna uma instância única do MongoClient."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.mongo_host)

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
        
