from database import get_database
from config import get_settings
from model import Model


settings = get_settings()


def salvar_modelo(data: Model):
    with get_database() as db:
        collection = db[settings.collection_model]
        return collection.insert_one(data.__dict__).inserted_id


def listar_modelos():
    with get_database() as db:
        collection = db[settings.collection_model]
        return collection.find().to_list()
