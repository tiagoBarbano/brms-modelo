from database import get_database
from config import get_settings
from model import HistoricoModel


settings = get_settings()


def insert_historico_modelo(data: HistoricoModel):
    with get_database() as db:
        collection = db[settings.collection_historico_model]
        return collection.insert_one(data.__dict__).inserted_id
    
def find_all_historico_modelo():
    with get_database() as db:
        collection = db[settings.collection_historico_model]
        return collection.find().to_list()   