from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

UPLOAD_BASES_PATH = "upload_bases"
DIRETORIO_MODELOS_PATH = "modelos_historicos"
MODELOS_CUSTOM_PATH = "modelos_custom"

class Settings(BaseSettings):
    aws_s3_bucket_name: str
    aws_region: str
    mongo_password: str
    mongo_user: str
    mongo_host: str
    mongo_read_preference: str
    database_mongodb: str
    collection_model: str
    collection_historico_model: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings():
    return Settings()
