from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_s3_bucket_name: str
    aws_region: str
    mongo_password: str
    mongo_user: str
    mongo_host: str
    mongo_read_preference: str
    database_mongodb: str
    collection_model: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings():
    return Settings()
