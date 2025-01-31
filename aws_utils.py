import boto3
import urllib3

from typing import Any
from io import BytesIO
from config import get_settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


settings = get_settings()


class AwsService:
    __slots__ = ["s3_client", "bucket"]

    def __init__(self):
        ## Rodar local com o Localstack -> https://docs.localstack.cloud/user-guide/aws/s3/
        # self.s3_client = boto3.client("s3", endpoint_url = "http://localhost:4566", aws_access_key_id="test", aws_secret_access_key="test")

        self.s3_client = boto3.client(
            "s3", region_name=settings.aws_region, verify=False
        )

    def upload_file(self, file: Any, key: str):
        """
        Faz upload de um arquivo local para o S3.
        """
        try:
            file_content = file.file.read()
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=BytesIO(file_content),
                ContentType=file.content_type,
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao fazer upload do arquivo: {e}")

    def get_file(self, key: str, file_path: str):
        """
        Faz download de um arquivo do S3 para o caminho local.
        """
        try:
            self.s3_client.download_file(self.bucket, key, file_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao fazer download do arquivo: {e}")
