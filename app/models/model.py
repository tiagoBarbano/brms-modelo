import uuid

from dataclasses import field, dataclass
from datetime import datetime
from enum import Enum


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


@dataclass(kw_only=True)
class MongoBaseModel:
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))

    status: Status = Status.ACTIVE.value

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Model(MongoBaseModel):
    nome: str
    tipo: str
    arquivo_pickle: str
    metricas: str
    data_treinamento: str


@dataclass
class HistoricoModel(MongoBaseModel):
    data_treinamento: str
    modelo: str
    mae: str
    mse: str
    r2: str
    arquivo: str
    tempo_treinamento: str
