from enum import Enum
from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class ElasticCloudConfig(DBConfig, BaseModel):
    cloud_id: SecretStr
    password: SecretStr
    number_of_shards: int | None = 1

    def to_dict(self) -> dict:
        return {
            "cloud_id": self.cloud_id.get_secret_value(),
            "basic_auth": ("elastic", self.password.get_secret_value()),
            "number_of_shards": self.number_of_shards,
        }


class ESElementType(str, Enum):
    float = "float"  # 4 byte
    byte = "byte"  # 1 byte, -128 to 127


class ElasticCloudIndexConfig(BaseModel, DBCaseConfig):
    element_type: ESElementType = ESElementType.float
    index: IndexType = IndexType.HNSW

    metric_type: MetricType | None = None
    efConstruction: int | None = None
    M: int | None = None
    num_candidates: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "cosine"

    def index_param(self) -> dict:
        params = {
            "type": "dense_vector",
            "index": True,
            "element_type": self.element_type.value,
            "similarity": self.parse_metric(),
            "index_options": {
                "type": "hnsw",
                "m": self.M,
                "ef_construction": self.efConstruction,
            },
        }
        return params

    def search_param(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
        }
 

class HNSWConfig(ElasticCloudIndexConfig):
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        sup = super().index_param()
        sup["index_options"] = sup["index_options"] | {
            "type":"hnsw"
        }
        return sup


class INT8HNSWConfig(ElasticCloudIndexConfig):
    index: IndexType = IndexType.HNSW_INT8

    def index_param(self) -> dict:
        sup = super().index_param()
        sup["index_options"] = sup["index_options"] | {
            "type":"int8_hnsw"
        }
        return sup

_elasticcloud_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.HNSW_INT8: INT8HNSWConfig,
}
