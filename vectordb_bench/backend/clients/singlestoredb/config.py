from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

# SINGLESTOREDB_URL_PLACEHOLDER = "singlestoredb://%s:%s@%s/%s"
POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"

class SingleStoreDBConfig(DBConfig):
    user_name: SecretStr = "admin"
    password: SecretStr
    host: str
    port: int
    db_name: str
    partition_count: int | None = None
    columnstore_max_blobsize : int | None = None
    columnstore_segment_rows : int | None = None
    disable_compaction : bool | None = None

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        db_str = self.db_name
        host_str = self.host
        port_param = int(self.port)
        
        disable_compaction_dict = { 
         "disable_compaction": self.disable_compaction
        } if  self.disable_compaction != None else {}
        ret = {}
        if self.disable_compaction != None:
            ret["disable_compaction"] = self.disable_compaction
        if self.columnstore_segment_rows != None:
            ret["columnstore_segment_rows"] = self.columnstore_segment_rows
        if self.columnstore_max_blobsize != None:
            ret["columnstore_max_blobsize"] = self.columnstore_max_blobsize
        if self.partition_count != None:
            ret["partition_count"] = self.partition_count
        return ret | {
            "host": host_str,
            "port": port_param,
            "user": user_str,
            "password": pwd_str,
            "database": db_str
        } 


class SingleStoreDBIndexConfig(BaseModel, DBCaseConfig):
    """Base config for SingleStore"""
    index: IndexType
    metric_type: MetricType | None = None
    def effective_metric(self) -> MetricType:
        if self.metric_type == MetricType.COSINE:
            return MetricType.L2
        else:
            return self.metric_type

    def parse_metric(self) -> str:
        effective_metric = self.effective_metric()
        if effective_metric == MetricType.L2:
            return "EUCLIDEAN_DISTANCE"
        elif effective_metric == MetricType.IP:
            return "DOT_PRODUCT"
        # if metric is CONSINE, use EUCLIDEAN_DISTANCE
        # both are the same after normalization
        raise Exception(f"unsupported metric {effective_metric}")

    def index_param(self) -> dict:
        return {
            # "metric" : self.parse_metric()
            "metric_type" : self.parse_metric(),
        }

    def search_param(self) -> dict:
        return { }

class HNSWConfig(SingleStoreDBIndexConfig):
    index: IndexType = IndexType.HNSW
    M: int
    efConstruction: int
    ef: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "HNSW_FLAT",
            "M": self.M,
            "efConstruction": self.efConstruction
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "ef": self.ef,
        }

class IVFConfig(SingleStoreDBIndexConfig):
    nlist: int
    nprobe: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_FLAT",
            "nlist" : self.nlist,
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "nprobe" : self.nprobe,
        }

class IVFPQConfig(IVFConfig):
    index: IndexType = IndexType.IVFPQ
    quantizationRatio: int
    reorder_k: int

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_PQ",
            "m_dim_divisor": self.quantizationRatio,
        }

    def search_param(self) -> dict:
        return super().search_param() | {
            "k": self.reorder_k,
        }

class IVFPQFSConfig(IVFPQConfig):
    index: IndexType = IndexType.IVFPQFS

    def index_param(self) -> dict:
        return super().index_param() | {
            "index_type": "IVF_PQFS",
            # "with_raw_data" : False,
        }

    def search_param(self) -> dict:
        p = super().search_param()
        # del p["nprobe"]
        return p

# class HNSWPQConfig(HNSWConfig):
#     nbits: int | None = None
#
#     def index_param(self) -> dict:
#         return super().index_param() | {
#             "nbits": self.nbits,
#         }
#
#     def search_param(self) -> dict:
#         return super().search_param()

_singlestore_case_config = {
    IndexType.HNSW: HNSWConfig,
    IndexType.IVFFlat: IVFConfig,
    IndexType.IVFPQ: IVFPQConfig,
    IndexType.IVFPQFS: IVFPQFSConfig,

    IndexType.AUTOINDEX: SingleStoreDBIndexConfig,
    IndexType.DISKANN: SingleStoreDBIndexConfig,
    IndexType.Flat: SingleStoreDBIndexConfig,
}
