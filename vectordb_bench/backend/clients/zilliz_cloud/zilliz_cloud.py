"""Wrapper around the ZillizCloud vector database over VectorDB"""

from ..milvus.milvus import Milvus
from ..api import DBCaseConfig, IndexUse


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVectorDBBench",
        drop_old: bool = False,
        name: str = "ZillizCloud",
        index_use: IndexUse = IndexUse.BOTH_KEEP,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
            name=name,
            index_use=index_use,
            **kwargs,
        )
