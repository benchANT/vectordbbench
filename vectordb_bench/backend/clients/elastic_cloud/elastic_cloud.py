import logging
import time
from contextlib import contextmanager
from typing import Iterable
from ..api import VectorDB, IndexUse
from .config import ElasticCloudIndexConfig
from elasticsearch import NotFoundError
from elasticsearch.helpers import bulk
from elasticsearch.helpers import parallel_bulk

for logger in ("elasticsearch", "elastic_transport"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

class ElasticCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ElasticCloudIndexConfig,
        indice: str = "vdb_bench_indice",  # must be lowercase
        id_col_name: str = "id",
        vector_col_name: str = "vector",
        drop_old: bool = False,
        index_use: IndexUse = IndexUse.BOTH_KEEP,
        **kwargs,
    ):
        if index_use != IndexUse.BOTH_KEEP:
            log.warning("index_use must be set to 'keep'")
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.index_use = index_use
        self.drop_old = drop_old
        self.indice = indice
        self.id_col_name = id_col_name
        self.vector_col_name = vector_col_name
        self.number_of_shards = db_config["number_of_shards"]
        del(db_config["number_of_shards"])

        from elasticsearch import Elasticsearch

        client = Elasticsearch(**self.db_config)

        if self.drop_old:
            log.info(f"Elasticsearch client drop_old check if index exists: {self.indice}")
            is_existed_res = client.indices.exists(index=self.indice)
            if is_existed_res.raw:
               log.info(f"Elasticsearch client drop_old indices: {self.indice}")
               client.indices.delete(index=self.indice)
            self._create_indice(client)

        del(client)

    @contextmanager
    def init(self) -> None:
        """connect to elasticsearch"""
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch(**self.db_config, request_timeout=180)

        yield
        # self.client.transport.close()
        self.client = None
        del(self.client)

    def _create_mappings(self) -> None:
        index_params = self.case_config.index_param()
        return {
            "_source": {"excludes": [self.vector_col_name]},
            "properties": {
                self.id_col_name: {"type": "integer", "store": True},
                self.vector_col_name: {
                    "dims": self.dim,
                    "type": index_params["type"],
                    "similarity": index_params["similarity"],
                    "element_type": index_params["element_type"],
                    "index": True,
                    "index_options": index_params["index_options"],
                },
            }
        }

    def _create_indice(self, client) -> None:
        settings = {
            "index": {
                "refresh_interval": -1,
#                "number_of_replicas": 0,
            }
        }
        if self.number_of_shards != None and self.number_of_shards > 1:
            settings = settings | {
                 "index": {
                     "number_of_shards": self.number_of_shards,
                 }
            }
        mappings = self._create_mappings()
        log.debug(f"creating index with mapping {mappings} and settings {settings}")
        try:
            client.indices.create(index=self.indice, mappings=mappings, settings=settings)
        except Exception as e:
            log.warning(f"Failed to create indice: {self.indice} error: {str(e)}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the elasticsearch."""
        assert self.client is not None, "should self.init() first"

        insert_data = [
            {
                "_index": self.indice,
                "_source": {
                    self.id_col_name: metadata[i],
                    self.vector_col_name: embeddings[i],
                },
            }
            for i in range(len(embeddings))
        ]
        try:
            for success, info in parallel_bulk(self.client, insert_data, thread_count=1):
                if not success:
                    raise Exception("A document failed to insert: %s" % info)
            return (len(embeddings), None)
        except Exception as e:
            log.warning(f"Failed to insert data: {self.indice} error: {str(e)}")
            return (0, e)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[tuple[int, float]]: list of k most similar embeddings in (id, score) tuple to the query embedding.
        """
        assert self.client is not None, "should self.init() first"
        # is_existed_res = self.client.indices.exists(index=self.indice)
        # assert is_existed_res.raw == True, "should self.init() first"

        knn = {
            "field": self.vector_col_name,
            "k": k,
            "num_candidates": self.case_config.num_candidates,
            "filter": [{"range": {self.id_col_name: {"gt": filters["id"]}}}]
            if filters
            else [],
            "query_vector": query,
        }

        try:
            res = self.client.search(
                index=self.indice,
                knn=knn,
                size=k,
                _source=False,
                docvalue_fields=[self.id_col_name],
                stored_fields="_none_",
                filter_path=[f"hits.hits.fields.{self.id_col_name}"],
            )
            res = [h["fields"][self.id_col_name][0] for h in res["hits"]["hits"]]

            return res
        except Exception as e:
            log.warning(f"Failed to search: {self.indice} error: {str(e)}")
            raise e from None

    def optimize(self):
        """optimize will be called between insertion and search in performance cases."""
        assert self.client is not None, "should self.init() first"
        try:
            log.debug("optimize: force merging to one segment")
            force_merge_task_id = self.client.indices.forcemerge(index=self.indice, max_num_segments=1, wait_for_completion=False, flush=True)['task']
            log.info(f"Elasticsearch force merge task id: {force_merge_task_id}")
            SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC = 30
            while True:
                time.sleep(SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC)
                task_status = self.client.tasks.get(task_id=force_merge_task_id)
                if task_status['completed']:
                    break
                log.debug("optimize: waiting for force merge to complete")
            log.debug("optimize: refreshing the index")
            self.client.indices.refresh(index=self.indice)
        except NotFoundError as e:
            log.debug("optimize: not able to find merge task id. Assuming completion")
        except Exception as e:
            log.warning(f"Failed to optimize: {self.indice} error: {str(e)}")
            raise e from None
        settings = {
            "index": {
                "number_of_replicas": 2,
            }
        }
#        log.debug("optimize: merge done. Setting replicas to 3")
#        ret = self.client.indices.put_settings(settings=settings, index=self.indice, timeout=-1)
#        while True:
#            time.sleep(SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC)
#            ret = self.client.indices.stats(index=self.indice, level='indices')
#            if ret["_shards"]["total"] == ret["_shards"]["successful"]:
#                break
#            log.debug("optimize: waiting for replica creation") 
        log.debug("optimize: done.")

    def ready_to_load(self):
        """ready_to_load will be called before load in load cases."""
        pass
