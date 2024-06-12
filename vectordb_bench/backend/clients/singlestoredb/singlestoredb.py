"""Wrapper around the SingleStore vector database over VectorDB"""

import logging
import tempfile
import json
import os
import traceback
from typing import Any, Type
import multiprocessing as mp
import threading
from contextlib import contextmanager
from typing import Any
from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType, IndexUse
from .config import SingleStoreDBConfig, SingleStoreDBIndexConfig, SingleStoreOptimizeStrategy

from ..api import VectorDB, DBCaseConfig, MetricType
import singlestoredb as s2
import numpy as np
log = logging.getLogger(__name__)

@contextmanager
def TemporaryPipe(mode="r"):
    """ Context manager for creating and automatically destroying a
        named pipe.

        It returns a file name rather than opening the pipe because
        the open() is expected to block until there's a
        reader. Instead it returns the name and expects you to launch
        a reader and *then* open it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'pipe')
        os.mkfifo(path)
        try:
            yield path
        finally:
            os.unlink(path)

def vector_to_hex(v):
    return np.array(v, np.float32).tobytes(order="C").hex()

# +# def LoadFromPipe(cursor, path):
# +#     cursor.execute("load data local infile \"%s\" into table points(id, @v) "
# +#                    "format csv "
# +#                    "fields terminated by ',' enclosed by '' escaped by '' "
# +#                    "lines terminated by '\n' "
# +#                    f"set embedding = unhex(@v):>vector({self.dim}, F32);"
# +#                    % (path))

# URL_PLACEHOLDER = "%s:%s@%s:%s/%s"
URL_PLACEHOLDER = "%s:%s@%s:%s"

class SingleStoreDB(VectorDB):

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "SingleStoreDBCollection",
        drop_old: bool = False,
        index_use: IndexUse = IndexUse.BOTH_KEEP,
        **kwargs,
    ):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB.__init__ called")
            log.debug("SingelStoreDB.__init__ the config: " + str(db_config))
        # some fixed parameters
        self._index_name = "singlestoredb_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        # flexible parameters
        self.db_config = db_config
        self.case_config = db_case_config
        self.db_name = db_config["database"]
        self.table_name = collection_name
        self.dim = dim
        self.drop_old = drop_old
        self.index_use = index_use
        self.partition_count = db_config.get("partition_count", 1)
        self.internal_columnstore_max_uncompressed_blob_size = db_config.get("columnstore_max_blobsize", None)
        self.columnstore_segment_rows = db_config.get("columnstore_segment_rows", None)
        self.disable_compaction = db_config.get("disable_compaction", False)
        self.optimize_strategy = db_config.get("optimize_strategy", None)
        # hard-coded for now
        self.vector_data_type = " F32 "
        self.table_options = " option 'SeekableString' "
        # cleanup
        del self.db_config["database"]
        if "partition_count" in self.db_config:
            del self.db_config["partition_count"]
        if "columnstore_max_blobsize" in self.db_config:
            del self.db_config["columnstore_max_blobsize"]
        if "columnstore_segment_rows" in self.db_config:
            del self.db_config["columnstore_segment_rows"]
        if "disable_compaction" in self.db_config:
            del self.db_config["disable_compaction"]
        if "optimize_strategy" in self.db_config:
            del self.db_config["optimize_strategy"]
        # placeholders for later
        self._cur = None
        self._ann_query = None
        self.warmed_query_debug = False
        
        # do some sanity checks
        if self.disable_compaction == True and self.internal_columnstore_max_uncompressed_blob_size == None:
            log.error("columnstore_max_blobsize must be set, when disable_compaction is True")
            raise Exception("columnstore_max_blobsize must be set, when disable_compaction is True")
        if self.partition_count != None and self.columnstore_segment_rows != None and (self.columnstore_segment_rows % self.partition_count) != 0:
            log.error("columnstore_segment_rows divided by partition_count should be an integer")

        # code starts here
        conn = self._connect()
        self._cur = conn.cursor()

        self._create_database()
        self._create_table()

        if (self.index_use == IndexUse.LOAD or
            self.index_use == IndexUse.BOTH_RESET or
            self.index_use == IndexUse.BOTH_KEEP):
           self._create_index()
        elif not self.drop_old:
            log.warning("should not use index for load, but drop old not set")
            log.warning("TODO: try to remove existing index")
        else:
            log.info("not using index for load")

        self._cur.execute("SHOW VARIABLES LIKE 'memsql_version'")
        log.info(f"using Memsql version: {self._cur.fetchall()}")
        self._cur = None
    
    def _create_table(self):
        with_attributes_string = ""
        compaction_string = (f"columnstore_flush_bytes={self.internal_columnstore_max_uncompressed_blob_size} ") if self.disable_compaction else ""
#                             f" columnstore_disk_insert_threshold=0.5") 


        segment_rows_string =(f"columnstore_segment_rows={self.columnstore_segment_rows}") if self.columnstore_segment_rows != None else ""
        if len(compaction_string) > 0 and len(segment_rows_string) > 0:
            with_attributes_string = f" with( {compaction_string}, {segment_rows_string} ) " 
        elif len(compaction_string) > 0:
            with_attributes_string = f" with( {compaction_string} )"
        elif len(segment_rows_string) > 0:
            with_attributes_string = segment_rows_string

        # finally, create table if needed, if it was not deleted, this will not have any impact
        create_table_command = (f"create table if not exists {self.table_name}("
                            f"{self._primary_field} int option 'Integer', "
                            f"{self._vector_field} vector({self.dim}, {self.vector_data_type}) NOT NULL "
                            f"{ self.table_options }, "
                            f"SORT KEY () {with_attributes_string})")
        log.info(f"creating table: {create_table_command}")
        self._cur.execute(create_table_command)
        if not self.drop_old:
            if self.partition_count != None:
                log.warning("skipping partition_count config due to drop_old set to false") 
            if self.columnstore_segment_rows != None:
                log.warning("skipping columnstore_segment_rows config due to drop_old set to false") 
            if self.disable_compaction == True:
                log.warning("compaction not necessarily disabled")
            log.warning("previous table was not dropped. the following properties have been set, but may not be in effect")
            log.warning(f"\t dim: {self.dim}")
            log.warning(f"\t table_options: {self.table_options}")
            log.warning(f"\t tmp_columnstore_segment_rows: {with_attributes_string}")

    def _create_database(self):
        if self.drop_old:
            log.info(f"deleting database {self.db_name}...")
            self._cur.execute(f"drop database if exists {self.db_name}")
        else:
            if self.partition_count != None:
                log.warning("drop_old not set and partition count set: may not have any impact if database already exists")
        
        paritions = f" partitions {self.partition_count}" if self.partition_count != None and self.partition_count > 1 else ""
        creation_command = f"create database if not exists {self.db_name} {paritions}" 
        log.info(f"creating database: {creation_command}")
        self._cur.execute(creation_command)
        self._cur.execute(f"use {self.db_name}")

        # modify global properties if needed
        # removed, as cannot be set on Helios -> fails
        #if self.disable_compaction:
        #    log.info(f"setting global columnstore_disk_insert_threshold 0.5")
        #    self._cur.execute("set global columnstore_disk_insert_threshold = 0.5")
        if self.internal_columnstore_max_uncompressed_blob_size != None:
            log.info(f"setting global internal_columnstore_max_uncompressed_blob_size {self.internal_columnstore_max_uncompressed_blob_size}")
            self._cur.execute(f"set global internal_columnstore_max_uncompressed_blob_size = {self.internal_columnstore_max_uncompressed_blob_size}")
        else:
            log.info("not setting internal_columnstore_max_uncompressed_blob_size")
        # self._cur.execute("set global sub_to_physical_partition_ratio = 0")
        # self._cur.execute("set global query_parallelism_per_leaf_core = 0")
        # self._cur.execute("set session query_parallelism_per_leaf_core = 0")

    def _connect(self):
        # connection_params = URL_PLACEHOLDER%(self.db_config['user'], self.db_config['password'], self.db_config['host'], str(self.db_config['port']), self.db_config['database'])
        connection_params = URL_PLACEHOLDER%(self.db_config['user'], self.db_config['password'], self.db_config['host'], str(self.db_config['port']))
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"SingleStoreDB._connect called: {connection_params}")
        return s2.connect(connection_params + "?local_infile=True")

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB.config_cls called")
        return SingleStoreDBConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB.case_config_cls called")
        return SingleStoreDBIndexConfig

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB.init called")
        conn = self._connect()
        self._cur = conn.cursor()
        self._cur.execute(f"use {self.db_name}")
        yield
        self._cur = None

    def _drop_index(self):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB._dropIndex called")
        command = (f"ALTER TABLE {self.table_name} DROP INDEX {self._index_name}")
        self._cur.execute(command)

    def _create_index(self):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingelStoreDB._createIndex called")
        if not self.drop_old:
            log.warning("trying to create index even though drop_old was not set")
        
        params = self.case_config.index_param()
        if "m_dim_divisor" in params:
            params["m"] = self.dim // params["m_dim_divisor"]
            del params["m_dim_divisor"]
        command_skeleton = (f"ALTER TABLE {self.table_name} "
                    f" ADD VECTOR INDEX {self._index_name} (embedding) index_options '%s'")
        command = command_skeleton % (json.dumps(params))
        log.info(f"create_index query: {command}")
        self._cur.execute(command)

    def ready_to_load(self):
        pass

    # This serves two purposes:
    # - Triggers query compilation and waits for it to complete.
    # - Loads the relevant index into memory.
    #
    def _warm_query(self, doExplain=False):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingleStoreDB._warm_query called")
        # assert(self.drop_old==False)
        self._cur.execute("set interpreter_mode=llvm")
        query = self.build_query(np.array([0] * self.dim,
                                          dtype=np.dtype('float32')), 10)
        if log.isEnabledFor(logging.DEBUG) and doExplain == True:
            log.debug(f"warm up query: {query}")
            self._cur.execute("explain " + query)
            log.debug(f"result of explained warm-up call: {self._cur.fetchall()}")
            self.warmed_query_debug = True
        self._cur.execute(query)

    def _optimize_add_on(self):
        query = None
        if self.optimize_strategy == SingleStoreOptimizeStrategy.OPTIMIZE:
            query = f"OPTIMIZE TABLE {self.table_name} WARM BLOB CACHE FOR INDEX {self._index_name};"
        elif self.optimize_strategy == SingleStoreOptimizeStrategy.SELECT_ALL:
            query = f"SELECT * FROM {self.table_name}"
        elif self.optimize_strategy == SingleStoreOptimizeStrategy.SELECT_SUM:
            query = f"SELECT SUM({self._primary_field}), VECTOR_SUM({self._vector_field}:>blob):>vector({self.dim}) FROM {self.table_name}"

        if query != None:
            log.debug(f"add-on optimize query: {query}.")
            self._cur.execute(query)
        else:
            log.debug("not running any add-on optimize query.")

    def optimize(self):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingleStoreDB.optimize called")
        # assert(self.drop_old==False)
        self._cur.execute("OPTIMIZE TABLE " + self.table_name + " flush")
        self._cur.execute("OPTIMIZE TABLE " + self.table_name + " full")
        if (self.index_use == IndexUse.LOAD or
            self.index_use == IndexUse.BOTH_RESET):
            self._drop_index()
        
        if (self.index_use == IndexUse.RUN or
            self.index_use == IndexUse.BOTH_RESET):
            self._create_index()
        elif self.index_use == IndexUse.BOTH_KEEP:
            log.warning("keeping index for run phase")
        elif not self.drop_old:
            log.warning("should not use index for run, but drop old not set, unknown state")
            log.warning("TODO: try to remove existing index")
        if self.optimize_strategy != None:
            self._optimize_add_on()
        self._warm_query(doExplain = True)

    def ready_to_search(self):
        if log.isEnabledFor(logging.DEBUG):
            log.debug("SingleStoreDB.ready_to_search called")
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        #if log.isEnabledFor(logging.DEBUG):
        #    log.debug("SingleStoreDB.insert_embedding called")
        try:
            def LoadFromPipe(path):
                self._cur.execute(f"LOAD DATA LOCAL INFILE \"%s\" INTO table {self.table_name}(id, @v) "
                                  "format csv "
                                  "fields terminated by ',' enclosed by '' escaped by '' "
                                  "lines terminated by '\n' "
                                  f"set embedding = unhex(@v):>vector({self.dim}, {self.vector_data_type});"
                                  % (path))

            # We're going to stream data over the socket via LOAD DATA
            # LOCAL INFILE "/path/to/named/pipe". The pipe lets us use
            # LOAD DATA rather than slower manual INSERTs, but also avoids
            # having to actually stage the data as a file on disk.
            with TemporaryPipe() as pipe_name:
                t = threading.Thread(target=LoadFromPipe, args=(pipe_name,))
                t.start()
                with open(pipe_name, "w") as f:
                    for i, embedding in zip(metadata, embeddings):
                        f.write("%d,%s\n" % (i, vector_to_hex(embedding)))
                t.join()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into table ({self.table_name}), error: {e}")
            return 0, e

    def need_normalize_cosine(self) -> bool:
        """Whether this database need to normalize dataset to support COSINE"""
        return True

    def build_query(self, v, k):
        if not self._ann_query:
            if log.isEnabledFor(logging.DEBUG):
                log.debug("SingleStoreDB.build_query")
            metric = self.case_config.effective_metric()
            assert metric in (MetricType.L2, MetricType.IP)
            param_str = json.dumps(self.case_config.search_param())
            if metric == MetricType.L2:
                self._ann_query = (f"select id FROM {self.table_name} order by embedding <-> X'%s' "
                                   f"search_options = '{param_str}' "
                                   "limit %d")
            elif metric != None:
                self._ann_query = (f"select id FROM {self.table_name} order by embedding <*> X'%s' "
                                   f"search_options = '{param_str}' "
                                   "desc limit %d")
            else:
                raise Exception("metric is None")
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"build_query query: {self._ann_query}")
            self._warm_query()
        return self._ann_query % (vector_to_hex(v), k)

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        #if log.isEnabledFor(logging.DEBUG):
        #    log.debug("SingleStoreDB.search_embedding")
        # assert(self.drop_old==False)
        assert filters is None
        # print (self.build_query(query, k))
        self._cur.execute(self.build_query(query, k))
        # +        # ret = self._cur.fetchall()
        # for id, partition_id in ret:
        #     print(partition_id)
        # return [id for id, parititon_id in ret]
        return [id for id, in self._cur.fetchall()]
