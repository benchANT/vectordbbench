import environs
import inspect
import pathlib
from . import log_util


env = environs.Env()
env.read_env(".env")

class config:
    ALIYUN_OSS_URL = "assets.zilliz.com.cn/benchmark/"
    AWS_S3_URL = "assets.zilliz.com/benchmark/"

    LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

    DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", AWS_S3_URL)
    DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
    NUM_PER_BATCH = env.int("NUM_PER_BATCH", 5000)

    DROP_OLD = env.bool("DROP_OLD", True)
    USE_SHUFFLED_DATA = env.bool("USE_SHUFFLED_DATA", True)
    NUM_CONCURRENCY = [1, 5, 10, 15, 20, 25, 30, 35]

    RESULTS_LOCAL_DIR = pathlib.Path(__file__).parent.joinpath("results")

    def _init_timeout(value, default) -> int | None:
        if value == None:
            return default
        if value == -1:
            return None
        else:
            return value

    CAPACITY_TIMEOUT_IN_SECONDS = 24 * 3600 # 24h
    LOAD_TIMEOUT = env.int("LOAD_TIMEOUT", None)
    LOAD_TIMEOUT_DEFAULT    = _init_timeout(value=LOAD_TIMEOUT, default=(2.5 * 3600)) # 2.5h
    LOAD_TIMEOUT_768D_1M    = _init_timeout(value=LOAD_TIMEOUT, default=(2.5 * 3600)) # 2.5h
    LOAD_TIMEOUT_768D_10M   = _init_timeout(value=LOAD_TIMEOUT, default=(25 * 3600)) # 25h
    LOAD_TIMEOUT_768D_100M  = _init_timeout(value=LOAD_TIMEOUT, default=(250 * 3600)) # 10.41d
    LOAD_TIMEOUT_1536D_500K = _init_timeout(value=LOAD_TIMEOUT, default=(2.5 * 3600)) # 2.5h
    LOAD_TIMEOUT_1536D_5M   = _init_timeout(value=LOAD_TIMEOUT, default=(25 * 3600)) # 25h

    OPTIMIZE_TIMEOUT = env.int("OPTIMIZE_TIMEOUT", None)
    OPTIMIZE_TIMEOUT_DEFAULT    = _init_timeout(value=OPTIMIZE_TIMEOUT, default=(15 * 60)) # 15min
    OPTIMIZE_TIMEOUT_768D_1M    = _init_timeout(value=OPTIMIZE_TIMEOUT, default=(15 * 60))# 15min
    OPTIMIZE_TIMEOUT_768D_10M   = _init_timeout(value=OPTIMIZE_TIMEOUT, default=(2.5 * 3600)) # 2.5h
    OPTIMIZE_TIMEOUT_768D_100M  = _init_timeout(value=OPTIMIZE_TIMEOUT, default=(25 * 3600)) # 1.04d
    OPTIMIZE_TIMEOUT_1536D_500K =  _init_timeout(value=OPTIMIZE_TIMEOUT, default=(15 * 60)) # 15min
    OPTIMIZE_TIMEOUT_1536D_5M   =  _init_timeout(value=OPTIMIZE_TIMEOUT, default=(2.5 * 3600)) # 2.5h

    def display(self) -> str:
        tmp = [
            i for i in inspect.getmembers(self)
            if not inspect.ismethod(i[1])
            and not i[0].startswith('_')
            and "TIMEOUT" not in i[0]
        ]
        return tmp

log_util.init(config.LOG_LEVEL)
