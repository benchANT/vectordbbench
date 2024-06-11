import argparse
import os

log_levels = [ "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL" ]
initParser = argparse.ArgumentParser(add_help=False)

initParser.add_argument('--skip-load-and-index', required=False, help="if set the load phase will not be run", action="store_true")
initParser.add_argument('--log-level', required=False, choices=list(log_levels), help="the debug level to use. INFO, if not set.")
initParser.add_argument('--disable-data-shuffle', help="if set data shuffling will be disabled", action="store_true")
initParser.add_argument('--batch-size', required=False, help="the batch size to use when loading the data set when --benchmark-type is set to performance", type=int)
initParser.add_argument('--load-timeout', required=False, help="the timeout to use during the load phase. this is not the database timeout, but the timout for the entire load phase. Set to -1 for 'no timeout'", type=int)
initParser.add_argument('--optimize-timeout', required=False, help="the timeout to use during the optimization phase. Set to -1 for 'no timeout'", type=int)

def handleEnvVariables(args):
    if args.log_level != None:
        os.environ["LOG_LEVEL"] = args.log_level
    if args.skip_load_and_index != None:
        os.environ["DROP_OLD"] = str(not args.skip_load_and_index)
    if args.disable_data_shuffle:
        os.environ["USE_SHUFFLED_DATA"] = str(not args.disable_data_shuffle)
    if args.batch_size:
        os.environ["NUM_PER_BATCH"] = str(args.batch_size)
    if args.load_timeout:
        os.environ["LOAD_TIMEOUT"] = str(args.load_timeout)
    if args.optimize_timeout:
        os.environ["OPTIMIZE_TIMEOUT"] = str(args.optimize_timeout)

initArgs = initParser.parse_known_args()[0]
print("converting parameters to env variables")
handleEnvVariables(initArgs)

# env = environs.Env()
# env.read_env(".env")

# class config:
#     ALIYUN_OSS_URL = "assets.zilliz.com.cn/benchmark/"
#     AWS_S3_URL = "assets.zilliz.com/benchmark/"

#     LOG_LEVEL = env.str("LOG_LEVEL", "INFO")

#     DEFAULT_DATASET_URL = env.str("DEFAULT_DATASET_URL", AWS_S3_URL)
#     DATASET_LOCAL_DIR = env.path("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
#     NUM_PER_BATCH = env.int("NUM_PER_BATCH", 5000)

#     DROP_OLD = env.bool("DROP_OLD", True)
#     USE_SHUFFLED_DATA = env.bool("USE_SHUFFLED_DATA", True)
#     NUM_CONCURRENCY = [1, 5, 10, 15, 20, 25, 30, 35]

#     RESULTS_LOCAL_DIR = pathlib.Path(__file__).parent.joinpath("results")

#     CAPACITY_TIMEOUT_IN_SECONDS = 24 * 3600 # 24h
#     LOAD_TIMEOUT_DEFAULT        = 2.5 * 3600 # 2.5h
#     LOAD_TIMEOUT_768D_1M        = 2.5 * 3600 # 2.5h
#     LOAD_TIMEOUT_768D_10M       =  25 * 3600 # 25h
#     LOAD_TIMEOUT_768D_100M      = 250 * 3600 # 10.41d

#     LOAD_TIMEOUT_1536D_500K     = 2.5 * 3600 # 2.5h
#     LOAD_TIMEOUT_1536D_5M       =  25 * 3600 # 25h

#     OPTIMIZE_TIMEOUT_DEFAULT    = 15 * 60   # 15min
#     OPTIMIZE_TIMEOUT_768D_1M    =  15 * 60   # 15min
#     OPTIMIZE_TIMEOUT_768D_10M   = 2.5 * 3600 # 2.5h
#     OPTIMIZE_TIMEOUT_768D_100M  =  25 * 3600 # 1.04d


#     OPTIMIZE_TIMEOUT_1536D_500K =  15 * 60   # 15min
#     OPTIMIZE_TIMEOUT_1536D_5M   =   2.5 * 3600 # 2.5h
#     def display(self) -> str:
#         tmp = [
#             i for i in inspect.getmembers(self)
#             if not inspect.ismethod(i[1])
#             and not i[0].startswith('_')
#             and "TIMEOUT" not in i[0]
#         ]
#         return tmp

# log_util.init(config.LOG_LEVEL)
