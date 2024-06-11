from . import initParser
import traceback
import logging
import subprocess
import os
import argparse

from vectordb_bench.backend.clients import (DB, DBConfig)
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.models import (
    DB, IndexType, CaseType, TaskConfig, CaseConfig, CaseConfigParamType, IndexUse
)

from vectordb_bench.frontend.const.dbCaseConfigs import (CaseConfigInput, CASE_CONFIG_MAP, InputType)

def indexChoices():
    choices = {}
    for i in IndexType:
        choices[i.name.lower()] = i
    return choices

def dbChoices():
    choices = {}
    for d in DB:
        choices[d.name.lower()] = d
    return choices

def indexUseChoices():
    choices = {}
    for i in IndexUse:
        choices[i.value.lower()] = i
    return choices

availableBenchTypes = {
        'performance': CaseLabel.Performance,
        'load': CaseLabel.Load,
}
availableDbs = dbChoices()
availableIndexes = indexChoices()
availableIndexUses = indexUseChoices()
availableDataSets = {
    'gist-100k': CaseType.CapacityDim960,
    'sift-500k': CaseType.CapacityDim128,
    'cohere-10M': CaseType.Performance768D10M,
    'cohere-1M': CaseType.Performance768D1M,
    'cohere-10M-1p': CaseType.Performance768D10M1P,
    'cohere-1M-1p': CaseType.Performance768D1M1P,
    'cohere-10M-99p': CaseType.Performance768D10M99P,
    'cohere-1M-99p': CaseType.Performance768D1M99P,
    'laion-100M': CaseType.Performance768D100M,
    'openai-500k': CaseType.Performance1536D500K,
    'openai-5M': CaseType.Performance1536D5M,
    'openai-500k-1p': CaseType.Performance1536D500K1P,
    'openai-5M-1p': CaseType.Performance1536D5M1P,
    'openai-500k-99p': CaseType.Performance1536D500K99P,
    'openai-5M-99p': CaseType.Performance1536D5M99P,
}

log_levels = [ "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL" ]

parser = argparse.ArgumentParser(parents=[initParser])

parser.add_argument('--label', required=False, help="the name of the benchmark series", type=str)
# parser.add_argument('--caseId', required=True, help="the name of the benchmark run", type=str)
parser.add_argument('--database', required=True, help="the backend to use", choices=list(availableDbs))
parser.add_argument('--index', required=False, help="the index to use", choices=list(availableIndexes))
parser.add_argument('--drop-old', required=False, help="if set the database will not be reset", action="store_true")
parser.add_argument('--index-use', required=False, choices=list(availableIndexUses), help="define when to use an index and when not")
parser.add_argument('--benchmark-type', required=True, choices=list(availableBenchTypes), help="the type of benchmark to run. currently restricted to configurations available for vanially VectorDBBench.")
parser.add_argument('--dataset', required=True, choices=list(availableDataSets), help="the dataset to use. These refer to the original VectorDBBench datasets.")
## moved to __init__.py
#parser.add_argument('--skip-load-and-index', required=False, help="if set the load phase will not be run", action="store_true")
#parser.add_argument('--log-level', required=False, choices=list(log_levels), help="the debug level to use. INFO, if not set.")
#parser.add_argument('--disable-data-shuffle', help="if set data shuffling will be disabled", action="store_true")
#parser.add_argument('--batch-size', required=False, help="the batch size to use when loading the data set when --benchmark-type is set to performance", type=int)

log = logging.getLogger("vectordb_bench benchant edition")

def addConfigOptionToSubgroup(group, db_prefix, case_prefix, config: CaseConfigInput):
    prefix: str = '--' + db_prefix + '.' + case_prefix + '.'
    required = False
    if config.inputType == InputType.Option:
        choices = config.inputConfig["options"]
        group.add_argument(prefix + config.label.name, required=required, choices=choices, help='parameter in ' + prefix + ' group for setting ' + config.label.name)
    elif config.inputType == InputType.Number:
        minMax = "min: " + str(config.inputConfig["min"]) + ", default: " + str(config.inputConfig["value"]) + ", max: " + str(config.inputConfig["max"])
        group.add_argument(prefix + config.label.name, type=int, help='parameter in ' + prefix + ' group for setting ' + config.label.name + ": " + minMax)
    else:
        raise Exception("unknown InputType: " + str(config.inputType))

    pass

def addConfigSubgroups():
    for d in DB:
        prefix = d.name.lower()
        properties = d.config_cls.schema().get("properties")
        group = parser.add_argument_group(prefix, 'parameters to chose when using ' + prefix + ' as --database')
        for p in properties:
            group.add_argument('--' + prefix + '.' + p, help='parameter in ' + prefix + ' group for setting ' + p, dest=p)
        dbconfOpts = CASE_CONFIG_MAP[d] if CASE_CONFIG_MAP.get(d) != None else []
        # for dbconfOpt in CASE_CONFIG_MAP[d]:
        for dbconfOpt in dbconfOpts:
            case = CASE_CONFIG_MAP[d][dbconfOpt]
            for el in case:
                x: CaseConfigInput = el
                if x.label != CaseConfigParamType.IndexType:
                    addConfigOptionToSubgroup(group=group, db_prefix=prefix, case_prefix=dbconfOpt.name.lower(), config=x)

def createDbConfig(db: DB, args) -> DBConfig:
    prefix = db.name.lower()
    properties = db.config_cls.schema().get("properties")
    dict = {}
    for p in properties:
        att = getattr(args, p)
        if att != None:
            dict[p] = att
    return db.config_cls(**dict)

## moved to __init__.py
# def handleEnvVariables(args):
#     if args.log_level != None:
#         os.environ["LOG_LEVEL"] = args.log_level
#     if args.skip_load_and_index != None:
#         os.environ["DROP_OLD"] = str(args.skip_load_and_index)
#     if args.disable_data_shuffle:
#         os.environ["USE_SHUFFLED_DATA"] = str(not args.disable_data_shuffle)
#     if args.batch_size:
#         os.environ["NUM_PER_BATCH"] = str(args.batch_size)

def createDbCaseConfig(db: DB, benchType: CaseLabel, args) -> CaseConfig:
    index = availableIndexes[args.index] if args.index != None else None
    dbCaseConfig = db.case_config_cls(index_type=index)
    # A) from db and benchType figure out the available config options
    flags = []
    if db in CASE_CONFIG_MAP and benchType in CASE_CONFIG_MAP[db]:
        flags = CASE_CONFIG_MAP[db][benchType]
    log.error(f"here are my flags for this db: {flags}")
    config = {}
    # B) for each config option do as follows
    # the code trusts that the order of flags matches all needs
    for flag in flags:
        if flag.label == CaseConfigParamType.IndexType:
            # if it is of type Index, check if selected index is allowed, if not fail
            if index in flag.inputConfig["options"]:
                config[CaseConfigParamType.IndexType] = index
            else:
                raise Exception("index " + str(index) + " not supported in database " + str(db.name))
        else:
            # if it is of a different type, check if config property is set,
            configkey = db.name.lower() + "." + benchType.name.lower() + "." + flag.label.name
            valueSet = getattr(args, configkey)
            if valueSet == None:
                if flag.inputType == InputType.Number:
                    log.info(configkey + " is not set using default")
                    config[flag.label] = flag.inputConfig["value"]
                else:
                    # config is not set, do not put it in config
                    log.info(configkey + " is not set, skipping")
                continue
            elif flag.isDisplayed(config):
                # check if the property can be set depending on index
                # it can be set, make sure to put it in configuration
                if flag.inputType == InputType.Option:
                    log.info(configkey + " is set to " + valueSet)
                    config[flag.label] = valueSet
                elif flag.inputType == InputType.Number:
                    valToUse = flag.inputConfig["value"]
                    if flag.inputConfig["min"] > valueSet:
                        log.info(configkey + " is " + str(valueSet) + " => too small, using default")
                    elif flag.inputConfig["max"] < valueSet:
                        log.info(configkey + " is " + str(valueSet) + " => too large, using default")
                    else:
                        log.info(configkey + " is " + str(valueSet))
                        valToUse = valueSet
                    config[flag.label] = valToUse
                else:
                    log.error("inputType " + str(flag.inputType) + " not supported")
                    raise Exception("inputType " + str(flag.inputType) + " not supported")
            else:
                # it is set, but cannot be set according to index, skip
                log.warning(configkey + " is set, but property not available for index " + str(index) + ", skipping")
    log.error("the config: " + str(config))
    return dbCaseConfig(**{key.value: value for key, value in config.items()})
    # return dbCaseConfig(**config)

addConfigSubgroups()

def main():
    args = parser.parse_args()
    log.info(f"all configs: {args}")

    # should already have been handled by __init__.py
    # handleEnvVariables(args)
    index_use = availableIndexUses[args.index_use] if args.index_use != None else IndexUse.BOTH_KEEP
    benchCase = availableDataSets[args.dataset].case_cls()
    benchType: CaseLabel = availableBenchTypes[args.benchmark_type]

    try:
        if benchCase.label != benchType:
            raise Exception("benchmark type and dataset type do not match")
        
        db=availableDbs[args.database]
        dbconf: DBConfig = createDbConfig(db, args)
        dbCaseConfig: CaseConfig = createDbCaseConfig(db, benchType, args)
        print("dbCaseConfig: " + str(dbCaseConfig) + " --- " + str(type(dbCaseConfig))) 
        myTask = TaskConfig(
                    db=db,
                    case_config=CaseConfig(
                        case_id=benchCase.case_id,
                        custom_case={},
                    ),
                    db_config = dbconf,
                    # FIXME: find out why this is so different
                    db_case_config=dbCaseConfig
        )
        tasks: list[TaskConfig] = [ myTask ]
        # generate_tasks(
        #    [  ]
        #)
        benchMarkRunner.set_index_use(index_use)
        benchMarkRunner.set_drop_old(not args.skip_load_and_index)
        benchMarkRunner.set_download_address(False)
        benchMarkRunner.run(tasks, args.label)
        benchMarkRunner._sync_running_task()
        result = benchMarkRunner.get_results()
        log.info(f"test result: {result}")
        # subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("exit streamlit...")
    except Exception as e:
        log.warning(f"exit, err={e}\nstack trace={traceback.format_exc(chain=True)}")


if __name__ == "__main__":
    main()
