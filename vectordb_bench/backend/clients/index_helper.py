from .api import IndexUse

def createIndexForLoad(index_use: IndexUse) -> bool:
    return (index_use == IndexUse.LOAD or
        index_use == IndexUse.BOTH_RESET or
        index_use == IndexUse.BOTH_KEEP)
