from enum import Enum

from pymilvus.orm.types import (CONSISTENCY_BOUNDED, CONSISTENCY_EVENTUALLY,
                                CONSISTENCY_SESSION, CONSISTENCY_STRONG)


class ConsistencyType(Enum):
    """
    Define the strength of the consistency within the distributed DB:
    https://milvus.io/docs/consistency.md

    """
    STRONG = CONSISTENCY_STRONG
    BOUNDED = CONSISTENCY_BOUNDED
    SESSION = CONSISTENCY_SESSION
    EVENTUALLY = CONSISTENCY_EVENTUALLY
