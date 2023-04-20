from enum import Enum
from pymilvus.client.types import MetricType
from pymilvus.orm.types import CONSISTENCY_STRONG, CONSISTENCY_BOUNDED, CONSISTENCY_EVENTUALLY, CONSISTENCY_SESSION

class FloatSimilarityMetric(Enum):
    """
    Specify the metric used for floating-point search. At inference time a query vector is broadcast to the vectors in the database
    using this approach. The string values of this enums are directly used by Milvus, see here for more info: https://milvus.io/docs/metric.md

    """
    # Euclidean
    L2 = MetricType.L2.name
    # Inner Product
    IP = MetricType.IP.name

class BinarySimilarityMetric(Enum):
    """
    Specify the metric used for binary search. These are distance metrics.

    """
    JACCARD = MetricType.JACCARD.name
    TANIMOTO = MetricType.TANIMOTO.name
    HAMMING = MetricType.HAMMING.name

class ConsistencyType(Enum):
    """
    Define the strength of the consistency within the distributed DB:
    https://milvus.io/docs/consistency.md

    """
    STRONG = CONSISTENCY_STRONG
    BOUNDED = CONSISTENCY_BOUNDED
    SESSION = CONSISTENCY_SESSION
    EVENTUALLY = CONSISTENCY_EVENTUALLY
