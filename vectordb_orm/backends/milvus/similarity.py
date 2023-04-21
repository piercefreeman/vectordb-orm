from enum import Enum

from pymilvus.client.types import MetricType


class MilvusFloatSimilarityMetric(Enum):
    """
    Specify the metric used for floating-point search. At inference time a query vector is broadcast to the vectors in the database
    using this approach. The string values of this enums are directly used by Milvus, see here for more info: https://milvus.io/docs/metric.md

    """
    # Euclidean
    L2 = MetricType.L2.name
    # Inner Product
    IP = MetricType.IP.name

class MilvusBinarySimilarityMetric(Enum):
    """
    Specify the metric used for binary search. These are distance metrics.

    """
    JACCARD = MetricType.JACCARD.name
    TANIMOTO = MetricType.TANIMOTO.name
    HAMMING = MetricType.HAMMING.name
