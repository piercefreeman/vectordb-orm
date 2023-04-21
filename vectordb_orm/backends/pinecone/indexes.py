from enum import Enum

from vectordb_orm.index import IndexBase


class PineconeSimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotproduct"


class PineconeIndex(IndexBase):
    """
    Pinecone only supports one type of index
    """
    def __init__(self, metric_type: PineconeSimilarityMetric):
        self.metric_type = metric_type
        self._assert_metric_type(metric_type)

    def get_index_parameters(self):
        return {}

    def get_inference_parameters(self):
        return {"metric_type": self.metric_type.name}

    def _assert_metric_type(self, metric_type: PineconeSimilarityMetric):
        # Only support valid combinations of metric type and index
        if not isinstance(metric_type, PineconeSimilarityMetric):
            raise ValueError(f"Index type {self} is not supported for metric type {metric_type}")
