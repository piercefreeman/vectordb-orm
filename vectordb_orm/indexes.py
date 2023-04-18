from abc import ABC, abstractmethod
from vectordb_orm.similarity import FloatSimilarityMetric, BinarySimilarityMetric

class IndexBase(ABC):
    """
    Specify indexes used for embedding creation: https://milvus.io/docs/index.md

    """
    index_type: str

    def __init__(
        self,
        metric_type: FloatSimilarityMetric | BinarySimilarityMetric | None = None,
    ):
        # Choose a reasonable default if metric_type is null, depending on the type of index
        if metric_type is None:
            if isinstance(self, tuple(FLOATING_INDEXES)):
                metric_type = FloatSimilarityMetric.L2
            elif isinstance(self, tuple(BINARY_INDEXES)):
                metric_type = BinarySimilarityMetric.JACCARD

        self._assert_metric_type(metric_type)
        self.metric_type = metric_type

    @abstractmethod
    def get_index_parameters(self):
        pass

    @abstractmethod
    def get_inference_parameters(self):
        """
        NOTE: For simplicity, each index type will have the same inference parameters as index parameters. Milvus does allow
        for the customization of these per-query, but we will see how this use-case develops.
        """
        pass

    def _assert_metric_type(self, metric_type: FloatSimilarityMetric | BinarySimilarityMetric):
        """
        Binary indexes only support binary metrics, and floating indexes only support floating metrics. Assert
        that the combination of metric type and index is valid.

        """
        # Only support valid combinations of metric type and index
        if isinstance(metric_type, FloatSimilarityMetric):
            if not isinstance(self, tuple(FLOATING_INDEXES)):
                raise ValueError(f"Index type {self} is not supported for metric type {metric_type}")
        elif isinstance(metric_type, BinarySimilarityMetric):
            if not isinstance (self, tuple(BINARY_INDEXES)):
                raise ValueError(f"Index type {self} is not supported for metric type {metric_type}")


class IVF_FLAT(IndexBase):
    """
    - High-speed query
    - Requires a recall rate as high as possible
    """
    index_type = "IVF_FLAT"

    def __init__(
        self,
        cluster_units: int,
        inference_comparison: int | None = None,
        metric_type: FloatSimilarityMetric | BinarySimilarityMetric | None = None,
    ):
        """
        :param cluster_units: Number of clusters (nlist in the docs)
        :param inference_comparison: Number of cluster centroids to compare during inference (nprobe in the docs)
            By default if this is not specified, it will be set to the same value as cluster_units.
        """
        super().__init__(metric_type=metric_type)

        if not (cluster_units >= 1 and cluster_units <= 65536):
            raise ValueError("cluster_units must be between 1 and 65536")
        if inference_comparison is not None and not (inference_comparison >= 1 and inference_comparison <= cluster_units):
            raise ValueError("inference_comparison must be between 1 and cluster_units")
        self.nlist = cluster_units
        self.nprobe = inference_comparison or cluster_units

    def get_index_parameters(self):
        return {"nlist": self.nlist}

    def get_inference_parameters(self):
        return {"nprobe": self.nprobe}


class BIN_FLAT(IndexBase):
    index_type = "BIN_FLAT"

    def get_index_parameters(self):
        return {}

    def get_inference_parameters(self):
        return {"metric_type": self.metric_type.name}


FLOATING_INDEXES : set[IndexBase] = {IVF_FLAT}
BINARY_INDEXES : set[IndexBase] = {BIN_FLAT}
