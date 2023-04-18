from abc import ABC, abstractmethod

class IndexBase(ABC):
    """
    Specify indexes used for embedding creation: https://milvus.io/docs/index.md

    """
    index_type: str

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
    ):
        """
        :param cluster_units: Number of clusters (nlist in the docs)
        :param inference_comparison: Number of cluster centroids to compare during inference (nprobe in the docs)
            By default if this is not specified, it will be set to the same value as cluster_units.
        """
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
