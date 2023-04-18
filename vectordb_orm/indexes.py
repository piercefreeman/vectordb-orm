from abc import ABC, abstractmethod

class IndexBase(ABC):
    """
    Specify indexes used for embedding creation: https://milvus.io/docs/index.md

    """
    index_type: str

    @abstractmethod
    def get_parameters(self):
        pass


class IVF_FLAT(IndexBase):
    """
    - High-speed query
    - Requires a recall rate as high as possible
    """
    index_type = "IVF_FLAT"

    def __init__(self, cluster_units: int):
        """
        :param cluster_units: Number of clusters (nlist in the docs)
        """
        if not (cluster_units >= 1 and cluster_units <= 65536):
            raise ValueError("cluster_units must be between 1 and 65536")
        self.nlist = cluster_units

    def get_parameters(self):
        return {"nlist": self.nlist}
