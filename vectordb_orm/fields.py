from abc import ABC
from typing import Any

from vectordb_orm.index import IndexBase


class BaseField(ABC):
    """
    BaseField is the superclass to all fields that require additional customization behavior. They
    are specified as values for the typehints that otherwise populate the class, like:

    ```
    class MyModel:
        embedding: np.array = EmbeddingField(dim=16)
    ```

    """
    def __init__(self, default: Any):
        self.default = default


class PrimaryKeyField(BaseField):
    def __init__(self):
        super().__init__(default=None)


class EmbeddingField(BaseField):
    def __init__(
        self,
        dim: int,
        index: IndexBase,
        default: Any = None,
    ):
        super().__init__(default=default)

        self.dim = dim
        self.index = index


class VarCharField(BaseField):
    def __init__(self, max_length: int, default: Any = None):
        super().__init__(default=default)

        self.max_length = max_length
