from typing import Any

from vectordb_orm.attributes import AttributeCompare
from vectordb_orm.backends.base import BackendBase
from vectordb_orm.base import VectorSchemaBase
from vectordb_orm.fields import EmbeddingField


class VectorQueryBuilder:
    """
    Recursive query builder to allow for chaining of queries.

    Example:
        >>> query(MyObject).filter(MyObject.text == "hello").limit(10).all()
        >>> query(MyObject.text).filter(MyObject.text == "hello").limit(10).all()
        >>> query(MyObject.text, MyObject.embedding).filter(MyObject.text == "hello").limit(10).all()

    """
    def __init__(
        self,
        backend: BackendBase,
        *query_objects: VectorSchemaBase | AttributeCompare
    ):
        if len(query_objects) == 0:
            raise ValueError("Must specify at least one query object.")

        self.backend = backend
        if isinstance(query_objects[0], AttributeCompare):
            self.cls = query_objects[0].base_cls
        elif issubclass(query_objects[0], VectorSchemaBase):
            self.cls = query_objects[0]
        else:
            raise ValueError(f"Query object must be of type `VectorSchemaBase` or `AttributeCompare`, got {type(query_objects[0])}")

        # Queries created over time
        self._query_objects = list(query_objects)
        self._filters: list[str] = []
        self._limit : int | None = None
        self._offset : int | None = None

        # Set if similarity ranking is used
        self._similarity_attribute : AttributeCompare | None = None
        self._similarity_value = None

    def filter(self, *filters: AttributeCompare):
        for f in filters:
            self._filters.append(f)
        return self

    def offset(self, offset: int):
        self._offset = offset
        return self

    def limit(self, limit: int | None):
        self._limit = limit
        return self

    def order_by_similarity(self, embedding_attr: AttributeCompare, query_embedding):
        if self._similarity_attribute is not None:
            raise ValueError("Only one similarity ordering can be used per query.")
        self._similarity_attribute = embedding_attr
        self._similarity_value = query_embedding
        return self

    def all(self):
        # Global configuration
        output_fields = self._get_output_fields()
        offset = self._offset if self._offset is not None else 0
        # Sum of limit and offset should be less than MAX_MILVUS_INT
        #limit = self._limit if self._limit is not None else (MAX_MILVUS_INT - offset)
        # TODO: offset is only relevant really for Milvus backends
        limit = self._limit if self._limit is not None else (self.backend.max_fetch_size - offset)

        return self.backend.search(
            schema=self.cls,
            output_fields=output_fields,
            filters=self._filters,
            search_embedding=self._similarity_value,
            search_embedding_key=self._similarity_attribute.attr if self._similarity_attribute is not None else None,
            limit=limit,
            offset=offset,
        )

    def _get_output_fields(self) -> list[str]:
        """
        Milvus requires us to specify the fields we're looking for explicitly. We by default ignore
        the embedding field since it's a lot of data to send over the wire and therefore require
        users to ask for this explicitly in the query definition.

        """
        fields : set[str] = set()
        for query_object in self._query_objects:
            if isinstance(query_object, AttributeCompare):
                field_configuration = self.cls._type_configuration.get(query_object.attr)
                if field_configuration and isinstance(field_configuration, EmbeddingField):
                    raise ValueError("Cannot query for embedding field. Use `order_by_similarity` instead.")
                fields.add(query_object.attr)
            elif issubclass(query_object, VectorSchemaBase):
                for attribute_name in self.cls.__annotations__.keys():
                    field_configuration = self.cls._type_configuration.get(attribute_name)
                    if field_configuration and isinstance(field_configuration, EmbeddingField):
                        continue
                    fields.add(attribute_name)
        return list(fields)
