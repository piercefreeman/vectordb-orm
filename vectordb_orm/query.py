from pymilvus import Milvus
from pymilvus.client.abstract import ChunkedQueryResult
from vectordb_orm.attributes import AttributeCompare
from typing import TYPE_CHECKING
from vectordb_orm.results import QueryResult
from vectordb_orm.fields import EmbeddingField
from vectordb_orm.base import MilvusBase


class MilvusQueryBuilder:
    """
    Recursive query builder to allow for chaining of queries.

    Example:
        >>> query(MyObject).filter(MyObject.text == "hello").limit(10).all()
        >>> query(MyObject.text).filter(MyObject.text == "hello").limit(10).all()
        >>> query(MyObject.text, MyObject.embedding).filter(MyObject.text == "hello").limit(10).all()

    """
    def __init__(
        self,
        milvus_client: Milvus,
        *query_objects: MilvusBase | AttributeCompare
    ):
        if len(query_objects) == 0:
            raise ValueError("Must specify at least one query object.")

        self.milvus_client = milvus_client
        if isinstance(query_objects[0], AttributeCompare):
            self.cls = query_objects[0].base_cls
        elif issubclass(query_objects[0], MilvusBase):
            self.cls = query_objects[0]
        else:
            raise ValueError(f"Query object must be of type `MilvusBase` or `AttributeCompare`, got {type(query_objects[0])}")

        # Queries created over time
        self._query_objects = list(query_objects)
        self._filters: list[str] = []
        self._limit : int | None = None

        # Set if similarity ranking is used
        self._similarity_attribute : AttributeCompare | None = None
        self._similarity_value = None

    def filter(self, *filters: AttributeCompare):
        for f in filters:
            self._filters.append(f.to_expression())
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
        filters = " and ".join(self._filters) if self._filters else None
        query_records = [self._similarity_value] if self._similarity_value is not None else None
        embedding_field_name = self._similarity_attribute.attr if self._similarity_attribute else None

        params = {"nprobe": 16}

        search_result = self.milvus_client.search(
            data=query_records,
            anns_field=embedding_field_name,
            param=params,
            limit=self._limit,
            collection_name=self.cls.collection_name(),
            expression=filters,
            output_fields=self._get_output_fields(),
        )
        return self._result_to_objects(search_result)

    def _get_output_fields(self) -> list[str]:
        """
        Milvus requires us to specify the fields we're looking for explicitly. We by default ignore
        the embedding field since it's a lot of data to send over the wire and therefore require
        users to ask for this explicitly in the query definition.

        """
        fields : set[str] = set()
        for query_object in self._query_objects:
            if isinstance(query_object, AttributeCompare):
                fields.add(query_object.attr)
            elif issubclass(query_object, MilvusBase):
                for attribute_name in self.cls.__annotations__.keys():
                    field_configuration = self.cls._type_configuration.get(attribute_name)
                    if field_configuration and isinstance(field_configuration, EmbeddingField):
                        continue
                    fields.add(attribute_name)
        return list(fields)

    def _result_to_objects(self, search_result: ChunkedQueryResult):
        query_results : list[QueryResult] = []

        for hit in search_result:
            for result in hit:
                obj = self.cls(text=result.entity.get("text"), embedding=None)
                obj.id = result.id
                query_results.append(QueryResult(obj, result.score, result.distance))

        return query_results
