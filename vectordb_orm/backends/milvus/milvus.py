from collections import defaultdict
from logging import info
from typing import Any, Type, get_args, get_origin

import numpy as np
from pymilvus import Collection, Milvus
from pymilvus.client.abstract import ChunkedQueryResult
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema

from vectordb_orm.attributes import AttributeCompare, OperationType
from vectordb_orm.backends.base import BackendBase
from vectordb_orm.backends.milvus.indexes import (BINARY_INDEXES,
                                                  FLOATING_INDEXES)
from vectordb_orm.base import VectorSchemaBase
from vectordb_orm.fields import (BaseField, EmbeddingField, PrimaryKeyField,
                                 VarCharField)
from vectordb_orm.results import QueryResult


class MilvusBackend(BackendBase):
    # https://milvus.io/docs/search.md
    max_fetch_size = 16384

    def __init__(self, milvus_client: Milvus):
        self.client = milvus_client

    def create_collection(self, schema: Type[VectorSchemaBase]):
        self._assert_embedding_validity(schema)
        self._assert_has_primary(schema)

        fields: list[FieldSchema] = []

        for attribute_name, type_hint in schema.__annotations__.items():
            fields.append(
                self._field_schema_from_typehint(
                    attribute_name,
                    type_hint,
                    schema._type_configuration.get(attribute_name)
                )
            )

        print(f"Creating collection {schema.collection_name()} with schema {fields}")

        field_schema = CollectionSchema(fields=fields, description=f"{schema.__name__} vectordb-generated collection")
        collection = Collection(schema.collection_name(), field_schema)

        # For all embeddings, create an associated index
        for attribute_name, field_configuration in schema._type_configuration.items():
            if isinstance(field_configuration, EmbeddingField):
                index = {
                    "index_type": field_configuration.index.index_type,
                    "metric_type": field_configuration.index.metric_type.value,
                    "params": field_configuration.index.get_index_parameters(),
                }
                print(f"Creating index {index} for field {attribute_name}")
                collection.create_index("embedding", index)

        return collection

    def clear_collection(self, schema: Type[VectorSchemaBase]):
        # Since Milvus can only delete entities by listing explicit primary keys,
        # the most efficient way to clear the collection is to fully delete and recreate it
        self.delete_collection(schema)
        self.create_collection(schema)

    def delete_collection(self, schema: Type[VectorSchemaBase]):
        self.client.drop_collection(schema.collection_name())

    def insert(self, entity: VectorSchemaBase) -> int:
        entities = self._dict_representation(entity)
        print("Entity insertion", entities)
        mutation_result = self.client.insert(collection_name=entity.__class__.collection_name(), entities=entities)
        return mutation_result.primary_keys[0]

    def insert_batch(
        self,
        entities: list[VectorSchemaBase],
        show_progress: bool,
    ) -> list[int]:
        if show_progress:
            raise ValueError("Milvus backend does not support batch insertion progress logging because it is done in one operation.")

        # Group by the schema type since we allow for the insertion of multiple different schema
        # `schema_to_entities` - input entities grouped by the schema name
        # `schema_to_original_index` - since we switch to a per-schema representation, keep track of a mapping
        #    from the schema to the original index in `entities`
        # `schema_to_ids` - map of schema to the resulting primary keys
        schema_to_original_index = defaultdict(list)
        schema_to_class = {}
        schema_to_ids = {}

        for i, entity in enumerate(entities):
            schema = entity.__class__
            schema_name = schema.collection_name()

            schema_to_original_index[schema_name].append(i)
            schema_to_class[schema_name] = schema

        for schema_name, schema_indexes in schema_to_original_index.items():
            schema = schema_to_class[schema_name]
            schema_entities = [entities[index] for index in schema_indexes]

            # The primary key should be null at this stage of things, so we ignore it
            # during the insertion
            ignore_keys = {
                self._get_primary(schema)
            }

            # Group this schema's objects by their keys
            by_key_values = defaultdict(list)
            by_key_type = {}

            for entity in schema_entities:
                for attribute_name, type_hint in entity.__annotations__.items():
                    value = getattr(entity, attribute_name)
                    db_type, value = self._type_to_value(type_hint, value)
                    by_key_values[attribute_name].append(value)
                    by_key_type[attribute_name] = db_type

            # Ensure each key in `schema_to_objects` matches the quantity of objects
            # that should have been created. This *shouldn't* happen but it's possible
            # some combination of programatically deleting attributes or annotations will
            # lead to this case. We proactively raise an error because this could result in
            # data corruption.
            all_lengths = {len(values) for values in by_key_values.values()}
            if len(all_lengths) > 1:
                raise ValueError(f"Inserted objects don't align for schema `{schema_name}`")

            payload = [
                {
                    "name": attribute_name,
                    "type": by_key_type[attribute_name],
                    "values": values,
                }
                for attribute_name, values in by_key_values.items()
                if attribute_name not in ignore_keys
            ]

            mutation_result = self.client.insert(collection_name=schema_name, entities=payload)
            schema_to_ids[schema_name] = mutation_result.primary_keys

        # Reorder ids to match the input entities
        ordered_ids = [None] * len(entities)
        for schema_name, primary_keys in schema_to_ids.items():
            for i, original_index in enumerate(schema_to_original_index[schema_name]):
                ordered_ids[original_index] = primary_keys[i]

        return ordered_ids

    def delete(self, entity: VectorSchemaBase):
        schema = entity.__class__
        identifier_key = self._get_primary(schema)
        # Milvus only supports deleting entities with the `in` conditional; equality doesn't work
        delete_expression = f"{identifier_key} in [{entity.id}]"
        self.client.delete(collection_name=schema.collection_name(), expr=delete_expression)

    def search(
        self,
        schema: Type[VectorSchemaBase],
        output_fields: list[str],
        filters: list[AttributeCompare] | None,
        search_embedding: np.ndarray | None,
        search_embedding_key: str | None,
        limit: int,
        offset: int,
    ):
        filters = " and ".join(
            [
                self._attribute_to_expression(f)
                for f in filters
            ]
        ) if filters else None

        optional_args = dict()
        if schema.consistency_type() is not None:
            optional_args["consistency_level"] = schema.consistency_type().value

        # Milvus supports to different quering behaviors depending on whether or not we are looking
        # for vector similarities
        # A `search` is used when we are looking for vector similarities, and a `query` is used more akin
        # to a traditional relational database when we're just looking to filter on metadata
        if search_embedding_key is not None:
            embedding_configuration : EmbeddingField = schema._type_configuration.get(search_embedding_key)

            # Go through the same type conversion as the embedding field during insert time
            _, similarity_value = self._type_to_value(
                schema.__annotations__[search_embedding_key],
                search_embedding,
            )
            query_records = [similarity_value]

            search_result = self.client.search(
                data=query_records,
                anns_field=search_embedding_key,
                param=embedding_configuration.index.get_inference_parameters(),
                limit=limit,
                offset=offset,
                collection_name=schema.collection_name(),
                expression=filters,
                output_fields=output_fields,
                **optional_args,
            )
        else:
            search_result = self.client.query(
                expr=filters,
                offset=offset,
                limit=limit,
                output_fields=output_fields,
                collection_name=schema.collection_name(),
                **optional_args,
            )
        return self._result_to_objects(schema, search_result)

    def flush(self, schema: Type[VectorSchemaBase]):
        self.client.flush([schema.collection_name()])

    def load(self, schema: Type[VectorSchemaBase]):
        self.client.load_collection(schema.collection_name())

    def _field_schema_from_typehint(cls, name: str, type_hint: Any, field_customization: BaseField | None = None):
        """
        Create a FieldSchema based on the provided type hint and field customization.

        This function is called internally during the creation of a MilvusBase instance.

        :param name: The name of the field.
        :param type_hint: The type hint for the field.
        :param field_customization: An optional customization object for the field.
        :returns: A FieldSchema instance.
        :raises ValueError: If an unsupported type hint is provided or the type hint configuration is incorrect.
        """
        if issubclass(extract_base_type(type_hint), np.ndarray):
            if not isinstance(field_customization, EmbeddingField):
                raise ValueError("Embedding typehints should be configured with vectordb.EmbeddingField")

            type_arguments = get_args(type_hint)
            vector_type = (
                DataType.BINARY_VECTOR
                if type_arguments and type_arguments[0] == np.bool_
                else DataType.FLOAT_VECTOR
            )

            return FieldSchema(name=name, dtype=vector_type, dim=field_customization.dim)
        elif issubclass(type_hint, str):
            if not isinstance(field_customization, VarCharField):
                raise ValueError("String typehints should be configured with vectordb.VarCharField")
            return FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=field_customization.max_length)
        elif issubclass(type_hint, int):
            # We currently only support ints as the primary key
            is_primary = isinstance(field_customization, PrimaryKeyField)

            # Mirror the python int sizing
            return FieldSchema(name=name, dtype=DataType.INT64, is_primary=is_primary, auto_id=is_primary)
        elif issubclass(type_hint, float):
            # Mirror the python float sizing, equal to a double
            return FieldSchema(name=name, dtype=DataType.DOUBLE)
        else:
            raise ValueError(f"{cls.__name__}: Typehint {name}:{type_hint} is not supported")

    def _type_to_value(self, type_hint, value: Any | None = None):
        """
        Use the typehint signatures to convert into milvus DataType. Also convert the values
        into the correct format for milvus insertion.

        """
        if issubclass(extract_base_type(type_hint), np.ndarray):
            type_arguments = get_args(type_hint)
            vector_type = (
                DataType.BINARY_VECTOR
                if type_arguments and type_arguments[0] == np.bool_
                else DataType.FLOAT_VECTOR
            )
            if vector_type == DataType.BINARY_VECTOR:
                if value is not None:
                    packed_uint8_array = np.packbits(value)
                    #value = packed_uint8_array.tobytes()
                    value = bytes(packed_uint8_array.tolist())
                return vector_type, value
            else:
                if value is not None:
                    value = value.tolist()
                return vector_type, value
        elif issubclass(type_hint, str):
            return DataType.VARCHAR, value
        elif issubclass(type_hint, int):
            return DataType.INT64, value
        elif issubclass(type_hint, float):
            return DataType.DOUBLE, value
        else:
            raise ValueError(f"Typehint {type_hint} is not supported")

    def _dict_representation(self, entity: VectorSchemaBase):
        """
        Convert the MilvusBase object to a dictionary representation for storage.

        This function is called internally during the insertion of a MilvusBase object.

        :returns: A list of dictionaries containing the name, type, and values of the attributes.
        """
        payload = []

        for attribute_name, type_hint in entity.__annotations__.items():
            value = getattr(entity, attribute_name)
            if value is not None:
                db_type, value = self._type_to_value(type_hint, value)
                payload.append(
                    {
                        "name": attribute_name,
                        "type": db_type,
                        "values": [value]
                    }
                )
        return payload

    def _assert_embedding_validity(self, schema: Type[VectorSchemaBase]):
        """
        Ensure that the embedding configurations align with the typehinting
        """
        # For all embeddings, create an associated index
        for attribute_name, field_configuration in schema._type_configuration.items():
            if not isinstance(field_configuration, EmbeddingField):
                continue

            field_index = field_configuration.index
            vector_type, _ = self._type_to_value(schema.__annotations__[attribute_name], None)
            # Ensure that the index type is compatible with these fields
            if vector_type == DataType.BINARY_VECTOR and not isinstance(field_index, tuple(BINARY_INDEXES)):
                raise ValueError(f"Index type {field_index} is not compatible with binary vectors.")
            elif vector_type == DataType.FLOAT_VECTOR and not isinstance(field_index, tuple(FLOATING_INDEXES)):
                raise ValueError(f"Index type {field_index} is not compatible with float vectors.")

            # Milvus max size
            # https://milvus.io/docs/limitations.md
            if field_configuration.dim > 32768:
                raise ValueError(f"Milvus only supports vectors with dimensions under 32768. {attribute_name} is too large: {field_configuration.dim}.")

    def _result_to_objects(
        self,
        schema: Type[VectorSchemaBase],
        search_result: ChunkedQueryResult | list[dict[str, Any]]
    ):
        query_results : list[QueryResult] = []

        if isinstance(search_result, ChunkedQueryResult):
            for hit in search_result:
                for result in hit:
                    entity = {
                        key: result.entity.get(key)
                        for key in result.entity.fields
                    }
                    obj = schema.from_dict(entity)
                    query_results.append(QueryResult(obj, score=result.score, distance=result.distance))
        else:
            for result in search_result:
                obj = schema.from_dict(result)
                query_results.append(QueryResult(obj))

        return query_results

    def _attribute_to_expression(self, attribute: AttributeCompare):
        value = attribute.value
        if isinstance(value, str):
            value = f"\"{attribute.value}\""

        operation_type_maps = {
            OperationType.EQUALS: '==',
            OperationType.GREATER_THAN: '>',
            OperationType.LESS_THAN: '<',
            OperationType.LESS_THAN_EQUAL: '<=',
            OperationType.GREATER_THAN_EQUAL: '>=',
            OperationType.NOT_EQUAL: '!='
        }

        return f"{attribute.attr} {operation_type_maps[attribute.op]} {value}"

def extract_base_type(type):
    """
    Given a type like `np.ndarray[np.float32]`, return the root type `np.ndarray`.
    Also works if passed a non-generic type like `np.ndarray`.
    """
    return get_origin(type) or type
