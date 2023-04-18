from typing import List
from pymilvus import Milvus, Collection
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from vectordb_orm.attributes import AttributeCompare
from vectordb_orm.fields import EmbeddingField, VarCharField, BaseField, PrimaryKeyField
from typing import Any
import numpy as np


class MilvusBaseMeta(type):
    def __getattr__(cls, name):
        if name in set(cls.__annotations__.keys()):
            return AttributeCompare(cls, name)
        raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")


class MilvusBase(metaclass=MilvusBaseMeta):
    _type_configuration: dict[str, BaseField]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        # Cache the type configurations and then remove the values. We want accessing the property of the
        # class to return the value of the field (or default to AttributeCompare via __getattr__ in the case of
        # a direct class attribute), not the field configuration itself.
        cls._type_configuration = {}
        for key in cls.__annotations__.keys():
            attribute_value = getattr(cls, key)
            cls._type_configuration[key] = attribute_value if isinstance(attribute_value, BaseField) else None
            delattr(cls, key)

    @classmethod
    def collection_name(self) -> str:
        if not hasattr(self, '__collection_name__'):
            raise ValueError(f"Class {self.__name__} does not have a collection name, specify `__collection_name__` on the class definition.")
        return self.__collection_name__

    @classmethod
    def embedding_dim(cls) -> int:
        return cls.__embedding_dim__

    @classmethod
    def _create_collection(cls, milvus_client: Milvus):
        cls._assert_has_primary()

        fields: list[FieldSchema] = []

        for attribute_name, type_hint in cls.__annotations__.items():
            fields.append(
                cls._field_schema_from_typehint(
                    attribute_name,
                    type_hint,
                    cls._type_configuration.get(attribute_name)
                )
            )

        schema = CollectionSchema(fields=fields, description=f"{cls.__name__} vectordb-generated collection")
        collection = Collection(cls.collection_name(), schema)

        # For all embeddings, create an associated index
        for attribute_name, field_configuration in cls._type_configuration.items():
            if isinstance(field_configuration, EmbeddingField):
                index = {
                    "index_type": field_configuration.index.index_type,
                    "metric_type": field_configuration.metric_type.value,
                    "params": field_configuration.index.get_parameters(),
                }
                collection.create_index("embedding", index)

        return collection

    @classmethod
    def _field_schema_from_typehint(cls, name: str, type_hint: Any, field_customization: BaseField | None = None):
        if issubclass(type_hint, np.ndarray):
            if not isinstance(field_customization, EmbeddingField):
                raise ValueError("Embedding typehints should be configured with vectordb.EmbeddingField")
            return FieldSchema(name=name, dtype=DataType.FLOAT_VECTOR, dim=field_customization.dim)
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

    def insert(self, milvus_client: Milvus) -> int:
        entities = self._dict_representation()
        mutation_result = milvus_client.insert(collection_name=self.collection_name(), entities=entities)
        self.id = mutation_result.primary_keys[0]
        return self.id

    def _dict_representation(self):
        type_converters = {
            np.ndarray: DataType.FLOAT_VECTOR,
            str: DataType.VARCHAR,
            int: DataType.INT64,
            float: DataType.DOUBLE,
        }

        return [
            {
                "name": attribute_name,
                "type": type_converters[type_hint],
                "values": [value],
            }
            for attribute_name, type_hint in self.__annotations__.items()
            for value in [getattr(self, attribute_name)]
            if value is not None
        ]

    @classmethod
    def _assert_has_primary(cls):
        """
        Ensure we have a primary key, this is the only field that's fully required
        """
        if not any(
            [
                isinstance(cls._type_configuration.get(attribute_name), PrimaryKeyField)
                for attribute_name in cls.__annotations__.keys()
            ]
        ):
            raise ValueError(f"Class {cls.__name__} does not have a primary key, specify `PrimaryKeyField` on the class definition.")
