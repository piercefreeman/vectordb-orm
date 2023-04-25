from typing import Any

from vectordb_orm.attributes import AttributeCompare
from vectordb_orm.enums import ConsistencyType
from vectordb_orm.fields import BaseField, PrimaryKeyField


class VectorSchemaBaseMeta(type):
    def __getattr__(cls, name):
        if name in set(cls.__annotations__.keys()):
            return AttributeCompare(cls, name)
        raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")


class VectorSchemaBase(metaclass=VectorSchemaBaseMeta):
    _type_configuration: dict[str, BaseField]

    def __init__(self, *values: Any, **data: Any) -> None:
        """
        Initialize a new data object using keyword arguments. This class is both used for creating new objects
        and for querying existing objects. The keyword arguments are used to specify the attributes of the object
        to be created.

        :param data: Keyword arguments with attribute names as keys.
        :raises ValueError: If positional arguments are provided or an unexpected keyword argument is encountered.
        """
        if values:
            raise ValueError("Use keyword arguments to initialize a Milvus object.")

        allowed_keys = set(self.__class__.__annotations__.keys())

        for key, value in data.items():
            if key not in allowed_keys:
                raise ValueError(f"Unexpected keyword argument '{key}'")

        for key in allowed_keys:
            if key in data:
                setattr(self, key, data[key])
            else:
                field_configuration = self._type_configuration.get(key)
                if field_configuration:
                    setattr(self, key, field_configuration.default)
                else:
                    raise ValueError(f"Missing required argument '{key}'")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        # Cache the type configurations and then remove the values. We want accessing the property of the
        # class to return the value of the field (or default to AttributeCompare via __getattr__ in the case of
        # a direct class attribute), not the field configuration itself.
        cls._type_configuration = {}
        for key in cls.__annotations__.keys():
            attribute_value = getattr(cls, key)
            cls._type_configuration[key] = attribute_value if isinstance(attribute_value, BaseField) else None
            # All attributes will result in some value because we return a AttributeCompare by default
            # for non-configured objects. Checking that the field is actually equal to a base
            # value allows us to only delete legitimate configuration objects. Otherwise attempting to delete
            # the synthetic AttributeCompare object will throw an error.
            if isinstance(attribute_value, BaseField):
                delattr(cls, key)

    @classmethod
    def collection_name(self) -> str:
        if not hasattr(self, '__collection_name__'):
            raise ValueError(f"Class {self.__name__} does not have a collection name, specify `__collection_name__` on the class definition.")
        return self.__collection_name__

    @classmethod
    def consistency_type(self) -> ConsistencyType | None:
        if not hasattr(self, '__consistency_type__'):
            return None
        return self.__consistency_type__

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a MilvusBase object from a dictionary.

        :param data: A dictionary containing the attribute names and values.
        :returns: A MilvusBase object.
        :raises ValueError: If an unexpected attribute name is encountered in the dictionary.
        """
        init_payload = {}
        allowed_keys = list(cls.__annotations__.keys())
        for attribute_name, value in data.items():
            if attribute_name not in allowed_keys:
                raise ValueError(f"Key `{attribute_name}` is not allowed on {cls.__name__}")
            init_payload[attribute_name] = value
        return cls(**init_payload)
