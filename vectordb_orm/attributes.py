from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectordb_orm.base import MilvusBase


class OperationType(Enum):
    """
    Types fo comparison supported by our querying language

    """
    EQUALS = 'EQUALS'
    GREATER_THAN = 'GREATER_THAN'
    LESS_THAN = 'LESS_THAN'
    LESS_THAN_EQUAL = 'LESS_THAN_EQUAL'
    GREATER_THAN_EQUAL = 'GREATER_THAN_EQUAL'
    NOT_EQUAL = 'NOT_EQUAL'


class AttributeCompare:
    """
    Attributes accessed on the class level are used for query construction like filtering or retrieval. This
    class allows for Python-native comparison definition like `MyObj.text == "value"`. The operation type is
    used to construct the query expression at execution time.

    """
    def __init__(self, base_cls: "MilvusBase", attr: str, value: Any = None, op: OperationType | None = None):
        self.base_cls = base_cls
        self.attr = attr
        self.value = value
        self.op = op

    def __eq__(self, other):
        return AttributeCompare(self.base_cls, self.attr, other, OperationType.EQUALS)

    def __gt__(self, other):
        return AttributeCompare(self.base_cls, self.attr, other, OperationType.GREATER_THAN)

    def __lt__(self, other):
        return AttributeCompare(self.base_cls, self.attr, other, OperationType.LESS_THAN)

    def __le__(self, other):
        return AttributeCompare(self.base_cls, self.attr, other, OperationType.LESS_THAN_EQUAL)

    def __ge__(self, other):
        return AttributeCompare(self.base_cls, self.attr, other, OperationType.GREATER_THAN_EQUAL)
