# vectordb-orm

`vectordb-orm` is an Object-Relational Mapping (ORM) library designed to work with vector databases, such as Milvus. The project aims to provide a consistent and convenient interface for working with vector data, allowing you to interact with vector databases using familiar ORM concepts and syntax.

## Getting Started

Here are some example code snippets demonstrating common behavior with vectordb-orm. vectordb-orm is designed around python typehints. You create a class definition by subclassing `MilvusBase` and providing typehints for the keys of your model, similar to pydantic. These fields also support custom initialization behavior if you want (or need) to modify their configuration options.

| Field Type      | Description                                                                                                                                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BaseField       | The `BaseField` provides the ability to add a default value for a given field. This should be used in cases where the more specific field types aren't relevant.                                                                           |
| PrimaryKeyField | The `PrimaryKeyField` is used to specify the primary key of your model, and one is required per class.                                                                                                                                     |
| VarCharField    | The `VarCharField` is used to specify a string field, and the `EmbeddingField` is used to specify a vector field.                                                                                                                          |
| EmbeddingField  | The `EmbeddingField` also supports specifying an index type, which is used to specify the index type for the field. The `EmbeddingField` also supports specifying a dimension, which is used to specify the dimension of the vector field. |

### Object Definition

```python
from vectordb_orm import MilvusBase, EmbeddingField, VarCharField, PrimaryKeyField
from pymilvus import Milvus
from vectordb_orm.indexes import IVF_FLAT
import numpy as np

class MyObject(MilvusBase):
    __collection_name__ = 'my_object_collection'

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=IVF_FLAT(cluster_units=128))
```

## Querying Syntax

```python
from vectordb_orm import MilvusSession

# Instantiate a MilvusSession
session = MilvusSession()

# Perform a simple boolean query
results = session.query(MyObject).filter(MyObject.text == 'bar').limit(2).all()

# Rank results by their similarity to a given reference vector
query_vector = np.array([8.0]*128)
results = session.query(MyObject).filter(MyObject.text == 'bar').order_by_similarity(MyObject.embedding, query_vector).limit(2).all()
```

## Installation

To get started with vectordb-orm, simply install the package and its dependencies, then import the necessary modules:

```bash
pip install vectordb-orm
```

We use poetry for local development work:

```bash
poetry install
poetry run pytest
```

## Why use an ORM?

Most vector databases use a JSON-like querying syntax where schemas and objects are specified as dictionary blobs. This makes it difficult to use IDE features like autocomplete or typehinting, and also can lead to error prone code while translating between Python logic and querying syntax.

An ORM provides a high-level, abstracted interface to work with databases. This abstraction makes it easier to write, read, and maintain code, as well as to switch between different database backends with minimal changes. Furthermore, an ORM allows developers to work with databases in a more Pythonic way, using Python objects and classes instead of raw SQL queries or low-level API calls.

## Comparison to SQLAlchemy

While vectordb-orm is inspired by the widely-used SQLAlchemy ORM, it is specifically designed for vector databases, such as Milvus. This means that vectordb-orm offers unique features tailored to the needs of working with vector data, such as similarity search, index management, and efficient data storage. Although the two ORMs share some similarities in terms of syntax and structure, vectordb-orm focuses on providing a seamless experience for working with vector databases.

## WIP

Please note that vectordb-orm is still a (somewhat large) work in progress. The current implementation focuses on Milvus integration, the goal is to eventually expand support to other vector databases. Contributions and feedback are welcome as we work to improve and expand the capabilities of vectordb-orm.
