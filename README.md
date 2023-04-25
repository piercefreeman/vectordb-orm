# vectordb-orm

`vectordb-orm` is an Object-Relational Mapping (ORM) library for vector databases. Define your data as objects and query for them using familiar SQL syntax, with all the added power of lighting fast vector search.

Right now [Milvus](https://milvus.io/) and [Pinecone](https://www.pinecone.io/) are supported with more backend engines planned for the future.

## Getting Started

Here are some simple examples demonstrating common behavior with vectordb-orm. First a note on structure. vectordb-orm is designed around the idea of a `Schema`, which is logically equivalent to a table in classic relational databases. This schema is marked up with typehints that define the type of vector and metadata that will be stored alongisde the objects.

You create a class definition by subclassing `VectorSchemaBase` and providing typehints for the keys of your model, similar to pydantic. These fields also support custom initialization behavior if you want (or need) to modify their configuration options.

### Object Definition

Defining a schema is almost entirely the same between backends but there are some small differences when it comes to index creation. In the below example the Milvus schema requires a Milvus index like `Milvus_IVF_FLAT` (for the full list of supported values see [here](./tree/main/vectordb_orm/backends/milvus/indexes.py)) and the Pinecone schema uses the default `PineconeIndex` with a cosine similarity metric.

Milvus:

```python
from vectordb_orm import VectorSchemaBase, EmbeddingField, VarCharField, PrimaryKeyField, Milvus_IVF_FLAT
import numpy as np

class MyObject(VectorSchemaBase):
    __collection_name__ = 'my_object_collection'

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=Milvus_IVF_FLAT(cluster_units=128))
```

Pinecone:

```python
from vectordb_orm import VectorSchemaBase, EmbeddingField, VarCharField, PrimaryKeyField, PineconeIndex, PineconeSimilarityMetric
import numpy as np

class MyObject(VectorSchemaBase):
    __collection_name__ = 'my_object_collection'

    id: int = PrimaryKeyField()
    text: str = VarCharField(max_length=128)
    embedding: np.ndarray = EmbeddingField(dim=128, index=PineconeIndex(metric_type=PineconeSimilarityMetric.COSINE))
```

### Indexing Data

To insert objects into the database, create a new instance of your object class and insert into the current session. The arguments to the init function mirror the typehinted schema that you defined above.

```python
obj = MyObject(text="my_text", embedding=np.array([1.0]*128))
session.insert(obj)
```

Once inserted, this object will be populated with a new `id` by the database engine. At this point it should be queryable (modulo some backends taking time for eventual consistency across different shards).

`vectordb-orm` also supports batch insertion. This is recommended in cases where you have a lot of data to insert at one time, since latencies can be significant on individual datapoints.

```python
obj = MyObject(text="my_text", embedding=np.array([1.0]*128))
session.insert_batch([obj], show_progress=True)
```

The optional `show_progress` allows you to show a progress bar to show the current status of the insertion and the estimated time remaining for the whole dataset.

### Querying Syntax

```python
session = VectorSession(...)

# Perform a simple boolean query
results = session.query(MyObject).filter(MyObject.text == 'bar').limit(2).all()

# Rank results by their similarity to a given reference vector
query_vector = np.array([8.0]*128)
results = session.query(MyObject).filter(MyObject.text == 'bar').order_by_similarity(MyObject.embedding, query_vector).limit(2).all()
```

### Session Creation

Milvus:

```python
from pymilvus import Milvus, connections
from vectordb_orm import MilvusBackend, VectorSession

# Instantiate a Milvus session
session = VectorSession(MilvusBackend(Milvus()))
connections.connect("default", host="localhost", port="19530")
session.create_collection(MyObject)
```

Pinecone:

```python
from vectordb_orm import PineconeBackend, VectorSession

# Instantiate a Pinecone session
session = VectorSession(
    PineconeBackend(
        api_key=getenv("PINECONE_API_KEY"),
        environment=getenv("PINECONE_ENVIRONMENT"),
    )
)
session.create_collection(MyObject)
```

## Embedding Types

We currently support two different types of embeddings: floating point and binary. We distinguish these based on the type signatures of the embedding array.

For binary:

```python
embedding: np.ndarray[np.bool_] = EmbeddingField(
    dim=128,
    index=FLAT()
)
```

For floating point:

```python
embedding: np.ndarray = EmbeddingField(
    dim=128,
    index=BIN_FLAT()
)
```

## Field Types


| Field Type      | Description                                                                                                                                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BaseField       | The `BaseField` provides the ability to add a default value for a given field. This should be used in cases where the more specific field types aren't relevant.                                                                           |
| PrimaryKeyField | The `PrimaryKeyField` is used to specify the primary key of your model, and one is required per class.                                                                                                                                     |
| VarCharField    | The `VarCharField` is used to specify a string field, and the `EmbeddingField` is used to specify a vector field.                                                                                                                          |
| EmbeddingField  | The `EmbeddingField` also supports specifying an index type, which is used to specify the index type for the field. The `EmbeddingField` also supports specifying a dimension, which is used to specify the dimension of the vector field. |

## Installation

To get started with vectordb-orm, simply install the package and its dependencies, then import the necessary modules:

```bash
pip install vectordb-orm
```

Make sure to have a vector database running on your system before connecting. We provide an archive of the [official](https://milvus.io/docs/install_standalone-docker.md) docker-compose that's mainly used for testing Milvus. Pinecone requires your API key and environment parameters.

```bash
git clone https://github.com/piercefreeman/vectordb-orm.git
cd vectordb-orm
docker-compose up -d
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

Please note that vectordb-orm is still a (somewhat large) work in progress. The current implementation focuses on Milvus integration; the goal is to eventually expand support to other vector databases. Contributions and feedback are welcome as we work to improve and expand the capabilities of vectordb-orm.
