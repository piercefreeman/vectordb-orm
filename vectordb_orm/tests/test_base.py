import pytest
from pymilvus import Milvus
from vectordb_orm import MilvusSession
from vectordb_orm.tests.models import MyObject
import numpy as np
from time import sleep
from vectordb_orm import MilvusBase, EmbeddingField, PrimaryKeyField
from vectordb_orm.indexes import IVF_FLAT

def test_create_object(collection):
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    assert my_object.text == 'example'
    assert np.array_equal(my_object.embedding, np.array([1.0] * 128))
    assert my_object.id is None


def test_insert_object(collection, milvus_client: Milvus, session: MilvusSession):
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    my_object.insert(milvus_client)
    assert my_object.id is not None

    collection.flush()
    collection.load()

    # Retrieve the object and ensure the values are equivalent
    results = session.query(MyObject).filter(MyObject.id == my_object.id).all()
    assert len(results) == 1

    result : MyObject = results[0].result
    assert result.text == my_object.text


def test_delete_object(collection, milvus_client: Milvus, session: MilvusSession):
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    my_object.insert(milvus_client)

    collection.flush()
    collection.load()

    results = session.query(MyObject).filter(MyObject.text == "example").all()
    assert len(results) == 1

    my_object.delete(milvus_client)

    # Allow enough time to become consistent
    collection.flush()
    collection.load()
    sleep(1)

    results = session.query(MyObject).filter(MyObject.text == "example").all()
    assert len(results) == 0


def test_invalid_typesignatures(milvus_client: Milvus, session: MilvusSession):
    class TestInvalidObject(MilvusBase):
        """
        An IVF_FLAT can't be used with a boolean embedding type
        """
        __collection_name__ = 'invalid_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=IVF_FLAT(cluster_units=128))

    with pytest.raises(ValueError, match="not compatible with binary vectors"):
        TestInvalidObject._create_collection(milvus_client)
