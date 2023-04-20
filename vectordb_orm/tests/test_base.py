import pytest
from vectordb_orm import VectorSession
from vectordb_orm.tests.models import MyObject
import numpy as np
from time import sleep
from vectordb_orm import VectorSchemaBase, EmbeddingField, PrimaryKeyField
from vectordb_orm.indexes import IVF_FLAT

def test_create_object():
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    assert my_object.text == 'example'
    assert np.array_equal(my_object.embedding, np.array([1.0] * 128))
    assert my_object.id is None


def test_insert_object(session: VectorSession):
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    session.insert(my_object)
    assert my_object.id is not None

    session.flush(MyObject)
    session.load(MyObject)

    # Retrieve the object and ensure the values are equivalent
    results = session.query(MyObject).filter(MyObject.id == my_object.id).all()
    assert len(results) == 1

    result : MyObject = results[0].result
    assert result.text == my_object.text


def test_delete_object(session: VectorSession):
    my_object = MyObject(text='example', embedding=np.array([1.0] * 128))
    session.insert(my_object)

    session.flush(MyObject)
    session.load(MyObject)

    results = session.query(MyObject).filter(MyObject.text == "example").all()
    assert len(results) == 1

    session.delete(my_object)

    session.flush(MyObject)
    session.load(MyObject)
    # Allow enough time to become consistent
    #sleep(1)

    results = session.query(MyObject).filter(MyObject.text == "example").all()
    assert len(results) == 0


def test_invalid_typesignatures(session: VectorSession):
    class TestInvalidObject(VectorSchemaBase):
        """
        An IVF_FLAT can't be used with a boolean embedding type
        """
        __collection_name__ = 'invalid_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=IVF_FLAT(cluster_units=128))

    with pytest.raises(ValueError, match="not compatible with binary vectors"):
        session.create_collection(TestInvalidObject)
