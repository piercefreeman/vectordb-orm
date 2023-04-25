from typing import Type

import numpy as np
import pytest

from vectordb_orm import (EmbeddingField, PrimaryKeyField, VectorSchemaBase,
                          VectorSession)
from vectordb_orm.backends.milvus.indexes import Milvus_IVF_FLAT
from vectordb_orm.tests.conftest import SESSION_MODEL_PAIRS


@pytest.mark.parametrize("session,model", SESSION_MODEL_PAIRS)
def test_create_object(session: str, model: Type[VectorSchemaBase]):
    my_object = model(text='example', embedding=np.array([1.0] * 128))
    assert my_object.text == 'example'
    assert np.array_equal(my_object.embedding, np.array([1.0] * 128))
    assert my_object.id is None


@pytest.mark.parametrize("session,model", SESSION_MODEL_PAIRS)
def test_from_dict(session: str, model: Type[VectorSchemaBase]):
    my_object = model.from_dict(
        {
            "text": 'example',
            "embedding": np.array([1.0] * 128)
        }
    )
    assert my_object.text == 'example'
    assert np.array_equal(my_object.embedding, np.array([1.0] * 128))
    assert my_object.id is None


@pytest.mark.parametrize("session,model", SESSION_MODEL_PAIRS)
def test_insert_object(session: str, model: Type[VectorSchemaBase], request):
    session : VectorSession = request.getfixturevalue(session)

    my_object = model(text='example', embedding=np.array([1.0] * 128))
    session.insert(my_object)
    assert my_object.id is not None

    session.flush(model)
    session.load(model)

    # Retrieve the object and ensure the values are equivalent
    results = session.query(model).filter(model.id == my_object.id).all()
    assert len(results) == 1

    result : model = results[0].result
    assert result.text == my_object.text

@pytest.mark.parametrize("session,model", SESSION_MODEL_PAIRS)
def test_insert_batch(session: str, model: Type[VectorSchemaBase], request):
    session : VectorSession = request.getfixturevalue(session)

    obj1 = model(text='example1', embedding=np.array([1.0] * 128))
    obj2 = model(text='example2', embedding=np.array([2.0] * 128))
    obj3 = model(text='example3', embedding=np.array([3.0] * 128))

    session.insert_batch([obj1, obj2, obj3])

    for obj in [obj1, obj2, obj3]:
        assert obj.id is not None

@pytest.mark.parametrize("session,model", SESSION_MODEL_PAIRS)
def test_delete_object(session: str, model: Type[VectorSchemaBase],request):
    session : VectorSession = request.getfixturevalue(session)

    my_object = model(text='example', embedding=np.array([1.0] * 128))
    session.insert(my_object)

    session.flush(model)
    session.load(model)

    results = session.query(model).filter(model.text == "example").all()
    assert len(results) == 1

    session.delete(my_object)

    session.flush(model)
    session.load(model)
    # Allow enough time to become consistent
    #sleep(1)

    results = session.query(model).filter(model.text == "example").all()
    assert len(results) == 0


def test_milvus_invalid_typesignatures(milvus_session: VectorSession):
    class TestInvalidObject(VectorSchemaBase):
        """
        An IVF_FLAT can't be used with a boolean embedding type
        """
        __collection_name__ = 'invalid_collection'

        id: int = PrimaryKeyField()
        embedding: np.ndarray[np.bool_] = EmbeddingField(dim=128, index=Milvus_IVF_FLAT(cluster_units=128))

    with pytest.raises(ValueError, match="not compatible with binary vectors"):
        milvus_session.create_collection(TestInvalidObject)


def test_mixed_configuration_fields(milvus_session: VectorSession):
    """
    Confirm that schema definitions can mix typehints that have configuration values
    and those that only have vanilla markup.

    """
    class TestMixedConfigurationObject(VectorSchemaBase):
        __collection_name__ = 'mixed_configuration_collection'

        id: int = PrimaryKeyField()
        is_valid: bool
        embedding: np.ndarray = EmbeddingField(dim=128, index=Milvus_IVF_FLAT(cluster_units=128))

    milvus_session.create_collection(TestMixedConfigurationObject)
