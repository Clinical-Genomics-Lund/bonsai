import asyncio
import json
from  fastapi.testclient import TestClient


import pytest
from bonsai_api.crud.sample import create_sample
from bonsai_api.crud.user import create_user, oauth2_scheme
from bonsai_api.db import Database, get_db
from bonsai_api.models.sample import PipelineResult, SampleInDatabase
from bonsai_api.models.user import UserInputCreate
from bonsai_api.main import app
from mongomock_motor import AsyncMongoMockClient

from .data import *

DATABASE = "testdb"


@pytest.fixture()
async def mongo_database():
    """Setup Bonsai database instance."""
    db = Database()
    db.client = AsyncMongoMockClient()
    # setup mock database
    db.setup()

    # load basic fixtures
    await db.user_collection.insert_one({
        "username": "admin", "password": "admin",
        "first_name": "Nollan", "last_name": "Nollsson",
        "email": "palceholder@email.com", "roles": ["admin"],
    })
    return db


@pytest.fixture(scope="function")
def mtuberculosis_sample(mtuberculosis_sample_path):
    """Sample db object."""
    with open(mtuberculosis_sample_path) as inpt:
        sample_obj = SampleInDatabase(**json.load(inpt))
    return sample_obj


@pytest.fixture(scope="function")
async def sample_database(mongo_database, mtuberculosis_sample_path):
    """Returns a database client with loaded test data."""

    # read fixture and add to database
    with open(mtuberculosis_sample_path) as inpt:
        data = PipelineResult(**json.load(inpt))
        # create sample in database
        sample = await create_sample(db=mongo_database, sample=data)
        return mongo_database


@pytest.fixture()
def fastapi_client(sample_database):
    """Setup API test client."""
    # disable authentication for test client
    app.dependency_overrides[oauth2_scheme] = lambda: ""

    # use mocked mongo database
    app.dependency_overrides[get_db] = lambda: sample_database

    client = TestClient(app)

    return client