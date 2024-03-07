import asyncio
import pytest
import json
from mongomock_motor import AsyncMongoMockClient

from .data import *
from app.crud.sample import create_sample
from app.db import Database
from app.models.sample import SampleInDatabase, PipelineResult

DATABASE = "testdb"


@pytest.fixture()
def mongo_database():
    """Setup Bonsai database instance."""
    db = Database()
    db.client = AsyncMongoMockClient()
    # setup mock database
    db.setup()
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