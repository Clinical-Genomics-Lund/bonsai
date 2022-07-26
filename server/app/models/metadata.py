"""Metadata models."""
from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from .base import RWModel


class SoupType(Enum):
    DB = "database"
    SW = "software"


class SoupVersion(RWModel):
    """Version of Software of Unknown Provenance."""

    name: str
    version: str
    type: SoupType


class RunInformation(RWModel):
    """Store information on a run how the run was conducted."""

    pipeline: str
    version: str
    commit: str
    analysis_profile: str = Field(..., alias="analysisProfile")
    configuration_files: List[str] = Field(..., alias="configurationFiles")
    run: str
    command: str
    date: datetime


SoupVersions = List[SoupVersion]


class RunMetadata(BaseModel):
    """Run metadata"""

    run: RunInformation
    databases: SoupVersions
