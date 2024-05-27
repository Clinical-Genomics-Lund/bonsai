"""Module for setting up the redis queue and scheduling jobs."""

from enum import Enum

from ..models.base import RWModel
from ..models.cluster import DistanceMethod, TypingMethod
from .queue import redis


class SubmittedJob(RWModel):  # pylint: disable=too-few-public-methods
    """Container for submitted jobs."""

    id: str
    task: str


class ClusterMethod(Enum):  # pylint: disable=too-few-public-methods
    """Index of methods for hierarchical clustering of samples."""

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"


class MsTreeMethods(Enum):  # pylint: disable=too-few-public-methods
    """Valid cluter methods."""

    MSTREE_V1 = "MSTree"
    MSTREE_V2 = "MSTreeV2"
    NEIGHBOR_JOINING = "NJ"
    RAPID_NJ = "RapidNJ"
    NINJA = "ninja"
