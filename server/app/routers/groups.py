"""Entrypoints for getting group data."""

from typing import List

from fastapi import (APIRouter, Depends, File, HTTPException, Path, Security,
                     status)
from pymongo.errors import DuplicateKeyError

from ..crud.errors import EntryNotFound, UpdateDocumentError
from ..crud.group import append_sample_to_group
from ..crud.group import create_group as create_group_record
from ..crud.group import get_group, get_groups
from ..crud.user import get_current_active_user
from ..db import db
from ..models.group import GroupInCreate, GroupInfoDatabase
from ..models.user import UserOutputDatabase

router = APIRouter()

DEFAULT_TAGS = [
    "groups",
]
READ_PERMISSION = "groups:read"
WRITE_PERMISSION = "groups:write"


@router.get("/groups/", response_model=List[GroupInfoDatabase], tags=DEFAULT_TAGS)
async def get_groups_in_db(
    current_user: UserOutputDatabase = Security(
        get_current_active_user, scopes=[READ_PERMISSION]
    )
):
    """Get information of the number of samples per group loaded into the database."""
    groups = await get_groups(db)
    return groups


@router.post(
    "/groups/",
    response_model=GroupInfoDatabase,
    status_code=status.HTTP_201_CREATED,
    tags=DEFAULT_TAGS,
)
async def create_group(
    group_info: GroupInCreate,
    current_user: UserOutputDatabase = Security(
        get_current_active_user, scopes=[WRITE_PERMISSION]
    ),
):
    """Create a new group document in the database"""
    # cast input information as group db object
    try:
        result = await create_group_record(db, group_info)
    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.details["errmsg"],
        )
    return result


@router.get(
    "/groups/{group_id}",
    response_model=GroupInfoDatabase,
    tags=DEFAULT_TAGS,
)
async def get_group_in_db(
    group_id: str,
    current_user: UserOutputDatabase = Security(
        get_current_active_user, scopes=[READ_PERMISSION]
    ),
):
    """Get information of the number of samples per group loaded into the database."""
    group = await get_group(db, group_id)
    return group


@router.post(
    "/groups/{group_id}/image", status_code=status.HTTP_200_OK, tags=["groups"]
)
async def update_image_for_group(
    group_id: str,
    image: bytes = File(...),
    current_user: UserOutputDatabase = Security(
        get_current_active_user, scopes=[WRITE_PERMISSION]
    ),
):
    """Add an impage to a group."""
    # cast input information as group db object
    return {"id": group_id, "file_size": len(image)}


@router.put(
    "/groups/{group_id}/sample", status_code=status.HTTP_200_OK, tags=["groups"]
)
async def add_sample_to_group(
    sample_id: str,
    group_id: str = Path(..., tilte="The id of the group to get"),
    current_user: UserOutputDatabase = Security(
        get_current_active_user, scopes=[WRITE_PERMISSION]
    ),
):
    """Add one or more samples to a group"""
    # cast input information as group db object
    try:
        await append_sample_to_group(db, sample_id, group_id)
    except EntryNotFound as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    except UpdateDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_304_NOT_MODIFIED,
            detail=str(error),
        )
