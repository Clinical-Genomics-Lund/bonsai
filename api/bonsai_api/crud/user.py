"""User CRUD operations."""
import logging
from typing import List, Annotated

from fastapi import Depends, HTTPException, Security, status, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt

from ..auth import get_password_hash, verify_password
from ..config import ALGORITHM, USER_ROLES, settings
from ..db import Database, get_db
from ..extensions.ldap_extension import ldap_connection
from ..models.auth import TokenData
from ..models.user import (
    SampleBasketObject,
    UserInputCreate,
    UserInputDatabase,
    UserOutputDatabase,
)
from .errors import EntryNotFound, UpdateDocumentError

LOG = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", scopes={})


async def get_user(db_obj: Database, username: str) -> UserInputDatabase:
    """Get user from database"""

    users = await get_users(db_obj, usernames=[username])
    if len(users) == 0:
        raise EntryNotFound(username)
    return users[0]


async def delete_user(db_obj: Database, username: str):
    """Delete user from the database"""
    resp = await db_obj.user_collection.delete_one({"username": username})
    if resp.deleted_count == 0:
        raise EntryNotFound(username)
    return username


async def update_user(db_obj: Database, username: str, user: UserInputCreate):
    """Delete user from the database"""
    # get old user object
    new_user_info = user.model_dump()
    if len(user.password) > 0:
        # create hash for new password
        LOG.info("Changed password for %s", username)
        new_user_info["hashed_password"] = get_password_hash(user.password)

    user_in_db = await get_user(db_obj, username=username)
    # update changed fields in created user object
    upd_user_info = user_in_db.model_copy(update=new_user_info)
    resp = await db_obj.user_collection.replace_one(
        {"username": username}, upd_user_info.model_dump()
    )
    if resp.matched_count == 0:
        raise EntryNotFound(username)
    if resp.modified_count == 0:
        raise UpdateDocumentError(username)


async def create_user(db_obj: Database, user: UserInputCreate) -> UserOutputDatabase:
    """Create new user in the database."""
    # create hash for password
    hashed_password = get_password_hash(user.password)
    user_db_fmt: UserInputDatabase = UserInputDatabase(
        hashed_password=hashed_password, **user.model_dump()
    )
    # store data in database
    resp_obj = await db_obj.user_collection.insert_one(user_db_fmt.model_dump())
    inserted_id = resp_obj.inserted_id
    user_obj = UserInputDatabase(
        id=str(inserted_id),
        **user_db_fmt.model_dump(),
    )
    return user_obj


async def get_users(
    db_obj: Database, usernames: List[str] | None = None
) -> List[UserInputDatabase]:
    """Get multiple users by username from database."""
    query = {}
    if usernames is not None:
        query["username"] = {"$in": usernames}

    upd_user_obj = []
    async for user in db_obj.user_collection.find(query):
        inserted_id = user["_id"]
        upd_user_obj.append(UserInputDatabase(id=str(inserted_id), **user))
    return upd_user_obj


async def authenticate_user(db_obj: Database, username: str, password: str) -> bool:
    """Authenticate user login atempt."""
    try:
        user: UserInputDatabase = await get_user(db_obj, username)
    except EntryNotFound:
        return False

    # use authenticate users with an LDAP server
    if settings.use_ldap_auth:
        is_authenticated = ldap_connection.authenticate(username, password)
    else:
        # use username and password to authenticate users
        is_authenticated = verify_password(password, user.hashed_password)
    return is_authenticated


async def get_current_user(
    security_scopes: SecurityScopes, 
    #token: Annotated[str, Depends(oauth2_scheme)],
    #token: str = Depends(oauth2_scheme),
    token: str = "",
    db: Database = Depends(get_db),
) -> UserOutputDatabase | None:
    """Get current user."""
    # check if API authentication is disabled
    if not settings.api_authentication:
        return None

    if security_scopes.scopes:
        authenticate_value = f"Bearer scope={security_scopes.scope_str}"
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credentials could not be validated",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as error:
        raise credentials_exception from error

    try:
        user: UserOutputDatabase = await get_user(db, username=token_data.username)
    except EntryNotFound as error:
        raise credentials_exception from error
    for scope in security_scopes.scopes:
        users_all_permissions = {
            perm for user_role in user.roles for perm in USER_ROLES.get(user_role, [])
        }
        if not scope in users_all_permissions:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credentials could not be validated",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user


async def get_current_active_user(
    db: Database = Depends(get_db),
    current_user: UserOutputDatabase = Security(get_current_user, scopes=["users:me"]),
) -> UserOutputDatabase | None:
    """Get current active user."""
    # disable API authentication
    if not settings.api_authentication:
        return None

    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_samples_in_user_basket(
    db: Database = Depends(get_db),
    current_user: UserOutputDatabase = Security(get_current_user, scopes=["users:me"]),
) -> List[SampleBasketObject]:
    """Get samples in the current user basket."""
    user: UserOutputDatabase = await get_user(db, username=current_user.username)
    return user.basket


async def add_samples_to_user_basket(
    current_user: UserOutputDatabase,
    sample_ids: List[SampleBasketObject],
    db: Database = Depends(get_db),
) -> SampleBasketObject:
    """Add samples to the basket of the current user."""
    update_obj = await db.user_collection.update_one(
        {"username": current_user.username},
        {
            "$addToSet": {
                "basket": {
                    "$each": jsonable_encoder(sample_ids, by_alias=False),
                },
            },
        },
    )

    # verify update
    if not update_obj.matched_count == 1:
        raise EntryNotFound(current_user.username)

    if not update_obj.modified_count == 1:
        raise UpdateDocumentError(current_user.username)

    # get updated object
    user: UserOutputDatabase = await get_user(db, username=current_user.username)
    return user.basket


async def remove_samples_from_user_basket(
    sample_ids: List[SampleBasketObject],
    db: Database = Depends(get_db),
    current_user: UserOutputDatabase = Security(get_current_user, scopes=["users:me"]),
) -> SampleBasketObject:
    """Remove samples to the basket of the current user."""
    update_obj = await db.user_collection.update_one(
        {"username": current_user.username},
        {
            "$pull": {
                "basket": {"sample_id": {"$in": sample_ids}},
            },
        },
    )

    # verify update
    if not update_obj.matched_count == 1:
        raise EntryNotFound(current_user.username)

    if not update_obj.modified_count == 1:
        sample_ids_in_basket = await db.user_collection.find_one(
            {"username": current_user.username}, {"basket": 1}
        )
        LOG.error(
            "try removing samples %s from basket. Content: %s",
            sample_ids,
            sample_ids_in_basket,
        )
        raise UpdateDocumentError(
            f"Error when updating the basket of {current_user.username}"
        )
    # get updated basket
    user: UserOutputDatabase = await get_user(db, username=current_user.username)
    return user.basket
